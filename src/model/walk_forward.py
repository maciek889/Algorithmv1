"""Walk-forward validation with embargo and time-decay sample weights."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import yaml

from src.data.exceptions import ConfigError, ValidationError

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HORIZON_CONFIG_PATH = REPO_ROOT / "config" / "horizon.yaml"


@dataclass(frozen=True)
class WalkForwardFold:
    """One walk-forward fold with sklearn-compatible integer positions."""

    fold_id: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_weights: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    embargo_start: pd.Timestamp
    embargo_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass(frozen=True)
class WalkForwardSplitter:
    """Expanding-window splitter for forward-looking financial labels.

    Test folds are calendar chunks, defaulting to quarter-sized windows.
    Training data expands up to ``test_start - horizon_days`` for each fold.
    """

    horizon_days: int
    initial_train_years: int = 4
    test_months: int = 3
    half_life_days: float = 365.25
    min_weight: float = 0.1
    min_train_observations: int = 252 * 3
    align_to_calendar_quarters: bool = True

    @classmethod
    def from_horizon_config(
        cls,
        path: Path = DEFAULT_HORIZON_CONFIG_PATH,
        **kwargs: object,
    ) -> "WalkForwardSplitter":
        """Create a splitter using ``horizon_days`` from horizon.yaml."""
        return cls(horizon_days=load_horizon_days(path), **kwargs)

    def split(self, X: pd.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Yield ``(train_indices, test_indices, train_weights)`` per fold."""
        for fold in self.iter_folds(X):
            yield fold.train_indices, fold.test_indices, fold.train_weights

    def iter_folds(self, X: pd.DataFrame) -> Iterator[WalkForwardFold]:
        """Yield fold metadata plus indices and sample weights."""
        self._validate_params()
        index = _validated_datetime_index(X)

        first_possible_test_start = (
            index.min()
            + pd.DateOffset(years=self.initial_train_years)
            + pd.Timedelta(days=self.horizon_days)
        )
        period_start = self._initial_test_period_start(pd.Timestamp(first_possible_test_start))
        max_date = index.max()
        positions = np.arange(len(index), dtype=int)

        fold_id = 0
        while period_start <= max_date:
            period_end = period_start + pd.DateOffset(months=self.test_months)
            test_mask = (index >= period_start) & (index < period_end)
            test_indices = positions[test_mask]

            if len(test_indices) > 0:
                test_dates = index[test_indices]
                test_start = pd.Timestamp(test_dates.min())
                train_cutoff = test_start - pd.Timedelta(days=self.horizon_days)
                train_mask = index <= train_cutoff
                train_indices = positions[train_mask]

                if len(train_indices) >= self.min_train_observations:
                    train_dates = index[train_indices]
                    train_end = pd.Timestamp(train_dates.max())
                    if test_start - train_end < pd.Timedelta(days=self.horizon_days):
                        raise ValidationError(
                            "Embargo violation: "
                            f"train_end={train_end.date()} test_start={test_start.date()} "
                            f"horizon_days={self.horizon_days}"
                        )

                    train_weights = self.calculate_time_decay_weights(train_dates)
                    yield WalkForwardFold(
                        fold_id=fold_id,
                        train_indices=train_indices,
                        test_indices=test_indices,
                        train_weights=train_weights,
                        train_start=pd.Timestamp(train_dates.min()),
                        train_end=train_end,
                        embargo_start=train_end + pd.Timedelta(days=1),
                        embargo_end=test_start - pd.Timedelta(days=1),
                        test_start=test_start,
                        test_end=pd.Timestamp(test_dates.max()),
                    )
                    fold_id += 1

            period_start = period_end

    def calculate_time_decay_weights(self, train_dates: pd.DatetimeIndex) -> np.ndarray:
        """Return exponentially decayed weights, clipped to ``min_weight``.

        Formula:
            ``weight_i = max(min_weight, 0.5 ** (age_days_i / half_life_days))``

        where ``age_days_i`` is the calendar-day distance from the newest
        training observation in the fold.
        """
        self._validate_params()
        if len(train_dates) == 0:
            return np.array([], dtype=float)
        if not isinstance(train_dates, pd.DatetimeIndex):
            raise ValidationError("train_dates must be a pandas DatetimeIndex.")

        newest = train_dates.max()
        age_days = ((newest - train_dates) / pd.Timedelta(days=1)).to_numpy(dtype=float)
        weights = np.power(0.5, age_days / self.half_life_days)
        return np.clip(weights, self.min_weight, 1.0).astype(float)

    def _initial_test_period_start(self, first_possible_test_start: pd.Timestamp) -> pd.Timestamp:
        if not self.align_to_calendar_quarters:
            return first_possible_test_start.normalize()

        quarter_month = ((first_possible_test_start.month - 1) // 3) * 3 + 1
        quarter_start = pd.Timestamp(
            year=first_possible_test_start.year,
            month=quarter_month,
            day=1,
        )
        if quarter_start < first_possible_test_start.normalize():
            quarter_start = quarter_start + pd.DateOffset(months=3)
        return pd.Timestamp(quarter_start)

    def _validate_params(self) -> None:
        if self.horizon_days < 1:
            raise ValidationError("horizon_days must be positive.")
        if self.initial_train_years < 0:
            raise ValidationError("initial_train_years cannot be negative.")
        if self.test_months < 1:
            raise ValidationError("test_months must be positive.")
        if self.half_life_days <= 0:
            raise ValidationError("half_life_days must be positive.")
        if not 0 < self.min_weight <= 1:
            raise ValidationError("min_weight must be in the interval (0, 1].")
        if self.min_train_observations < 1:
            raise ValidationError("min_train_observations must be positive.")


def load_horizon_days(path: Path = DEFAULT_HORIZON_CONFIG_PATH) -> int:
    """Load the calibrated embargo horizon from config/horizon.yaml."""
    if not path.exists():
        raise ConfigError(f"Horizon config not found: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError(f"Malformed YAML at {path}: {e}") from e

    try:
        horizon_days = int(payload["horizon_days"])
    except (KeyError, TypeError, ValueError) as e:
        raise ConfigError(f"Horizon config missing integer horizon_days: {e}") from e
    if horizon_days < 1:
        raise ConfigError("horizon_days must be positive.")
    return horizon_days


def _validated_datetime_index(X: pd.DataFrame) -> pd.DatetimeIndex:
    if not isinstance(X.index, pd.DatetimeIndex):
        raise ValidationError("Walk-forward input must use a DatetimeIndex.")
    if X.empty:
        raise ValidationError("Walk-forward input cannot be empty.")
    if X.index.has_duplicates:
        raise ValidationError("Walk-forward input index cannot contain duplicate dates.")
    if not X.index.is_monotonic_increasing:
        raise ValidationError("Walk-forward input index must be sorted ascending.")
    return X.index
