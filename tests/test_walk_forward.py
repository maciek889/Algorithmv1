"""Tests for Stage 1.4 walk-forward validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.exceptions import ConfigError, ValidationError
from src.model.walk_forward import WalkForwardSplitter, load_horizon_days


def test_walk_forward_embargo_is_enforced_for_every_fold() -> None:
    idx = pd.date_range("2020-01-01", "2022-12-31", freq="B")
    df = pd.DataFrame({"feature": np.arange(len(idx))}, index=idx)
    splitter = WalkForwardSplitter(
        horizon_days=14,
        initial_train_years=1,
        test_months=3,
        half_life_days=365.25,
        min_train_observations=10,
    )

    folds = list(splitter.iter_folds(df))

    assert folds
    for fold in folds:
        assert (fold.test_start - fold.train_end).days >= 14
        assert fold.train_end < fold.test_start
        assert set(fold.train_indices).isdisjoint(set(fold.test_indices))
        assert len(fold.train_weights) == len(fold.train_indices)


def test_split_returns_sklearn_compatible_tuple() -> None:
    idx = pd.date_range("2020-01-01", "2021-12-31", freq="B")
    df = pd.DataFrame({"feature": np.arange(len(idx))}, index=idx)
    splitter = WalkForwardSplitter(
        horizon_days=7,
        initial_train_years=1,
        test_months=3,
        min_train_observations=10,
    )

    train_indices, test_indices, train_weights = next(splitter.split(df))

    assert isinstance(train_indices, np.ndarray)
    assert isinstance(test_indices, np.ndarray)
    assert isinstance(train_weights, np.ndarray)
    assert len(train_weights) == len(train_indices)


def test_time_decay_weights_follow_half_life_formula() -> None:
    train_dates = pd.DatetimeIndex(["2024-01-01", "2024-01-11", "2024-01-21"])
    splitter = WalkForwardSplitter(
        horizon_days=1,
        half_life_days=10.0,
        min_weight=0.1,
        min_train_observations=1,
    )

    weights = splitter.calculate_time_decay_weights(train_dates)

    np.testing.assert_allclose(weights, np.array([0.25, 0.5, 1.0]))


def test_time_decay_weights_respect_floor() -> None:
    train_dates = pd.DatetimeIndex(["2023-01-01", "2024-01-01"])
    splitter = WalkForwardSplitter(
        horizon_days=1,
        half_life_days=10.0,
        min_weight=0.1,
        min_train_observations=1,
    )

    weights = splitter.calculate_time_decay_weights(train_dates)

    assert weights[0] == pytest.approx(0.1)
    assert weights[1] == pytest.approx(1.0)


def test_load_horizon_days_reads_config(tmp_path) -> None:
    path = tmp_path / "horizon.yaml"
    path.write_text("horizon_days: 14\n", encoding="utf-8")

    assert load_horizon_days(path) == 14


def test_load_horizon_days_rejects_invalid_value(tmp_path) -> None:
    path = tmp_path / "horizon.yaml"
    path.write_text("horizon_days: 0\n", encoding="utf-8")

    with pytest.raises(ConfigError):
        load_horizon_days(path)


def test_unsorted_datetime_index_is_rejected() -> None:
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-01"])
    df = pd.DataFrame({"feature": [1, 2]}, index=idx)
    splitter = WalkForwardSplitter(horizon_days=1, min_train_observations=1)

    with pytest.raises(ValidationError):
        list(splitter.iter_folds(df))
