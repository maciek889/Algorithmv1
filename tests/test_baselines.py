"""Tests for Stage 1.5 baseline evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.evaluate_baselines import (
    MA50TrendBaseline,
    RandomPriorBaseline,
    _aggregate_metrics,
)


def test_random_prior_baseline_is_deterministic_for_fixed_seed() -> None:
    df = pd.DataFrame(index=range(20))
    first = RandomPriorBaseline(class_one_probability=0.25, seed=7)
    second = RandomPriorBaseline(class_one_probability=0.25, seed=7)

    np.testing.assert_array_equal(first.predict(df), second.predict(df))


def test_ma50_trend_baseline_requires_price_above_ma50_and_rising_ma50() -> None:
    df = pd.DataFrame(
        {
            "price_vs_ma50": [0.01, -0.01, 0.02, 0.00],
            "ma50_slope_5d": [1, 1, 0, 1],
        }
    )

    predictions = MA50TrendBaseline().predict(df)

    np.testing.assert_array_equal(predictions, np.array([1, 0, 0, 0]))


def test_aggregate_metrics_returns_mean_and_sample_std() -> None:
    rows = [
        {"precision_class_1": 0.2, "recall_class_1": 0.4, "f1_class_1": 0.3, "accuracy": 0.6},
        {"precision_class_1": 0.4, "recall_class_1": 0.8, "f1_class_1": 0.5, "accuracy": 0.8},
    ]

    aggregate = _aggregate_metrics(rows)

    assert aggregate["precision_class_1"]["mean"] == 0.30000000000000004
    assert aggregate["precision_class_1"]["std"] == np.std([0.2, 0.4], ddof=1)
    assert aggregate["recall_class_1"]["mean"] == 0.6000000000000001
    assert aggregate["f1_class_1"]["mean"] == 0.4
    assert aggregate["accuracy"]["mean"] == 0.7
