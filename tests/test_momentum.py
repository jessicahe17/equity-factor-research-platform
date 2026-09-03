import pandas as pd
import pytest
from equity_factor_research.factors.momentum import (
    add_momentum_signal, 
    add_momentum_eligibility,
    find_momentum_inconsistencies,
    summarize_momentum_diagnostics,
)


def test_add_momentum_signal_uses_12_to_2_window():
    panel = pd.DataFrame(
        {
            "security_id": [10001] * 13,
            "date": pd.date_range(
                "2024-01-31",
                "2025-01-31",
                freq="ME",
            ),
            "total_return": [
                0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                0.01, 0.01, 0.01, 0.01, 0.01,
                0.50,   # Dec 2024 — skipped
                -0.25,  # Jan 2025 — outcome
            ],
        }
    )

    result = add_momentum_signal(panel)

    january_2025 = result.loc[
        result["date"] == pd.Timestamp("2025-01-31"),
        "momentum",
    ].iloc[0]

    expected = (1.01 ** 11) - 1

    assert january_2025 == pytest.approx(expected)


def test_add_momentum_signal_allows_missing_skipped_month():
    dates = list(
        pd.date_range(
            "2024-01-31",
            "2024-11-30",
            freq="ME",
        )
    )
    dates.append(pd.Timestamp("2025-01-31"))

    panel = pd.DataFrame(
        {
            "security_id": [10001] * 12,
            "date": dates,
            "total_return": [0.01] * 11 + [-0.25],
        }
    )

    result = add_momentum_signal(panel)

    january_2025 = result.loc[
        result["date"] == pd.Timestamp("2025-01-31"),
        "momentum",
    ].iloc[0]

    expected = (1.01 ** 11) - 1

    assert january_2025 == pytest.approx(expected)


def test_add_momentum_signal_rejects_missing_required_month():
    dates = list(
        pd.date_range(
            "2024-01-31",
            "2024-10-31",
            freq="ME",
        )
    )
    dates.extend(
        [
            pd.Timestamp("2024-12-31"),
            pd.Timestamp("2025-01-31"),
        ]
    )

    panel = pd.DataFrame(
        {
            "security_id": [10001] * 12,
            "date": dates,
            "total_return": [0.01] * 10 + [0.50, -0.25],
        }
    )

    result = add_momentum_signal(panel)

    january_2025 = result.loc[
        result["date"] == pd.Timestamp("2025-01-31"),
        "momentum",
    ].iloc[0]

    assert pd.isna(january_2025)


def test_add_momentum_signal_rejects_missing_required_return():
    panel = pd.DataFrame(
        {
            "security_id": [10001] * 13,
            "date": pd.date_range(
                "2024-01-31",
                "2025-01-31",
                freq="ME",
            ),
            "total_return": [
                0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                0.01, 0.01, 0.01, 0.01,
                float("nan"),  # Nov 2024 NaN
                0.50,          
                -0.25,         
            ],
        }
    )

    result = add_momentum_signal(panel)

    january_2025 = result.loc[
        result["date"] == pd.Timestamp("2025-01-31"),
        "momentum",
    ].iloc[0]

    assert pd.isna(january_2025)


def test_add_momentum_signal_keeps_securities_separate():
    dates = pd.date_range(
        "2024-01-31",
        "2025-01-31",
        freq="ME",
    )

    panel = pd.DataFrame(
        {
            "security_id": [10001] * 13 + [20002] * 13,
            "date": list(dates) + list(dates),
            "total_return": (
                [0.01] * 11 + [0.50, -0.25]
                + [0.02] * 11 + [-0.40, 0.30]
            ),
        }
    )

    result = add_momentum_signal(panel)

    momentum_10001 = result.loc[
        (result["security_id"] == 10001)
        & (result["date"] == pd.Timestamp("2025-01-31")),
        "momentum",
    ].iloc[0]

    momentum_20002 = result.loc[
        (result["security_id"] == 20002)
        & (result["date"] == pd.Timestamp("2025-01-31")),
        "momentum",
    ].iloc[0]

    assert momentum_10001 == pytest.approx((1.01 ** 11) - 1)
    assert momentum_20002 == pytest.approx((1.02 ** 11) - 1)


def test_add_momentum_signal_handles_unsorted_input():
    panel = pd.DataFrame(
        {
            "security_id": [10001] * 13,
            "date": pd.date_range(
                "2024-01-31",
                "2025-01-31",
                freq="ME",
            ),
            "total_return": [
                0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                0.01, 0.01, 0.01, 0.01, 0.01,
                0.50,
                -0.25,
            ],
        }
    )

    panel = panel.sample(
        frac=1,
        random_state=42,
    ).reset_index(drop=True)

    result = add_momentum_signal(panel)

    january_2025 = result.loc[
        result["date"] == pd.Timestamp("2025-01-31"),
        "momentum",
    ].iloc[0]

    assert january_2025 == pytest.approx((1.01 ** 11) - 1)


def test_add_momentum_signal_does_not_mutate_input():
    panel = pd.DataFrame(
        {
            "security_id": [10001] * 13,
            "date": pd.date_range(
                "2024-01-31",
                "2025-01-31",
                freq="ME",
            ),
            "total_return": [
                0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                0.01, 0.01, 0.01, 0.01, 0.01,
                0.50,
                -0.25,
            ],
        }
    )

    original = panel.copy(deep=True)

    add_momentum_signal(panel)

    pd.testing.assert_frame_equal(panel, original)


def test_add_momentum_signal_allows_minus_one_return():
    panel = pd.DataFrame(
        {
            "security_id": [10001] * 13,
            "date": pd.date_range(
                "2024-01-31",
                "2025-01-31",
                freq="ME",
            ),
            "total_return": [
                0.01, 0.01, 0.01, 0.01, 0.01,
                -1.00,  # required historical month
                0.01, 0.01, 0.01, 0.01, 0.01,
                0.50,   # Dec 2024
                -0.25,  # Jan 2025
            ],
        }
    )

    result = add_momentum_signal(panel)

    january_2025 = result.loc[
        result["date"] == pd.Timestamp("2025-01-31"),
        "momentum",
    ].iloc[0]

    assert january_2025 == pytest.approx(-1.0)


def test_add_momentum_eligibility_with_complete_history():
    panel = pd.DataFrame(
        {
            "security_id": [10001] * 13,
            "date": pd.date_range(
                "2024-01-31",
                "2025-01-31",
                freq="ME",
            ),
            "total_return": [0.01] * 13,
        }
    )

    result = add_momentum_eligibility(panel)

    eligible = result.loc[
        result["date"] == pd.Timestamp("2025-01-31"),
        "momentum_eligible",
    ].iloc[0]

    assert eligible


def test_add_momentum_eligibility_allows_skipped_month():
    dates = list(
        pd.date_range(
            "2024-01-31",
            "2024-11-30",
            freq="ME",
        )
    )
    dates.append(pd.Timestamp("2025-01-31"))
    
    panel = pd.DataFrame(
        {
            "security_id": [10001] * 12,
            "date": dates,
            "total_return": [0.01] * 11 + [-0.25],
        }
    )

    result = add_momentum_eligibility(panel)
    
    eligible = result.loc[
        result["date"] == pd.Timestamp("2025-01-31"),
        "momentum_eligible",
    ].iloc[0]
    
    assert eligible


def test_add_momentum_eligibility_rejects_missing_required_month():
    dates = list(
        pd.date_range(
            "2024-01-31",
            "2024-10-31",
            freq="ME",
        )
    )
    dates.extend(
        [
            pd.Timestamp("2024-12-31"),
            pd.Timestamp("2025-01-31"),
        ]
    )

    panel = pd.DataFrame(
        {
            "security_id": [10001] * 12,
            "date": dates,
            "total_return": [0.01] * 10 + [0.50, -0.25],
        }
    )
    result = add_momentum_eligibility(panel)
    
    eligible = result.loc[
        result["date"] == pd.Timestamp("2025-01-31"),
        "momentum_eligible",
    ].iloc[0]
    
    assert not eligible


def test_add_momentum_eligibility_rejects_missing_required_return():
    panel = pd.DataFrame(
        {
            "security_id": [10001] * 13,
            "date": pd.date_range(
                "2024-01-31",
                "2025-01-31",
                freq="ME",
            ),
            "total_return": [
                0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                0.01, 0.01, 0.01, 0.01,
                float("nan"),  # Nov 2024 NaN
                0.50,          
                -0.25,         
            ],
        }
    )
    result = add_momentum_eligibility(panel)

    eligible = result.loc[
        result["date"] == pd.Timestamp("2025-01-31"),
        "momentum_eligible",
    ].iloc[0]

    assert not eligible


def test_add_momentum_eligibility_allows_missing_outcome_return():
    panel = pd.DataFrame(
        {
            "security_id": [10001] * 13,
            "date": pd.date_range(
                "2024-01-31",
                "2025-01-31",
                freq="ME",
            ),
            "total_return": [
                0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                0.01, 0.01, 0.01, 0.01, 0.01,
                0.50,   
                float("nan"),  # Jan 2025 NaN
            ],
        }
    )
    result = add_momentum_eligibility(panel)
    
    eligible = result.loc[
        result["date"] == pd.Timestamp("2025-01-31"),
        "momentum_eligible",
    ].iloc[0]
    
    assert eligible


def test_find_momentum_inconsistencies():
    panel = pd.DataFrame(
        {
            "momentum_eligible": [
                True,
                False,
                True,
                False,
            ],
            "momentum": [
                0.20,
                float("nan"),
                float("nan"),
                0.10,
            ],
        }
    )

    result = find_momentum_inconsistencies(panel)
    expected = pd.Series([False, False, True, True])
    pd.testing.assert_series_equal(result, expected)


def test_summarize_momentum_diagnostics():
    panel = pd.DataFrame(
        {
            "security_id": [
                10001, 10002, 10003,
                10001, 10002, 10003,
            ],
            "date": [
                pd.Timestamp("2025-01-31"),
                pd.Timestamp("2025-01-31"),
                pd.Timestamp("2025-01-31"),
                pd.Timestamp("2025-02-28"),
                pd.Timestamp("2025-02-28"),
                pd.Timestamp("2025-02-28"),
            ],
            "in_universe": [
                True, True, False,
                True, True, True,
            ],
            "momentum_eligible": [
                True, False, True,
                True, True, False,
            ],
            "momentum": [
                0.10, float("nan"), 0.90,
                0.20, 0.40, float("nan"),
            ],
        }
    )

    result = summarize_momentum_diagnostics(panel)

    january = result.loc[
        result["date"] == pd.Timestamp("2025-01-31")
    ].iloc[0]

    february = result.loc[
        result["date"] == pd.Timestamp("2025-02-28")
    ].iloc[0]

    assert january["universe_count"] == 2
    assert january["eligible_count"] == 1
    assert january["coverage"] == pytest.approx(0.5)
    assert january["momentum_mean"] == pytest.approx(0.10)
    assert january["momentum_median"] == pytest.approx(0.10)

    assert february["universe_count"] == 3
    assert february["eligible_count"] == 2
    assert february["coverage"] == pytest.approx(2 / 3)
    assert february["momentum_mean"] == pytest.approx(0.30)
    assert february["momentum_median"] == pytest.approx(0.30)


def test_summarize_momentum_diagnostics_distribution_statistics():
    panel = pd.DataFrame(
        {
            "security_id": [10001, 10002, 10003, 10004, 10005],
            "date": [pd.Timestamp("2025-01-31")] * 5,
            "in_universe": [True] * 5,
            "momentum_eligible": [True] * 5,
            "momentum": [0.10, 0.20, 0.30, 0.40, 0.50],
        }
    )

    result = summarize_momentum_diagnostics(panel)

    january = result.iloc[0]

    assert january["universe_count"] == 5
    assert january["eligible_count"] == 5
    assert january["coverage"] == pytest.approx(1.0)

    assert january["momentum_mean"] == pytest.approx(0.30)
    assert january["momentum_median"] == pytest.approx(0.30)
    assert january["momentum_std"] == pytest.approx(0.158113883)
    assert january["momentum_p10"] == pytest.approx(0.14)
    assert january["momentum_p90"] == pytest.approx(0.46)


def test_momentum_pipeline_integration():
    dates = pd.date_range(
        "2024-01-31",
        "2025-01-31",
        freq="ME",
    )

    panel_10001 = pd.DataFrame(
        {
            "security_id": [10001] * 13,
            "date": dates,
            "total_return": [0.01] * 11 + [0.50, -0.25],
            "in_universe": [True] * 13,
        }
    )

    panel_20002 = pd.DataFrame(
        {
            "security_id": [20002] * 12,
            "date": [
                date
                for date in dates
                if date != pd.Timestamp("2024-11-30")
            ],
            "total_return": [0.01] * 10 + [0.50, -0.25],
            "in_universe": [True] * 12,
        }
    )

    panel_30003 = pd.DataFrame(
        {
            "security_id": [30003] * 13,
            "date": dates,
            "total_return": [0.02] * 11 + [-0.40, 0.30],
            "in_universe": [False] * 13,
        }
    )

    panel = pd.concat(
        [panel_10001, panel_20002, panel_30003],
        ignore_index=True,
    )

    result = add_momentum_signal(panel)
    result = add_momentum_eligibility(result)

    january_10001 = result.loc[
        (result["security_id"] == 10001)
        & (result["date"] == pd.Timestamp("2025-01-31"))
    ].iloc[0]

    january_20002 = result.loc[
        (result["security_id"] == 20002)
        & (result["date"] == pd.Timestamp("2025-01-31"))
    ].iloc[0]

    january_30003 = result.loc[
        (result["security_id"] == 30003)
        & (result["date"] == pd.Timestamp("2025-01-31"))
    ].iloc[0]

    assert january_10001["momentum_eligible"]
    assert january_10001["momentum"] == pytest.approx(
        (1.01 ** 11) - 1
    )

    assert not january_20002["momentum_eligible"]
    assert pd.isna(january_20002["momentum"])

    assert january_30003["momentum_eligible"]
    assert january_30003["momentum"] == pytest.approx(
        (1.02 ** 11) - 1
    )

    inconsistencies = find_momentum_inconsistencies(result)

    assert not inconsistencies.any()

    diagnostics = summarize_momentum_diagnostics(result)

    january_diagnostics = diagnostics.loc[
        diagnostics["date"] == pd.Timestamp("2025-01-31")
    ].iloc[0]

    assert january_diagnostics["universe_count"] == 2
    assert january_diagnostics["eligible_count"] == 1
    assert january_diagnostics["coverage"] == pytest.approx(0.5)
    assert january_diagnostics["momentum_mean"] == pytest.approx(
        (1.01 ** 11) - 1
    )