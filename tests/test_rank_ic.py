import pandas as pd
import pytest
import numpy as np

from equity_factor_research.analysis.rank_ic import (
    calculate_monthly_rank_ic,
    summarize_rank_ic,
    summarize_rank_ic_autocorrelation,
    calculate_rank_ic_rolling_diagnostics,
    summarize_rank_ic_by_subperiod,
)
from equity_factor_research.factors.momentum import(
    add_momentum_signal,
    add_momentum_eligibility,
)


def test_calculate_monthly_rank_ic_perfect_positive_relation():
    panel = pd.DataFrame(
        {
            "security_id": [10001, 10002, 10003, 10004],
            "date": [pd.Timestamp("2025-01-31")] * 4,
            "in_universe": [True] * 4,
            "momentum_eligible": [True] * 4,
            "momentum": [-0.20, 0.10, 0.30, 0.80],
            "total_return": [-0.05, 0.01, 0.04, 0.10],
        }
    )

    result = calculate_monthly_rank_ic(panel)

    january = result.iloc[0]

    assert january["date"] == pd.Timestamp("2025-01-31")
    assert january["n_obs"] == 4
    assert january["rank_ic"] == pytest.approx(1.0)


def test_calculate_monthly_rank_ic_perfect_negative_relation():
    panel = pd.DataFrame(
        {
            "security_id": [10001, 10002, 10003, 10004],
            "date": [pd.Timestamp("2025-01-31")] * 4,
            "in_universe": [True] * 4,
            "momentum_eligible": [True] * 4,
            "momentum": [-0.20, 0.10, 0.30, 0.80],
            "total_return": [0.10, 0.04, 0.01, -0.05],
        }
    )

    result = calculate_monthly_rank_ic(panel)

    assert result.iloc[0]["n_obs"] == 4
    assert result.iloc[0]["rank_ic"] == pytest.approx(-1.0)


def test_calculate_monthly_rank_ic_excludes_missing_outcomes():
    panel = pd.DataFrame(
        {
            "security_id": [10001, 10002, 10003],
            "date": [pd.Timestamp("2025-01-31")] * 3,
            "in_universe": [True] * 3,
            "momentum_eligible": [True] * 3,
            "momentum": [0.10, 0.20, 0.30],
            "total_return": [0.01, float("nan"), 0.03],
        }
    )

    result = calculate_monthly_rank_ic(panel)

    assert result.iloc[0]["n_obs"] == 2
    assert result.iloc[0]["rank_ic"] == pytest.approx(1.0)


def test_calculate_monthly_rank_ic_excludes_out_of_universe_stocks():
    panel = pd.DataFrame(
        {
            "security_id": [10001, 10002, 10003],
            "date": [pd.Timestamp("2025-01-31")] * 3,
            "in_universe": [True, True, False],
            "momentum_eligible": [True] * 3,
            "momentum": [0.10, 0.20, 0.90],
            "total_return": [0.01, 0.02, -0.50],
        }
    )

    result = calculate_monthly_rank_ic(panel)

    assert result.iloc[0]["n_obs"] == 2
    assert result.iloc[0]["rank_ic"] == pytest.approx(1.0)


def test_calculate_monthly_rank_ic_excludes_ineligible_stocks():
    panel = pd.DataFrame(
        {
            "security_id": [10001, 10002, 10003],
            "date": [pd.Timestamp("2025-01-31")] * 3,
            "in_universe": [True] * 3,
            "momentum_eligible": [True, True, False],
            "momentum": [0.10, 0.20, 0.90],
            "total_return": [0.01, 0.02, -0.50],
        }
    )

    result = calculate_monthly_rank_ic(panel)

    assert result.iloc[0]["n_obs"] == 2
    assert result.iloc[0]["rank_ic"] == pytest.approx(1.0)


def test_calculate_monthly_rank_ic_handles_multiple_months():
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
            "in_universe": [True] * 6,
            "momentum_eligible": [True] * 6,
            "momentum": [
                0.10, 0.20, 0.30,
                0.10, 0.20, 0.30,
            ],
            "total_return": [
                0.01, 0.02, 0.03,   # positive IC
                0.03, 0.02, 0.01,   # negative IC
            ],
        }
    )

    result = calculate_monthly_rank_ic(panel)

    january = result.loc[
        result["date"] == pd.Timestamp("2025-01-31")
    ].iloc[0]

    february = result.loc[
        result["date"] == pd.Timestamp("2025-02-28")
    ].iloc[0]

    assert january["rank_ic"] == pytest.approx(1.0)
    assert february["rank_ic"] == pytest.approx(-1.0)


def test_calculate_monthly_rank_ic_requires_at_least_two_observations():
    panel = pd.DataFrame(
        {
            "security_id": [10001],
            "date": [pd.Timestamp("2025-01-31")],
            "in_universe": [True],
            "momentum_eligible": [True],
            "momentum": [0.20],
            "total_return": [0.03],
        }
    )

    result = calculate_monthly_rank_ic(panel)

    assert result.iloc[0]["n_obs"] == 1
    assert pd.isna(result.iloc[0]["rank_ic"])


def test_calculate_monthly_rank_ic_returns_nan_for_constant_signal():
    panel = pd.DataFrame(
        {
            "security_id": [10001, 10002, 10003],
            "date": [pd.Timestamp("2025-01-31")] * 3,
            "in_universe": [True] * 3,
            "momentum_eligible": [True] * 3,
            "momentum": [0.20, 0.20, 0.20],   # constant
            "total_return": [0.01, 0.03, -0.02],
        }
    )

    result = calculate_monthly_rank_ic(panel)

    assert result.iloc[0]["n_obs"] == 3
    assert pd.isna(result.iloc[0]["rank_ic"])


def test_calculate_monthly_rank_ic_returns_nan_for_constant_outcome():
    panel = pd.DataFrame(
        {
            "security_id": [10001, 10002, 10003],
            "date": [pd.Timestamp("2025-01-31")] * 3,
            "in_universe": [True] * 3,
            "momentum_eligible": [True] * 3,
            "momentum": [0.10, 0.20, 0.30],
            "total_return": [0.02, 0.02, 0.02],  # constant
        }
    )

    result = calculate_monthly_rank_ic(panel)

    assert result.iloc[0]["n_obs"] == 3
    assert pd.isna(result.iloc[0]["rank_ic"])


def test_summarize_rank_ic():
    monthly_ic = pd.DataFrame(
        {
            "date": pd.date_range(
                "2025-01-31",
                periods=4,
                freq="ME",
            ),
            "rank_ic": [0.10, 0.20, -0.10, 0.30],
            "n_obs": [900, 910, 905, 915],
        }
    )

    result = summarize_rank_ic(monthly_ic)

    expected_mean = 0.125
    expected_median = 0.15
    expected_std = pd.Series([0.10, 0.20, -0.10, 0.30]).std()
    expected_t_stat = (
        expected_mean
        / (expected_std / np.sqrt(4))
    )

    assert result["n_months"] == 4
    assert result["mean_ic"] == pytest.approx(expected_mean)
    assert result["median_ic"] == pytest.approx(expected_median)
    assert result["std_ic"] == pytest.approx(expected_std)
    assert result["positive_ic_fraction"] == pytest.approx(0.75)
    assert result["ic_t_stat"] == pytest.approx(expected_t_stat)


def test_summarize_rank_ic_ignores_missing_months():
    monthly_ic = pd.DataFrame(
        {
            "date": pd.date_range(
                "2025-01-31",
                periods=4,
                freq="ME",
            ),
            "rank_ic": [0.10, np.nan, 0.20, -0.10],
            "n_obs": [900, 0, 910, 905],
        }
    )

    result = summarize_rank_ic(monthly_ic)

    valid = pd.Series([0.10, 0.20, -0.10])

    expected_mean = valid.mean()
    expected_median = valid.median()
    expected_std = valid.std()
    expected_t_stat = (
        expected_mean
        / (expected_std / np.sqrt(3))
    )

    assert result["n_months"] == 3
    assert result["mean_ic"] == pytest.approx(expected_mean)
    assert result["median_ic"] == pytest.approx(expected_median)
    assert result["std_ic"] == pytest.approx(expected_std)
    assert result["positive_ic_fraction"] == pytest.approx(2 / 3)
    assert result["ic_t_stat"] == pytest.approx(expected_t_stat)


def test_summarize_rank_ic_with_one_valid_month():
    monthly_ic = pd.DataFrame(
        {
            "date": [
                pd.Timestamp("2025-01-31"),
                pd.Timestamp("2025-02-28"),
            ],
            "rank_ic": [0.15, np.nan],
            "n_obs": [900, 0],
        }
    )

    result = summarize_rank_ic(monthly_ic)

    assert result["n_months"] == 1
    assert result["mean_ic"] == pytest.approx(0.15)
    assert result["median_ic"] == pytest.approx(0.15)
    assert result["positive_ic_fraction"] == pytest.approx(1.0)

    assert pd.isna(result["std_ic"])
    assert pd.isna(result["ic_t_stat"])


def test_summarize_rank_ic_with_no_valid_months():
    monthly_ic = pd.DataFrame(
        {
            "date": [
                pd.Timestamp("2025-01-31"),
                pd.Timestamp("2025-02-28"),
            ],
            "rank_ic": [np.nan, np.nan],
            "n_obs": [0, 1],
        }
    )

    result = summarize_rank_ic(monthly_ic)

    assert result["n_months"] == 0
    assert pd.isna(result["mean_ic"])
    assert pd.isna(result["median_ic"])
    assert pd.isna(result["std_ic"])
    assert pd.isna(result["positive_ic_fraction"])
    assert pd.isna(result["ic_t_stat"])


def test_summarize_rank_ic_returns_nan_t_stat_for_zero_variance():
    monthly_ic = pd.DataFrame(
        {
            "date": pd.date_range(
                "2025-01-31",
                periods=3,
                freq="ME",
            ),
            "rank_ic": [0.10, 0.10, 0.10],
            "n_obs": [900, 910, 905],
        }
    )

    result = summarize_rank_ic(monthly_ic)

    assert result["n_months"] == 3
    assert result["mean_ic"] == pytest.approx(0.10)
    assert result["median_ic"] == pytest.approx(0.10)
    assert result["std_ic"] == pytest.approx(0.0)
    assert result["positive_ic_fraction"] == pytest.approx(1.0)
    assert pd.isna(result["ic_t_stat"])


def test_summarize_rank_ic_autocorrelation():
    monthly_ic = pd.DataFrame(
        {
            "date": pd.date_range(
                "2025-01-31",
                periods=5,
                freq="ME",
            ),
            "rank_ic": [
                0.10,
                0.20,
                0.30,
                0.40,
                0.50,
            ],
            "n_obs": [900, 910, 905, 920, 915],
        }
    )

    result = summarize_rank_ic_autocorrelation(monthly_ic)

    assert result["n_pairs"] == 4
    assert result["lag1_autocorr"] == pytest.approx(1.0)


def test_summarize_rank_ic_autocorrelation_missing_month():
    monthly_ic = pd.DataFrame(
        {
            "date": [
                pd.Timestamp("2025-01-31"),
                pd.Timestamp("2025-03-31"),
                pd.Timestamp("2025-04-30"),
                pd.Timestamp("2025-05-31"),
            ],
            "rank_ic": [
                0.10,
                0.20,
                0.30,
                0.40,
            ],
            "n_obs": [900, 910, 905, 920],
        }
    )

    result = summarize_rank_ic_autocorrelation(monthly_ic)

    assert result["n_pairs"] == 2
    assert result["lag1_autocorr"] == pytest.approx(1.0)


def test_summarize_rank_ic_autocorrelation_missing_ic_value():
    monthly_ic = pd.DataFrame(
        {
            "date": [
                pd.Timestamp("2025-01-31"),
                pd.Timestamp("2025-02-28"),
                pd.Timestamp("2025-03-31"),
                pd.Timestamp("2025-04-30"),
            ],
            "rank_ic": [
                0.10,
                np.nan,
                0.20,
                0.30,
            ],
            "n_obs": [900, 0, 910, 905],
        }
    )

    result = summarize_rank_ic_autocorrelation(monthly_ic)

    assert result["n_pairs"] == 1
    assert pd.isna(result["lag1_autocorr"])


def test_calculate_rank_ic_rolling_diagnostics():
    monthly_ic = pd.DataFrame(
        {
            "date": pd.date_range(
                "2024-01-31",
                periods=12,
                freq="ME",
            ),
            "rank_ic": [
                0.10, 0.20, -0.10, 0.30,
                0.10, -0.20, 0.20, 0.10,
                0.00, 0.30, -0.10, 0.20,
            ],
        }
    )

    result = calculate_rank_ic_rolling_diagnostics(monthly_ic)

    december = result.iloc[-1]

    values = monthly_ic["rank_ic"]

    assert result.loc[
        :10,
        [
            "rolling_mean_ic_12m",
            "rolling_std_ic_12m",
            "rolling_positive_fraction_12m",
        ],
    ].isna().all().all()

    assert december["rolling_mean_ic_12m"] == pytest.approx(
        values.mean()
    )

    assert december["rolling_std_ic_12m"] == pytest.approx(
        values.std()
    )

    assert december[
        "rolling_positive_fraction_12m"
    ] == pytest.approx(
        (values > 0).mean()
    )


def test_calculate_rank_ic_rolling_diagnostics_missing_calendar_month():
    dates = pd.date_range(
        "2024-01-31",
        periods=13,
        freq="ME",
    ).delete(5)

    monthly_ic = pd.DataFrame(
        {
            "date": dates,
            "rank_ic": [0.10] * 12,
        }
    )

    result = calculate_rank_ic_rolling_diagnostics(monthly_ic)

    june_2024 = result.loc[
        result["date"] == pd.Timestamp("2024-06-30")
    ].iloc[0]

    december_2024 = result.loc[
        result["date"] == pd.Timestamp("2024-12-31")
    ].iloc[0]

    assert pd.isna(june_2024["rolling_mean_ic_12m"])
    assert pd.isna(december_2024["rolling_mean_ic_12m"])
    assert pd.isna(december_2024["rolling_std_ic_12m"])
    assert pd.isna(december_2024["rolling_positive_fraction_12m"])


def test_calculate_rank_ic_rolling_diagnostics_missing_ic():
    monthly_ic = pd.DataFrame(
        {
            "date": pd.date_range(
                "2024-01-31",
                periods=12,
                freq="ME",
            ),
            "rank_ic": [
                0.10, 0.10,   0.10, 0.10,
                0.10, np.nan, 0.10, 0.10,
                0.10, 0.10,   0.10, 0.10,
            ],
        }
    )

    result = calculate_rank_ic_rolling_diagnostics(monthly_ic)

    december = result.iloc[-1]

    assert pd.isna(december["rolling_mean_ic_12m"])
    assert pd.isna(december["rolling_std_ic_12m"])
    assert pd.isna(december["rolling_positive_fraction_12m"])


def test_summarize_rank_ic_by_subperiod():
    monthly_ic = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2001-01-31",
                    "2001-02-28",
                    "2009-01-31",
                    "2009-02-28",
                    "2017-01-31",
                    "2017-02-28",
                ]
            ),
            "rank_ic": [
                0.10, 0.20, 
                -0.10,0.10,
                0.20, 0.40,
            ],
            "n_obs": [
                900, 910, 
                920, 930,
                940, 950,
            ],
        }
    )

    result = summarize_rank_ic_by_subperiod(monthly_ic)

    assert result["subperiod"].tolist() == [
        "2001-2008",
        "2009-2016",
        "2017-2025",
    ]

    assert result["n_months"].tolist() == [2, 2, 2]

    assert result.loc[
        result["subperiod"] == "2001-2008",
        "mean_ic",
    ].iloc[0] == pytest.approx(0.15)

    assert result.loc[
        result["subperiod"] == "2009-2016",
        "mean_ic",
    ].iloc[0] == pytest.approx(0.00)

    assert result.loc[
        result["subperiod"] == "2017-2025",
        "mean_ic",
    ].iloc[0] == pytest.approx(0.30)


def test_summarize_rank_ic_by_subperiod_excludes_missing_ic():
    monthly_ic = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2001-01-31",
                    "2001-02-28",
                    "2001-03-31",
                    "2009-01-31",
                    "2017-01-31",
                ]
            ),
            "rank_ic": [
                0.10, np.nan, 0.30,
                0.20,
                0.40,
            ],
            "n_obs": [
                900, 910, 920,
                930,
                940,
            ],
        }
    )

    result = summarize_rank_ic_by_subperiod(monthly_ic)

    early = result.loc[
        result["subperiod"] == "2001-2008"
    ].iloc[0]

    assert early["n_months"] == 2
    assert early["mean_ic"] == pytest.approx(0.20)
    assert early["positive_ic_fraction"] == pytest.approx(1.0)


def test_rank_ic_predictive_analysis_integration():
    dates = pd.date_range(
        "2001-01-31",
        "2002-12-31",
        freq="ME",
    )

    return_by_security = {
        1: 0.03,
        2: 0.02,
        3: 0.01,
        4: -0.50,
    }

    rows = []

    for security_id, monthly_return in return_by_security.items():
        for date in dates:
            total_return = monthly_return

            if (
                security_id == 3
                and date == pd.Timestamp("2002-12-31")
            ):
                total_return = np.nan

            rows.append(
                {
                    "security_id": security_id,
                    "date": date,
                    "total_return": total_return,
                    "in_universe": security_id != 4,
                }
            )

    panel = pd.DataFrame(rows)

    panel = add_momentum_signal(panel)
    panel = add_momentum_eligibility(panel)

    monthly_ic = calculate_monthly_rank_ic(panel)

    december_panel = panel.loc[
        panel["date"] == pd.Timestamp("2002-12-31")
    ]

    security_3 = december_panel.loc[
        december_panel["security_id"] == 3
    ].iloc[0]

    assert security_3["momentum_eligible"]
    assert pd.notna(security_3["momentum"])
    assert pd.isna(security_3["total_return"])

    december_ic = monthly_ic.loc[
        monthly_ic["date"] == pd.Timestamp("2002-12-31")
    ].iloc[0]

    assert december_ic["n_obs"] == 2
    assert december_ic["rank_ic"] == pytest.approx(1.0)

    # Full-sample summary
    summary = summarize_rank_ic(monthly_ic)

    assert summary["n_months"] == 12
    assert summary["mean_ic"] == pytest.approx(1.0)
    assert summary["positive_ic_fraction"] == pytest.approx(1.0)

    # Autocorrelation
    autocorr = summarize_rank_ic_autocorrelation(monthly_ic)

    assert autocorr["n_pairs"] == 11
    assert pd.isna(autocorr["lag1_autocorr"])

    # Rolling diagnostics
    rolling = calculate_rank_ic_rolling_diagnostics(monthly_ic)

    december_rolling = rolling.loc[
        rolling["date"] == pd.Timestamp("2002-12-31")
    ].iloc[0]

    assert december_rolling[
        "rolling_mean_ic_12m"
    ] == pytest.approx(1.0)

    assert december_rolling[
        "rolling_std_ic_12m"
    ] == pytest.approx(0.0)

    assert december_rolling[
        "rolling_positive_fraction_12m"
    ] == pytest.approx(1.0)

    # Subperiod summary
    subperiods = summarize_rank_ic_by_subperiod(monthly_ic)

    early = subperiods.loc[
        subperiods["subperiod"] == "2001-2008"
    ].iloc[0]

    assert early["n_months"] == 12
    assert early["mean_ic"] == pytest.approx(1.0)