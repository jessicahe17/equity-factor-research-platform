import numpy as np
import pandas as pd
import pytest

from equity_factor_research.analysis.robustness import (
    build_rank_ic_robustness_comparison,
    build_long_short_robustness_comparison,
)
from equity_factor_research.analysis.rank_ic import(
    calculate_monthly_rank_ic,
)
from equity_factor_research.data.universe import(
    add_universe_membership,
)
from equity_factor_research.analysis.portfolio_sorts import(
    assign_momentum_quintiles,
    calculate_monthly_quintile_returns,
    calculate_monthly_long_short_returns,
    calculate_monthly_value_weighted_quintile_returns,
)
from equity_factor_research.factors.momentum import(
    add_momentum_signal,
    add_momentum_eligibility,
)


def test_build_rank_ic_robustness_comparison():
    baseline = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-31",
                    "2025-02-28",
                    "2025-03-31",
                ]
            ),
            "rank_ic": [0.10, 0.20, 0.30],
            "n_obs": [100, 100, 100],
        }
    )

    alternative = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-31",
                    "2025-02-28",
                    "2025-03-31",
                ]
            ),
            "rank_ic": [-0.10, 0.00, 0.10],
            "n_obs": [80, 80, 80],
        }
    )

    result = build_rank_ic_robustness_comparison(
        {
            "baseline": baseline,
            "alternative": alternative,
        }
    )

    assert list(result["specification"]) == ["baseline", "alternative"]
    assert len(result) == 2

    baseline_row = result.loc[
        result["specification"] == "baseline"
    ].iloc[0]

    alternative_row = result.loc[
        result["specification"] == "alternative"
    ].iloc[0]

    assert baseline_row["n_months"] == 3
    assert baseline_row["mean_ic"] == pytest.approx(0.20)
    assert baseline_row[
        "positive_ic_fraction"
    ] == pytest.approx(1.0)

    assert alternative_row["n_months"] == 3
    assert alternative_row["mean_ic"] == pytest.approx(0.0)
    assert alternative_row[
        "positive_ic_fraction"
    ] == pytest.approx(1 / 3)

    assert baseline_row["ic_t_stat"] == pytest.approx(3.464101615)
    assert alternative_row["ic_t_stat"] == pytest.approx(0.0)


def test_build_rank_ic_robustness_comparison_excludes_missing_ic():
    monthly_ic = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-31",
                    "2025-02-28",
                    "2025-03-31",
                ]
            ),
            "rank_ic": [0.10, np.nan, 0.30],
            "n_obs": [100, 0, 100],
        }
    )

    result = build_rank_ic_robustness_comparison(
        {
            "baseline": monthly_ic,
        }
    )

    row = result.iloc[0]

    assert row["n_months"] == 2
    assert row["mean_ic"] == pytest.approx(0.20)
    assert row["positive_ic_fraction"] == pytest.approx(1.0)
    assert row["ic_t_stat"] == pytest.approx(2.0)


def test_build_rank_ic_robustness_comparison_empty():
    result = build_rank_ic_robustness_comparison({})

    assert result.empty

    assert list(result.columns) == [
        "specification",
        "n_months",
        "mean_ic",
        "ic_t_stat",
        "positive_ic_fraction",
    ]


def test_build_long_short_robustness_comparison():
    baseline = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-31",
                    "2025-02-28",
                    "2025-03-31",
                ]
            ),
            "q1_return": [0.00, 0.00, 0.00],
            "q5_return": [0.01, 0.02, 0.03],
            "long_short_return": [0.01, 0.02, 0.03],
        }
    )

    alternative = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-31",
                    "2025-02-28",
                    "2025-03-31",
                ]
            ),
            "q1_return": [0.00, 0.00, 0.00],
            "q5_return": [-0.01, 0.00, 0.01],
            "long_short_return": [-0.01, 0.00, 0.01],
        }
    )

    result = build_long_short_robustness_comparison(
        {
            "baseline": baseline,
            "alternative": alternative,
        }
    )

    assert list(result["specification"]) == ["baseline", "alternative"]
    assert len(result) == 2

    baseline_row = result.loc[
        result["specification"] == "baseline"
    ].iloc[0]

    alternative_row = result.loc[
        result["specification"] == "alternative"
    ].iloc[0]

    assert baseline_row["n_months"] == 3
    assert baseline_row["mean_monthly_return"] == pytest.approx(0.02)
    assert baseline_row["return_t_stat"] == pytest.approx(3.464101615)
    assert baseline_row["annualized_mean_return"] == pytest.approx(0.24)
    assert baseline_row["annualized_volatility"] == pytest.approx(
        0.01 * np.sqrt(12)
    )
    assert baseline_row[
        "positive_return_fraction"
    ] == pytest.approx(1.0)

    assert alternative_row["n_months"] == 3
    assert alternative_row["mean_monthly_return"] == pytest.approx(0.0)
    assert alternative_row["return_t_stat"] == pytest.approx(0.0)
    assert alternative_row["annualized_mean_return"] == pytest.approx(0.0)
    assert alternative_row[
        "annualized_volatility"
    ] == pytest.approx(0.01 * np.sqrt(12))
    assert alternative_row[
        "positive_return_fraction"
    ] == pytest.approx(1 / 3)


def test_build_long_short_robustness_comparison_excludes_missing_returns():
    long_short = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-31",
                    "2025-02-28",
                    "2025-03-31",
                ]
            ),
            "q1_return": [0.00, np.nan, 0.00],
            "q5_return": [0.01, np.nan, 0.03],
            "long_short_return": [0.01, np.nan, 0.03],
        }
    )

    result = build_long_short_robustness_comparison(
        {
            "baseline": long_short,
        }
    )

    row = result.iloc[0]

    assert row["n_months"] == 2
    assert row["mean_monthly_return"] == pytest.approx(0.02)
    assert row["return_t_stat"] == pytest.approx(2.0)
    assert row["annualized_mean_return"] == pytest.approx(0.24)
    assert row["annualized_volatility"] == pytest.approx(
        np.sqrt(0.0002) * np.sqrt(12)
    )
    assert row[
        "positive_return_fraction"
    ] == pytest.approx(1.0)


def test_build_long_short_robustness_comparison_empty():
    result = build_long_short_robustness_comparison({})

    assert result.empty

    assert list(result.columns) == [
        "specification",
        "n_months",
        "mean_monthly_return",
        "return_t_stat",
        "annualized_mean_return",
        "annualized_volatility",
        "positive_return_fraction",
    ]


def test_robustness_comparison_integration():
    panel = pd.DataFrame(
        {
            "security_id": list(range(1, 11)),
            "date": pd.to_datetime(
                ["2025-01-31"] * 10
            ),
            "base_eligible": [True] * 10,
            "lagged_market_cap": [
                1000.0, 900.0, 800.0, 700.0, 600.0,
                500.0, 400.0, 300.0, 200.0, 100.0,
            ],
            "momentum": [
                0.10, 0.20, 0.30, 0.40, 0.50,
                0.60, 0.70, 0.80, 0.90, 1.00,
            ],
            "momentum_eligible": [True] * 10,
            "total_return": [
                0.01, 0.02, 0.03, 0.04, 0.05,
                0.06, 0.07, 0.08, 0.09, 0.10,
            ],
        }
    )

    top_5 = add_universe_membership(panel, n=5)
    top_10 = add_universe_membership(panel, n=10)

    rank_ic_top_5 = calculate_monthly_rank_ic(top_5)
    rank_ic_top_10 = calculate_monthly_rank_ic(top_10)

    ic_comparison = build_rank_ic_robustness_comparison(
        {
            "top_5": rank_ic_top_5,
            "top_10": rank_ic_top_10,
        }
    )

    assert list(ic_comparison["specification"]) == [
        "top_5",
        "top_10",
    ]
    assert len(ic_comparison) == 2
    assert ic_comparison["n_months"].tolist() == [1, 1]
    assert ic_comparison["mean_ic"].notna().all()

    sorted_top_5 = assign_momentum_quintiles(top_5)
    sorted_top_10 = assign_momentum_quintiles(top_10)

    quintile_returns_top_5 = (
        calculate_monthly_quintile_returns(
            sorted_top_5
        )
    )
    quintile_returns_top_10 = (
        calculate_monthly_quintile_returns(
            sorted_top_10
        )
    )

    long_short_top_5 = (
        calculate_monthly_long_short_returns(
            quintile_returns_top_5
        )
    )
    long_short_top_10 = (
        calculate_monthly_long_short_returns(
            quintile_returns_top_10
        )
    )

    long_short_comparison = (
        build_long_short_robustness_comparison(
            {
                "top_5": long_short_top_5,
                "top_10": long_short_top_10,
            }
        )
    )

    assert list(
        long_short_comparison["specification"]
    ) == [
        "top_5",
        "top_10",
    ]
    assert len(long_short_comparison) == 2
    assert (
        long_short_comparison["n_months"].tolist()
        == [1, 1]
    )
    assert (
        long_short_comparison[
            "mean_monthly_return"
        ].notna().all()
    )


def test_robustness_pipeline_integration():
    dates = pd.date_range(
        "2024-01-31",
        "2025-01-31",
        freq="ME",
    )

    records = []

    for security_id in range(1, 11):
        for date in dates:
            total_return = security_id / 100.0

            if (
                security_id == 10
                and date == pd.Timestamp("2025-01-31")
            ):
                total_return = np.nan

            records.append(
                {
                    "security_id": security_id,
                    "date": date,
                    "total_return": total_return,
                    "base_eligible": True,
                    "lagged_market_cap": (
                        1100.0 - 100.0 * security_id
                    ),
                }
            )

    base_panel = pd.DataFrame(records)

    baseline = add_momentum_signal(
        base_panel,
        start_lag=12,
        end_lag=2,
    )

    baseline = add_momentum_eligibility(
        baseline,
        start_lag=12,
        end_lag=2,
    )

    baseline = add_universe_membership(
        baseline,
        n=10,
    )

    baseline_january = baseline.loc[
        baseline["date"]
        == pd.Timestamp("2025-01-31")
    ]

    security_10 = baseline_january.loc[
        baseline_january["security_id"] == 10
    ].iloc[0]

    assert security_10["momentum_eligible"]
    assert pd.notna(security_10["momentum"])
    assert pd.isna(security_10["total_return"])

    baseline_ic = calculate_monthly_rank_ic(baseline)

    baseline_sorted = assign_momentum_quintiles(baseline)

    baseline_quintile_returns = (
        calculate_monthly_quintile_returns(baseline_sorted)
    )

    baseline_long_short = (
        calculate_monthly_long_short_returns(
            baseline_quintile_returns
        )
    )

    sorted_security_10 = baseline_sorted.loc[
        (
            baseline_sorted["date"]
            == pd.Timestamp("2025-01-31")
        )
        & (baseline_sorted["security_id"] == 10)
    ].iloc[0]

    assert sorted_security_10["momentum_quintile"] == 5

    value_weighted_quintile_returns = (
        calculate_monthly_value_weighted_quintile_returns(
            baseline_sorted
        )
    )

    value_weighted_long_short = (
        calculate_monthly_long_short_returns(
            value_weighted_quintile_returns
        )
    )

    january_vw_q5 = (
        value_weighted_quintile_returns.loc[
            (
                value_weighted_quintile_returns["date"]
                == pd.Timestamp("2025-01-31")
            )
            & (
                value_weighted_quintile_returns[
                    "momentum_quintile"
                ]
                == 5
            )
        ].iloc[0]
    )

    assert january_vw_q5["n_assigned"] == 2
    assert january_vw_q5["n_weightable"] == 2
    assert january_vw_q5["n_obs"] == 1
    assert january_vw_q5["weight_coverage"] == pytest.approx(2 / 3)
    assert january_vw_q5["portfolio_return"] == pytest.approx(0.09)

    top_5 = add_momentum_signal(
        base_panel,
        start_lag=12,
        end_lag=2,
    )

    top_5 = add_momentum_eligibility(
        top_5,
        start_lag=12,
        end_lag=2,
    )

    top_5 = add_universe_membership(
        top_5,
        n=5,
    )

    top_5_ic = calculate_monthly_rank_ic(top_5)
    top_5_sorted = assign_momentum_quintiles(top_5)
    top_5_returns = calculate_monthly_quintile_returns(top_5_sorted)
    top_5_long_short = (
        calculate_monthly_long_short_returns(top_5_returns)
    )

    january_top_5_ids = set(
        top_5.loc[
            (
                top_5["date"]
                == pd.Timestamp("2025-01-31")
            )
            & top_5["in_universe"],
            "security_id",
        ]
    )

    assert january_top_5_ids == {1, 2, 3, 4, 5}

    momentum_6_2 = add_momentum_signal(
        base_panel,
        start_lag=6,
        end_lag=2,
    )

    momentum_6_2 = add_momentum_eligibility(
        momentum_6_2,
        start_lag=6,
        end_lag=2,
    )

    momentum_6_2 = add_universe_membership(
        momentum_6_2,
        n=10,
    )

    ic_6_2 = calculate_monthly_rank_ic(momentum_6_2)
    sorted_6_2 = assign_momentum_quintiles(momentum_6_2)
    returns_6_2 = calculate_monthly_quintile_returns(sorted_6_2)
    long_short_6_2 = (
        calculate_monthly_long_short_returns(returns_6_2)
    )

    momentum_12_1 = add_momentum_signal(
        base_panel,
        start_lag=12,
        end_lag=1,
    )

    momentum_12_1 = add_momentum_eligibility(
        momentum_12_1,
        start_lag=12,
        end_lag=1,
    )

    momentum_12_1 = add_universe_membership(
        momentum_12_1,
        n=10,
    )

    ic_12_1 = calculate_monthly_rank_ic(momentum_12_1)
    sorted_12_1 = assign_momentum_quintiles(momentum_12_1)
    returns_12_1 = calculate_monthly_quintile_returns(sorted_12_1)
    long_short_12_1 = (
        calculate_monthly_long_short_returns(returns_12_1)
    )

    ic_comparison = build_rank_ic_robustness_comparison(
        {
            "baseline": baseline_ic,
            "top_5": top_5_ic,
            "6_2": ic_6_2,
            "12_1": ic_12_1,
        }
    )

    assert list(
        ic_comparison["specification"]
    ) == [
        "baseline",
        "top_5",
        "6_2",
        "12_1",
    ]

    assert (
        ic_comparison.iloc[0]["specification"]
        == "baseline"
    )
    assert ic_comparison["mean_ic"].notna().all()

    long_short_comparison = (
        build_long_short_robustness_comparison(
            {
                "baseline": baseline_long_short,
                "value_weighted": value_weighted_long_short,
                "top_5": top_5_long_short,
                "6_2": long_short_6_2,
                "12_1": long_short_12_1,
            }
        )
    )

    assert list(
        long_short_comparison["specification"]
    ) == [
        "baseline",
        "value_weighted",
        "top_5",
        "6_2",
        "12_1",
    ]

    assert (
        long_short_comparison.iloc[0]["specification"]
        == "baseline"
    )

    assert long_short_comparison["mean_monthly_return"].notna().all()