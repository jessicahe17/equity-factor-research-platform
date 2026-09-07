import numpy as np
import pandas as pd
import pytest

from equity_factor_research.analysis.portfolio_sorts import (
    assign_momentum_quintiles,
    calculate_monthly_quintile_returns,
    calculate_monthly_long_short_returns,
    summarize_long_short_returns,
    calculate_long_short_rolling_diagnostics,
    summarize_long_short_returns_by_subperiod,
)
from equity_factor_research.factors.momentum import(
    add_momentum_signal,
    add_momentum_eligibility,
)


def test_assign_momentum_quintiles():
    panel = pd.DataFrame(
        {
            "security_id": range(1, 11),
            "date": pd.to_datetime(["2025-01-31"] * 10),
            "momentum": [
                0.01, 0.02,
                0.03, 0.04,
                0.05, 0.06,
                0.07, 0.08,
                0.09, 0.10,
            ],
            "momentum_eligible": [True] * 10,
            "in_universe": [True] * 10,
        }
    )

    result = assign_momentum_quintiles(panel)

    assert result["momentum_quintile"].tolist() == [
        1, 1,
        2, 2,
        3, 3,
        4, 4,
        5, 5,
    ]


def test_assign_momentum_quintiles_excludes_ineligible_securities():
    panel = pd.DataFrame(
        {
            "security_id": range(1, 9),
            "date": pd.to_datetime(["2025-01-31"] * 8),
            "momentum": [
                0.01, 0.02, 0.03, 0.04, 0.05, 0.06,
                0.07, np.nan,
            ],
            "momentum_eligible": [
                True, True, True, True, True, True,
                False, True,
            ],
            "in_universe": [
                True, True, True, True, True, True,
                True, False,
            ],
        }
    )

    result = assign_momentum_quintiles(panel)

    assert result.loc[
        result["security_id"].between(1, 6),
        "momentum_quintile",
    ].notna().all()

    assert result.loc[
        result["security_id"].isin([7, 8]),
        "momentum_quintile",
    ].isna().all()


def test_assign_momentum_quintiles_allows_missing_outcome():
    panel = pd.DataFrame(
        {
            "security_id": range(1, 6),
            "date": pd.to_datetime(["2025-01-31"] * 5),
            "momentum": [
                0.01, 0.02, 0.03, 0.04, 0.05,
            ],
            "momentum_eligible": [True] * 5,
            "in_universe": [True] * 5,
            "total_return": [
                0.02, 0.01, np.nan, -0.01, 0.03,
            ],
        }
    )

    result = assign_momentum_quintiles(panel)

    security_3 = result.loc[
        result["security_id"] == 3
    ].iloc[0]

    assert pd.isna(security_3["total_return"])
    assert security_3["momentum_quintile"] == 3


def test_assign_momentum_quintiles_requires_at_least_five_securities():
    panel = pd.DataFrame(
        {
            "security_id": range(1, 5),
            "date": pd.to_datetime(["2025-01-31"] * 4),
            "momentum": [0.01, 0.02, 0.03, 0.04],
            "momentum_eligible": [True] * 4,
            "in_universe": [True] * 4,
        }
    )

    result = assign_momentum_quintiles(panel)

    assert result["momentum_quintile"].isna().all()


def test_assign_momentum_quintiles_handles_tied_momentum_deterministically():
    panel = pd.DataFrame(
        {
            "security_id": [
                10, 2,
                8, 4,
                6, 1,
                3, 5,
                7, 9,
            ],
            "date": pd.to_datetime(["2025-01-31"] * 10),
            "momentum": [
                0.10, 0.10,
                0.20, 0.20,
                0.30, 0.30,
                0.40, 0.40,
                0.50, 0.50,
            ],
            "momentum_eligible": [True] * 10,
            "in_universe": [True] * 10,
        }
    )

    result = assign_momentum_quintiles(panel)

    sorted_result = result.sort_values(
        ["momentum", "security_id"]
    )

    assert sorted_result["momentum_quintile"].tolist() == [
        1, 1,
        2, 2,
        3, 3,
        4, 4,
        5, 5,
    ]


def test_assign_momentum_quintiles_assigns_each_month_independently():
    panel = pd.DataFrame(
        {
            "security_id": list(range(1, 6)) * 2,
            "date": pd.to_datetime(
                ["2025-01-31"] * 5
                + ["2025-02-28"] * 5
            ),
            "momentum": [
                # January
                0.01, 0.02, 0.03, 0.04, 0.05,

                # February: reversed ordering
                0.05, 0.04, 0.03, 0.02, 0.01,
            ],
            "momentum_eligible": [True] * 10,
            "in_universe": [True] * 10,
        }
    )

    result = assign_momentum_quintiles(panel)

    january = (
        result.loc[
            result["date"] == pd.Timestamp("2025-01-31")
        ]
        .sort_values("security_id")
    )

    february = (
        result.loc[
            result["date"] == pd.Timestamp("2025-02-28")
        ]
        .sort_values("security_id")
    )

    assert january["momentum_quintile"].tolist() == [
        1, 2, 3, 4, 5
    ]

    assert february["momentum_quintile"].tolist() == [
        5, 4, 3, 2, 1
    ]


def test_calculate_monthly_quintile_returns():
    panel = pd.DataFrame(
        {
            "security_id": range(1, 11),
            "date": pd.to_datetime(["2025-01-31"] * 10),
            "momentum_quintile": [
                1, 1,
                2, 2,
                3, 3,
                4, 4,
                5, 5,
            ],
            "total_return": [
                -0.10, -0.20,
                -0.05, 0.01,
                0.00, 0.02,
                0.03, 0.05,
                0.08, 0.12,
            ],
        }
    )

    result = calculate_monthly_quintile_returns(panel)

    assert result["momentum_quintile"].tolist() == [
        1, 2, 3, 4, 5
    ]

    assert result["portfolio_return"].tolist() == pytest.approx(
        [
            -0.15,
            -0.02,
            0.01,
            0.04,
            0.10,
        ]
    )

    assert result["n_assigned"].tolist() == [
        2, 2, 2, 2, 2
    ]

    assert result["n_obs"].tolist() == [
        2, 2, 2, 2, 2
    ]


def test_calculate_monthly_quintile_returns_excludes_missing_returns():
    panel = pd.DataFrame(
        {
            "security_id": [1, 2, 3, 4],
            "date": pd.to_datetime(["2025-01-31"] * 4),
            "momentum_quintile": [1, 1, 5, 5],
            "total_return": [
                -0.10, np.nan,
                0.08,  0.12,
            ],
        }
    )

    result = calculate_monthly_quintile_returns(panel)

    q1 = result.loc[
        result["momentum_quintile"] == 1
    ].iloc[0]

    q5 = result.loc[
        result["momentum_quintile"] == 5
    ].iloc[0]

    assert q1["n_assigned"] == 2
    assert q1["n_obs"] == 1
    assert q1["portfolio_return"] == pytest.approx(-0.10)

    assert q5["n_assigned"] == 2
    assert q5["n_obs"] == 2
    assert q5["portfolio_return"] == pytest.approx(0.10)


def test_calculate_monthly_quintile_returns_all_returns_missing():
    panel = pd.DataFrame(
        {
            "security_id": [1, 2],
            "date": pd.to_datetime(["2025-01-31"] * 2),
            "momentum_quintile": [3, 3],
            "total_return": [np.nan, np.nan],
        }
    )

    result = calculate_monthly_quintile_returns(panel)

    q3 = result.iloc[0]

    assert q3["n_assigned"] == 2
    assert q3["n_obs"] == 0
    assert pd.isna(q3["portfolio_return"])


def test_calculate_monthly_quintile_returns_handles_multiple_months():
    panel = pd.DataFrame(
        {
            "security_id": [
                1, 2, 3, 4,
                1, 2, 3, 4,
            ],
            "date": pd.to_datetime(
                ["2025-01-31"] * 4
                + ["2025-02-28"] * 4
            ),
            "momentum_quintile": [
                1, 1, 5, 5,
                1, 1, 5, 5,
            ],
            "total_return": [
                # January
                -0.10, -0.20,
                0.10,  0.20,

                # February
                0.02,  0.04,
                -0.01, 0.03,
            ],
        }
    )

    result = calculate_monthly_quintile_returns(panel)

    january = result.loc[
        result["date"] == pd.Timestamp("2025-01-31")
    ]

    february = result.loc[
        result["date"] == pd.Timestamp("2025-02-28")
    ]

    jan_q1 = january.loc[
        january["momentum_quintile"] == 1
    ].iloc[0]

    jan_q5 = january.loc[
        january["momentum_quintile"] == 5
    ].iloc[0]

    feb_q1 = february.loc[
        february["momentum_quintile"] == 1
    ].iloc[0]

    feb_q5 = february.loc[
        february["momentum_quintile"] == 5
    ].iloc[0]

    assert jan_q1["portfolio_return"] == pytest.approx(-0.15)
    assert jan_q5["portfolio_return"] == pytest.approx(0.15)

    assert feb_q1["portfolio_return"] == pytest.approx(0.03)
    assert feb_q5["portfolio_return"] == pytest.approx(0.01)


def test_calculate_monthly_long_short_returns():
    quintile_returns = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-31",
                    "2025-01-31",
                    "2025-02-28",
                    "2025-02-28",
                ]
            ),
            "momentum_quintile": [
                1, 5,
                1, 5,
            ],
            "portfolio_return": [
                -0.04,
                0.06,
                0.02,
                0.05,
            ],
        }
    )

    result = calculate_monthly_long_short_returns(quintile_returns)

    assert result["q1_return"].tolist() == pytest.approx(
        [-0.04, 0.02]
    )

    assert result["q5_return"].tolist() == pytest.approx(
        [0.06, 0.05]
    )

    assert result["long_short_return"].tolist() == pytest.approx(
        [0.10, 0.03]
    )


def test_calculate_monthly_long_short_returns_preserves_missing_leg():
    quintile_returns = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-31",
                    "2025-01-31",
                ]
            ),
            "momentum_quintile": [1, 5],
            "portfolio_return": [
                np.nan,
                0.05,
            ],
        }
    )

    result = calculate_monthly_long_short_returns(
        quintile_returns
    )

    row = result.iloc[0]

    assert pd.isna(row["q1_return"])
    assert row["q5_return"] == pytest.approx(0.05)
    assert pd.isna(row["long_short_return"])


def test_calculate_monthly_long_short_returns_handles_missing_quintile():
    quintile_returns = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-31",
                    "2025-02-28",
                    "2025-02-28",
                ]
            ),
            "momentum_quintile": [
                5,
                1,
                5,
            ],
            "portfolio_return": [
                0.05,
                -0.02,
                0.04,
            ],
        }
    )

    result = calculate_monthly_long_short_returns(
        quintile_returns
    )

    january = result.loc[
        result["date"] == pd.Timestamp("2025-01-31")
    ].iloc[0]

    february = result.loc[
        result["date"] == pd.Timestamp("2025-02-28")
    ].iloc[0]

    assert pd.isna(january["q1_return"])
    assert january["q5_return"] == pytest.approx(0.05)
    assert pd.isna(january["long_short_return"])

    assert february["long_short_return"] == pytest.approx(0.06)


def test_summarize_long_short_returns():
    long_short_returns = pd.DataFrame(
        {
            "date": pd.date_range(
                "2025-01-31",
                periods=4,
                freq="ME",
            ),
            "q1_return": [-0.03, -0.01, 0.02, -0.02],
            "q5_return": [0.05,  0.03,  0.01, 0.04],
            "long_short_return": [0.08, 0.04, -0.01, 0.06],
        }
    )

    result = summarize_long_short_returns(long_short_returns)

    values = long_short_returns["long_short_return"]

    expected_t_stat = (
        values.mean()
        / (values.std() / np.sqrt(len(values)))
    )

    assert result["n_months"] == 4

    assert result["mean_monthly_return"] == pytest.approx(
        values.mean()
    )

    assert result["median_monthly_return"] == pytest.approx(
        values.median()
    )

    assert result["std_monthly_return"] == pytest.approx(
        values.std()
    )

    assert result[
        "positive_return_fraction"
    ] == pytest.approx(
        (values > 0).mean()
    )

    assert result["return_t_stat"] == pytest.approx(
        expected_t_stat
    )

    assert result[
        "annualized_mean_return"
    ] == pytest.approx(
        values.mean() * 12
    )

    assert result[
        "annualized_volatility"
    ] == pytest.approx(
        values.std() * np.sqrt(12)
    )


def test_summarize_long_short_returns_excludes_missing_returns():
    long_short_returns = pd.DataFrame(
        {
            "date": pd.date_range(
                "2025-01-31",
                periods=4,
                freq="ME",
            ),
            "long_short_return": [
                0.05, np.nan, -0.01, 0.02,
            ],
        }
    )

    result = summarize_long_short_returns(
        long_short_returns
    )

    assert result["n_months"] == 3
    assert result["mean_monthly_return"] == pytest.approx(
        (0.05 - 0.01 + 0.02) / 3
    )
    assert result[
        "positive_return_fraction"
    ] == pytest.approx(2 / 3)


def test_summarize_long_short_returns_returns_nan_t_stat_for_zero_variance():
    long_short_returns = pd.DataFrame(
        {
            "date": pd.date_range(
                "2025-01-31",
                periods=3,
                freq="ME",
            ),
            "long_short_return": [
                0.02, 0.02, 0.02,
            ],
        }
    )

    result = summarize_long_short_returns(
        long_short_returns
    )

    assert result["n_months"] == 3
    assert result["mean_monthly_return"] == pytest.approx(0.02)
    assert result["std_monthly_return"] == pytest.approx(0.0)
    assert pd.isna(result["return_t_stat"])


def test_summarize_long_short_returns_one_valid_month():
    long_short_returns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-31"]),
            "long_short_return": [0.03],
        }
    )

    result = summarize_long_short_returns(long_short_returns)

    assert result["n_months"] == 1
    assert result["mean_monthly_return"] == pytest.approx(0.03)
    assert result["median_monthly_return"] == pytest.approx(0.03)
    assert result["positive_return_fraction"] == pytest.approx(1.0)
    assert pd.isna(result["std_monthly_return"])
    assert pd.isna(result["return_t_stat"])
    assert result["annualized_mean_return"] == pytest.approx(0.36)
    assert pd.isna(result["annualized_volatility"])


def test_calculate_long_short_rolling_diagnostics():
    long_short_returns = pd.DataFrame(
        {
            "date": pd.date_range(
                "2024-01-31",
                periods=12,
                freq="ME",
            ),
            "long_short_return": [
                0.05, 0.03, -0.02, 0.04,
                0.01, -0.01, 0.06, 0.02,
                0.00, 0.03, -0.02, 0.05,
            ],
        }
    )

    result = calculate_long_short_rolling_diagnostics(long_short_returns)

    values = long_short_returns["long_short_return"]

    assert result.loc[
        :10,
        [
            "rolling_mean_return_12m",
            "rolling_std_return_12m",
            "rolling_positive_fraction_12m",
        ],
    ].isna().all().all()

    december = result.iloc[-1]

    assert december[
        "rolling_mean_return_12m"
    ] == pytest.approx(
        values.mean()
    )

    assert december[
        "rolling_std_return_12m"
    ] == pytest.approx(
        values.std()
    )

    assert december[
        "rolling_positive_fraction_12m"
    ] == pytest.approx(
        (values > 0).mean()
    )


def test_calculate_long_short_rolling_diagnostics_missing_calendar_month():
    dates = pd.date_range(
        "2024-01-31",
        periods=13,
        freq="ME",
    ).delete(5)

    long_short_returns = pd.DataFrame(
        {
            "date": dates,
            "long_short_return": [0.02] * 12,
        }
    )

    result = calculate_long_short_rolling_diagnostics(long_short_returns)

    june_2024 = result.loc[
        result["date"] == pd.Timestamp("2024-06-30")
    ].iloc[0]

    december_2024 = result.loc[
        result["date"] == pd.Timestamp("2024-12-31")
    ].iloc[0]

    assert pd.isna(june_2024["rolling_mean_return_12m"])
    assert pd.isna(december_2024["rolling_mean_return_12m"])
    assert pd.isna(december_2024["rolling_std_return_12m"])
    assert pd.isna(december_2024["rolling_positive_fraction_12m"])


def test_calculate_long_short_rolling_diagnostics_missing_return():
    long_short_returns = pd.DataFrame(
        {
            "date": pd.date_range(
                "2024-01-31",
                periods=12,
                freq="ME",
            ),
            "long_short_return": [
                0.02, 0.02, 0.02, 0.02,
                0.02, np.nan, 0.02, 0.02,
                0.02, 0.02, 0.02, 0.02,
            ],
        }
    )

    result = calculate_long_short_rolling_diagnostics(long_short_returns)

    december = result.iloc[-1]

    assert pd.isna(december["rolling_mean_return_12m"])
    assert pd.isna(december["rolling_std_return_12m"])
    assert pd.isna(december["rolling_positive_fraction_12m"])


def test_summarize_long_short_returns_by_subperiod():
    long_short_returns = pd.DataFrame(
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
            "long_short_return": [
                0.02, 0.04,
                -0.01, 0.01,
                0.03, 0.05,
            ],
        }
    )

    result = summarize_long_short_returns_by_subperiod(
        long_short_returns
    )

    assert result["subperiod"].tolist() == [
        "2001-2008",
        "2009-2016",
        "2017-2025",
    ]

    assert result["n_months"].tolist() == [2, 2, 2]

    assert result.loc[
        result["subperiod"] == "2001-2008",
        "mean_monthly_return",
    ].iloc[0] == pytest.approx(0.03)

    assert result.loc[
        result["subperiod"] == "2009-2016",
        "mean_monthly_return",
    ].iloc[0] == pytest.approx(0.00)

    assert result.loc[
        result["subperiod"] == "2017-2025",
        "mean_monthly_return",
    ].iloc[0] == pytest.approx(0.04)


def test_summarize_long_short_returns_by_subperiod_excludes_missing_returns():
    long_short_returns = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2001-01-31",
                    "2001-02-28",
                    "2001-03-31",
                ]
            ),
            "long_short_return": [
                0.02, np.nan, 0.04,
            ],
        }
    )

    result = summarize_long_short_returns_by_subperiod(
        long_short_returns
    )

    early = result.loc[
        result["subperiod"] == "2001-2008"
    ].iloc[0]

    assert early["n_months"] == 2
    assert early["mean_monthly_return"] == pytest.approx(0.03)
    assert early["positive_return_fraction"] == pytest.approx(1.0)


def test_momentum_portfolio_analysis_integration():
    dates = pd.date_range(
        "2001-01-31",
        "2002-12-31",
        freq="ME",
    )

    rows = []

    for security_id in range(1, 12):
        monthly_return = (
            security_id / 100
            if security_id <= 10
            else 0.50
        )

        for date in dates:
            total_return = monthly_return

            if (
                security_id == 10
                and date == pd.Timestamp("2002-12-31")
            ):
                total_return = np.nan

            rows.append(
                {
                    "security_id": security_id,
                    "date": date,
                    "total_return": total_return,
                    "in_universe": security_id <= 10,
                }
            )

    panel = pd.DataFrame(rows)

    panel = add_momentum_signal(panel)
    panel = add_momentum_eligibility(panel)
    panel = assign_momentum_quintiles(panel)

    # December portfolio membership
    december_panel = panel.loc[
        panel["date"] == pd.Timestamp("2002-12-31")
    ]

    security_10 = december_panel.loc[
        december_panel["security_id"] == 10
    ].iloc[0]

    security_11 = december_panel.loc[
        december_panel["security_id"] == 11
    ].iloc[0]

    assert security_10["momentum_eligible"]
    assert pd.notna(security_10["momentum"])
    assert security_10["momentum_quintile"] == 5
    assert pd.isna(security_10["total_return"])

    assert pd.isna(security_11["momentum_quintile"])

    # Check Q1 and Q5 membership
    q1_ids = december_panel.loc[
        december_panel["momentum_quintile"] == 1,
        "security_id",
    ].tolist()

    q5_ids = december_panel.loc[
        december_panel["momentum_quintile"] == 5,
        "security_id",
    ].tolist()

    assert q1_ids == [1, 2]
    assert q5_ids == [9, 10]

    # Monthly quintile returns
    quintile_returns = calculate_monthly_quintile_returns(
        panel
    )

    december_quintiles = quintile_returns.loc[
        quintile_returns["date"]
        == pd.Timestamp("2002-12-31")
    ]

    december_q1 = december_quintiles.loc[
        december_quintiles["momentum_quintile"] == 1
    ].iloc[0]

    december_q5 = december_quintiles.loc[
        december_quintiles["momentum_quintile"] == 5
    ].iloc[0]

    assert december_q1["n_assigned"] == 2
    assert december_q1["n_obs"] == 2
    assert december_q1["portfolio_return"] == pytest.approx(
        0.015
    )

    assert december_q5["n_assigned"] == 2
    assert december_q5["n_obs"] == 1
    assert december_q5["portfolio_return"] == pytest.approx(
        0.09
    )

    # Monthly long-short returns
    long_short = calculate_monthly_long_short_returns(
        quintile_returns
    )

    assert len(long_short) == 12

    december_long_short = long_short.loc[
        long_short["date"] == pd.Timestamp("2002-12-31")
    ].iloc[0]

    assert december_long_short[
        "long_short_return"
    ] == pytest.approx(
        0.075
    )

    # Full-sample summary
    summary = summarize_long_short_returns(
        long_short
    )

    expected_mean = (
        11 * 0.08 + 0.075
    ) / 12

    assert summary["n_months"] == 12
    assert summary[
        "mean_monthly_return"
    ] == pytest.approx(
        expected_mean
    )
    assert summary[
        "positive_return_fraction"
    ] == pytest.approx(1.0)

    # Rolling diagnostics
    rolling = calculate_long_short_rolling_diagnostics(
        long_short
    )

    assert rolling.loc[
        :10,
        [
            "rolling_mean_return_12m",
            "rolling_std_return_12m",
            "rolling_positive_fraction_12m",
        ],
    ].isna().all().all()

    december_rolling = rolling.iloc[-1]

    assert december_rolling[
        "rolling_mean_return_12m"
    ] == pytest.approx(
        expected_mean
    )

    assert december_rolling[
        "rolling_positive_fraction_12m"
    ] == pytest.approx(1.0)

    # Subperiod summary
    subperiods = (
        summarize_long_short_returns_by_subperiod(
            long_short
        )
    )

    early = subperiods.loc[
        subperiods["subperiod"] == "2001-2008"
    ].iloc[0]

    assert early["n_months"] == 12
    assert early[
        "mean_monthly_return"
    ] == pytest.approx(
        expected_mean
    )