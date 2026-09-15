from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from equity_factor_research.data.providers.compustat import (
    assemble_compustat_monthly_panel,
)
from equity_factor_research.data.transforms import (
    construct_equity_panel,
)
from equity_factor_research.data.universe import (
    add_lagged_market_cap,
    add_base_eligibility,
    add_lagged_base_eligibility,
    add_universe_membership,
)
from equity_factor_research.factors.momentum import(
    add_momentum_signal,
    add_momentum_eligibility,
)
from equity_factor_research.analysis.rank_ic import(
    calculate_monthly_rank_ic,
    summarize_rank_ic,
    summarize_rank_ic_autocorrelation,
    calculate_rank_ic_rolling_diagnostics,
    summarize_rank_ic_by_subperiod,
)
from equity_factor_research.analysis.portfolio_sorts import(
    assign_momentum_quintiles,
    calculate_monthly_quintile_returns,
    calculate_monthly_long_short_returns,
    calculate_long_short_rolling_diagnostics,
    calculate_monthly_value_weighted_quintile_returns,
)
from equity_factor_research.analysis.robustness import(
    build_long_short_robustness_comparison,
    build_rank_ic_robustness_comparison,
)


SECM_PATH = (
    Path.home()
    / "Downloads"
    / "compustat_secm_2000_2025.csv.gz"
)

SEC_HISTORY_PATH = (
    Path.home()
    / "Downloads"
    / "compustat_sec_history_exchg.csv.gz"
)

SECURITY_PATH = (
    Path.home()
    / "Downloads"
    / "compustat_security.csv.gz"
)

REPORT_START = pd.Timestamp("2001-01-31")
REPORT_END = pd.Timestamp("2025-12-31")

UNIVERSE_SIZE = 1000

FIGURE_DIR = Path("results") / "figures"


def main():
    secm = pd.read_csv(
        SECM_PATH,
        dtype={
            "gvkey": "string",
            "iid": "string",
            "tic": "string",
            "tpci": "string",
        },
        usecols=[
            "gvkey",
            "iid",
            "tic",
            "datadate",
            "tpci",
            "prccm",
            "trt1m",
            "cshom",
        ],
    )

    sec_history = pd.read_csv(
        SEC_HISTORY_PATH,
        dtype={
            "gvkey": "string",
            "iid": "string",
            "item": "string",
            "itemvalue": "string",
        },
    )

    security = pd.read_csv(
        SECURITY_PATH,
        dtype={
            "gvkey": "string",
            "iid": "string",
            "dlrsni": "string",
        },
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    provider_panel = assemble_compustat_monthly_panel(
        secm,
        sec_history,
        security,
    )

    panel = construct_equity_panel(provider_panel)
    panel = add_lagged_market_cap(panel)
    panel = add_base_eligibility(panel)
    panel = add_lagged_base_eligibility(panel)
    panel = add_universe_membership(
        panel,
        n=UNIVERSE_SIZE,
    )

    base_panel = panel

    print("VALIDATED PANEL")
    print("-----------------")
    print(f"Rows: {len(panel):,}")
    print(f"Unique securities: {panel['security_id'].nunique():,}")
    print(
        "Date range:",
        panel["date"].min(),
        "to",
        panel["date"].max(),
    )

    panel = add_momentum_signal(base_panel)
    panel = add_momentum_eligibility(panel)


    reported_mask = panel["date"].between(
        REPORT_START,
        REPORT_END,
    )

    reported_panel = panel.loc[reported_mask].copy()

    monthly_universe_size = (
        reported_panel
        .groupby("date")["in_universe"]
        .sum()
    )

    print()
    print("REPORTED SAMPLE")
    print("---------------")
    print(f"Months: {reported_panel['date'].nunique()}")
    print(
        "Reported date range:",
        reported_panel["date"].min(),
        "to",
        reported_panel["date"].max(),
    )
    print(
        "Universe size range:",
        int(monthly_universe_size.min()),
        "to",
        int(monthly_universe_size.max()),
    )

    print()
    print("MOMENTUM COLUMNS")
    print("----------------")
    print(
        panel[
            [
                "date",
                "security_id",
                "in_universe",
                "momentum",
                "momentum_eligible",
                "total_return",
            ]
        ].dtypes
    )


    research_universe = reported_panel.loc[
        reported_panel["in_universe"]
    ].copy()

    research_universe["has_valid_signal"] = (
        research_universe["momentum"].notna()
    )

    research_universe["has_valid_outcome"] = (
        research_universe["total_return"].notna()
    )

    research_universe["has_signal_and_outcome"] = (
        research_universe["has_valid_signal"]
        & research_universe["has_valid_outcome"]
    )

    monthly_coverage = (
        research_universe
        .groupby("date")
        .agg(
            n_universe=("security_id", "size"),
            n_valid_signal=("has_valid_signal", "sum"),
            n_momentum_eligible=("momentum_eligible", "sum"),
            n_valid_outcome=("has_valid_outcome", "sum"),
            n_signal_and_outcome=("has_signal_and_outcome", "sum"),
        )
        .reset_index()
    )

    monthly_coverage["signal_coverage"] = (
        monthly_coverage["n_valid_signal"]
        / monthly_coverage["n_universe"]
    )

    monthly_coverage["analysis_coverage"] = (
        monthly_coverage["n_signal_and_outcome"]
        / monthly_coverage["n_universe"]
    )

    print()
    print("MONTHLY MOMENTUM COVERAGE — FIRST 12 MONTHS")
    print("-------------------------------------------")
    print(monthly_coverage.head(12).to_string(index=False))

    print()
    print("MONTHLY MOMENTUM COVERAGE — LAST 12 MONTHS")
    print("------------------------------------------")
    print(monthly_coverage.tail(12).to_string(index=False))

    print()
    print("COVERAGE SUMMARY")
    print("----------------")
    print(
        monthly_coverage[
            [
                "n_universe",
                "n_valid_signal",
                "n_momentum_eligible",
                "n_valid_outcome",
                "n_signal_and_outcome",
                "signal_coverage",
                "analysis_coverage",
            ]
        ].describe()
    )


    signal_eligibility_mismatch = (
        research_universe["has_valid_signal"]
        != research_universe["momentum_eligible"]
    )

    print()
    print("SIGNAL / ELIGIBILITY CONSISTENCY")
    print("--------------------------------")
    print(
        "Mismatched rows:",
        signal_eligibility_mismatch.sum(),
    )


    lowest_coverage = (
        monthly_coverage
        .nsmallest(10, "signal_coverage")
    )

    print()
    print("LOWEST SIGNAL-COVERAGE MONTHS")
    print("-----------------------------")
    print(
        lowest_coverage[
            [
                "date",
                "n_valid_signal",
                "n_valid_outcome",
                "n_signal_and_outcome",
                "signal_coverage",
                "analysis_coverage",
            ]
        ].to_string(index=False)
    )

    print()
    print("MISSINGNESS TOTALS — REPORTED SAMPLE")
    print("------------------------------------")

    print(
        "Missing signals:",
        (~research_universe["has_valid_signal"]).sum(),
    )

    print(
        "Missing outcomes:",
        (~research_universe["has_valid_outcome"]).sum(),
    )

    print(
        "Valid signal but missing outcome:",
        (
            research_universe["has_valid_signal"]
            & ~research_universe["has_valid_outcome"]
        ).sum(),
    )

    print(
        "Missing both signal and outcome:",
        (
            ~research_universe["has_valid_signal"]
            & ~research_universe["has_valid_outcome"]
        ).sum(),
    )

    valid_momentum = research_universe.loc[
        research_universe["has_valid_signal"],
        "momentum",
    ]

    momentum_quantiles = valid_momentum.quantile(
        [
            0.000,
            0.001,
            0.010,
            0.050,
            0.250,
            0.500,
            0.750,
            0.950,
            0.990,
            0.999,
            1.000,
        ]
    )

    print()
    print("MOMENTUM DISTRIBUTION")
    print("---------------------")
    print(f"Count: {valid_momentum.count():,}")
    print(f"Mean:  {valid_momentum.mean():.6f}")
    print(f"Std:   {valid_momentum.std():.6f}")

    print()
    print("Quantiles:")
    print(momentum_quantiles)


    monthly_rank_ic = calculate_monthly_rank_ic(
        reported_panel,
    )

    print()
    print("MONTHLY RANK IC — FIRST 12 MONTHS")
    print("---------------------------------")
    print(
        monthly_rank_ic
        .head(12)
        .to_string(index=False)
    )

    print()
    print("MONTHLY RANK IC — LAST 12 MONTHS")
    print("--------------------------------")
    print(
        monthly_rank_ic
        .tail(12)
        .to_string(index=False)
    )


    rank_ic_summary = summarize_rank_ic(
        monthly_rank_ic
    )

    print()
    print("RANK IC SUMMARY")
    print("---------------")
    print(rank_ic_summary)


    rank_ic_coverage_check = (
        monthly_rank_ic[
            ["date", "n_obs"]
        ]
        .merge(
            monthly_coverage[
                ["date", "n_signal_and_outcome"]
            ],
            on="date",
            how="left",
        )
    )

    rank_ic_coverage_check["difference"] = (
        rank_ic_coverage_check["n_obs"]
        - rank_ic_coverage_check["n_signal_and_outcome"]
    )

    print()
    print("RANK IC COVERAGE CHECK")
    print("----------------------")
    print(
        rank_ic_coverage_check["difference"]
        .value_counts()
        .sort_index()
    )


    rank_ic_autocorrelation = summarize_rank_ic_autocorrelation(
        monthly_rank_ic
    )

    print()
    print("RANK IC AUTOCORRELATION")
    print("-----------------------")
    print(rank_ic_autocorrelation)


    rolling_rank_ic = calculate_rank_ic_rolling_diagnostics(
        monthly_rank_ic
    )


    fig, ax = plt.subplots(
        figsize=(10, 5),
    )

    ax.plot(
        rolling_rank_ic["date"],
        rolling_rank_ic["rolling_mean_ic_12m"],
    )

    ax.axhline(
        0.0,
        linewidth=1,
        linestyle="--",
    )

    ax.set_title(
        "12-Month Rolling Mean Rank IC — 12–2 Momentum"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling Mean Rank IC")

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "rolling_rank_ic_12m.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


    print()
    print("ROLLING RANK IC — FIRST ROWS")
    print("----------------------------")
    print(
        rolling_rank_ic
        .head(15)
        .to_string(index=False)
    )

    print()
    print("ROLLING RANK IC — LAST ROWS")
    print("---------------------------")
    print(
        rolling_rank_ic
        .tail(15)
        .to_string(index=False)
    )

    print()
    print("LOWEST RANK IC MONTHS")
    print("---------------------")
    print(
        monthly_rank_ic
        .nsmallest(10, "rank_ic")
        .to_string(index=False)
    )

    print()
    print("HIGHEST RANK IC MONTHS")
    print("----------------------")
    print(
        monthly_rank_ic
        .nlargest(10, "rank_ic")
        .to_string(index=False)
    )


    rank_ic_subperiod_summary = summarize_rank_ic_by_subperiod(
        monthly_rank_ic
    )

    print()
    print("RANK IC SUBPERIOD SUMMARY")
    print("-------------------------")
    print(
        rank_ic_subperiod_summary
        .to_string(index=False)
    )


    valid_rolling_rank_ic = rolling_rank_ic.dropna(
        subset=["rolling_mean_ic_12m"]
    )

    print()
    print("LOWEST 12-MONTH ROLLING MEAN IC")
    print("-------------------------------")
    print(
        valid_rolling_rank_ic
        .nsmallest(5, "rolling_mean_ic_12m")
        .to_string(index=False)
    )

    print()
    print("HIGHEST 12-MONTH ROLLING MEAN IC")
    print("--------------------------------")
    print(
        valid_rolling_rank_ic
        .nlargest(5, "rolling_mean_ic_12m")
        .to_string(index=False)
    )


    quintile_panel = assign_momentum_quintiles(
        reported_panel
    )

    monthly_quintile_returns = calculate_monthly_quintile_returns(
        quintile_panel
    )

    quintile_counts = (
        quintile_panel.loc[
            quintile_panel["momentum_quintile"].notna()
        ]
        .groupby(
            ["date", "momentum_quintile"],
            observed=True,
        )
        .size()
        .unstack()
    )

    print()
    print("QUINTILE COUNTS — FIRST 12 MONTHS")
    print("---------------------------------")
    print(
        quintile_counts
        .head(12)
        .to_string()
    )

    print()
    print("QUINTILE COUNT SUMMARY")
    print("----------------------")
    print(quintile_counts.describe())

    print()
    print("EQUAL-WEIGHTED QUINTILE RETURNS — FIRST 25 ROWS")
    print("-----------------------------------------------")
    print(
        monthly_quintile_returns
        .head(25)
        .to_string(index=False)
    )

    print()
    print("EQUAL-WEIGHTED QUINTILE RETURNS — LAST 25 ROWS")
    print("----------------------------------------------")
    print(
        monthly_quintile_returns
        .tail(25)
        .to_string(index=False)
    )

    monthly_quintile_returns["n_missing_returns"] = (
        monthly_quintile_returns["n_assigned"]
        - monthly_quintile_returns["n_obs"]
    )

    print()
    print("QUINTILE OUTCOME-RETURN COVERAGE")
    print("--------------------------------")
    print(
        monthly_quintile_returns[
            [
                "n_assigned",
                "n_obs",
                "n_missing_returns",
            ]
        ].describe()
    )

    quintile_return_summary = (
        monthly_quintile_returns
        .groupby(
            "momentum_quintile",
            observed=True,
        )
        .agg(
            n_months=("portfolio_return", "count"),
            mean_monthly_return=("portfolio_return", "mean"),
            median_monthly_return=("portfolio_return", "median"),
            std_monthly_return=("portfolio_return", "std"),
        )
        .reset_index()
    )

    print()
    print("EQUAL-WEIGHTED QUINTILE RETURN SUMMARY")
    print("--------------------------------------")
    print(
        quintile_return_summary
        .to_string(index=False)
    )

    long_short_returns = calculate_monthly_long_short_returns(
        monthly_quintile_returns
    )

    print()
    print("EQUAL-WEIGHTED LONG-SHORT RETURNS — FIRST 12 MONTHS")
    print("---------------------------------------------------")
    print(
        long_short_returns
        .head(12)
        .to_string(index=False)
    )

    print()
    print("EQUAL-WEIGHTED LONG-SHORT RETURNS — LAST 12 MONTHS")
    print("--------------------------------------------------")
    print(
        long_short_returns
        .tail(12)
        .to_string(index=False)
    )

    long_short_summary = pd.Series(
        {
            "n_months": long_short_returns[
                "long_short_return"
            ].count(),
            "mean_monthly_return": long_short_returns[
                "long_short_return"
            ].mean(),
            "median_monthly_return": long_short_returns[
                "long_short_return"
            ].median(),
            "std_monthly_return": long_short_returns[
                "long_short_return"
            ].std(),
            "positive_month_fraction": (
                long_short_returns["long_short_return"] > 0
            ).mean(),
        }
    )

    long_short_summary["t_stat"] = (
        long_short_summary["mean_monthly_return"]
        / (
            long_short_summary["std_monthly_return"]
            / long_short_summary["n_months"] ** 0.5
        )
    )

    print()
    print("EQUAL-WEIGHTED LONG-SHORT SUMMARY")
    print("---------------------------------")
    print(long_short_summary)

    rolling_long_short = (
        calculate_long_short_rolling_diagnostics(
            long_short_returns
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 5),
    )

    ax.plot(
        rolling_long_short["date"],
        rolling_long_short["rolling_mean_return_12m"],
    )

    ax.axhline(
        0.0,
        linewidth=1,
        linestyle="--",
    )

    ax.set_title(
        "12-Month Rolling Mean Q5−Q1 Return — 12–2 Momentum"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling Mean Monthly Return")

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "rolling_long_short_12m.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


    print()
    print("ROLLING LONG-SHORT — FIRST 15 ROWS")
    print("----------------------------------")
    print(
        rolling_long_short
        .head(15)
        .to_string(index=False)
    )

    print()
    print("ROLLING LONG-SHORT — LAST 15 ROWS")
    print("---------------------------------")
    print(
        rolling_long_short
        .tail(15)
        .to_string(index=False)
    )

    valid_rolling_long_short = (
        rolling_long_short.dropna(
            subset=["rolling_mean_return_12m"]
        )
    )

    print()
    print("LOWEST 12-MONTH ROLLING LONG-SHORT RETURNS")
    print("-------------------------------------------")
    print(
        valid_rolling_long_short
        .nsmallest(
            5,
            "rolling_mean_return_12m",
        )
        .to_string(index=False)
    )

    print()
    print("HIGHEST 12-MONTH ROLLING LONG-SHORT RETURNS")
    print("--------------------------------------------")
    print(
        valid_rolling_long_short
        .nlargest(
            5,
            "rolling_mean_return_12m",
        )
        .to_string(index=False)
    )

    rank_ic_long_short = (
        monthly_rank_ic[
            ["date", "rank_ic"]
        ]
        .merge(
            long_short_returns[
                ["date", "long_short_return"]
            ],
            on="date",
            how="inner",
        )
    )

    print()
    print("RANK IC / LONG-SHORT MONTHLY CORRELATION")
    print("----------------------------------------")
    print(
        rank_ic_long_short[
            ["rank_ic", "long_short_return"]
        ].corr()
    )

    value_weighted_quintile_returns = (
        calculate_monthly_value_weighted_quintile_returns(
            quintile_panel
        )
    )

    print()
    print("VALUE-WEIGHTED QUINTILE RETURNS — FIRST 25 ROWS")
    print("-----------------------------------------------")
    print(
        value_weighted_quintile_returns
        .head(25)
        .to_string(index=False)
    )

    print()
    print("VALUE-WEIGHTED QUINTILE RETURNS — LAST 25 ROWS")
    print("----------------------------------------------")
    print(
        value_weighted_quintile_returns
        .tail(25)
        .to_string(index=False)
    )

    print()
    print("VALUE-WEIGHTED COVERAGE SUMMARY")
    print("-------------------------------")
    print(
        value_weighted_quintile_returns[
            [
                "n_assigned",
                "n_weightable",
                "n_obs",
                "weight_coverage",
            ]
        ].describe()
    )

    print()
    print(
        "Rows with n_weightable < n_assigned:",
        (
            value_weighted_quintile_returns["n_weightable"]
            < value_weighted_quintile_returns["n_assigned"]
        ).sum(),
    )

    print(
        "Rows with weight coverage < 99%:",
        (
            value_weighted_quintile_returns["weight_coverage"]
            < 0.99
        ).sum(),
    )

    value_weighted_quintile_summary = (
        value_weighted_quintile_returns
        .groupby(
            "momentum_quintile",
            observed=True,
        )
        .agg(
            n_months=("portfolio_return", "count"),
            mean_monthly_return=("portfolio_return", "mean"),
            median_monthly_return=("portfolio_return", "median"),
            std_monthly_return=("portfolio_return", "std"),
        )
        .reset_index()
    )

    print()
    print("VALUE-WEIGHTED QUINTILE RETURN SUMMARY")
    print("--------------------------------------")
    print(
        value_weighted_quintile_summary
        .to_string(index=False)
    )

    value_weighted_long_short = (
        calculate_monthly_long_short_returns(
            value_weighted_quintile_returns
        )
    )

    print()
    print("VALUE-WEIGHTED LONG-SHORT RETURNS — FIRST 12 MONTHS")
    print("---------------------------------------------------")
    print(
        value_weighted_long_short
        .head(12)
        .to_string(index=False)
    )

    print()
    print("VALUE-WEIGHTED LONG-SHORT RETURNS — LAST 12 MONTHS")
    print("--------------------------------------------------")
    print(
        value_weighted_long_short
        .tail(12)
        .to_string(index=False)
    )

    weighting_comparison = (
        build_long_short_robustness_comparison(
            {
                "equal_weighted": long_short_returns,
                "value_weighted": value_weighted_long_short,
            }
        )
    )

    print()
    print("EQUAL-WEIGHTED VS VALUE-WEIGHTED LONG-SHORT")
    print("-------------------------------------------")
    print(
        weighting_comparison
        .to_string(index=False)
    )

    ew_vw_monthly = (
        long_short_returns[
            ["date", "long_short_return"]
        ]
        .rename(
            columns={
                "long_short_return": "equal_weighted_return"
            }
        )
        .merge(
            value_weighted_long_short[
                ["date", "long_short_return"]
            ].rename(
                columns={
                    "long_short_return": "value_weighted_return"
                }
            ),
            on="date",
            how="inner",
        )
    )

    print()
    print("EW / VW LONG-SHORT CORRELATION")
    print("------------------------------")
    print(
        ew_vw_monthly[
            [
                "equal_weighted_return",
                "value_weighted_return",
            ]
        ].corr()
    )

    six_two_panel = add_momentum_signal(
        base_panel,
        start_lag=6,
        end_lag=2,
    )

    six_two_panel = add_momentum_eligibility(
        six_two_panel
    )

    six_two_reported_panel = six_two_panel.loc[
        six_two_panel["date"].between(
            REPORT_START,
            REPORT_END,
        )
    ].copy()

    six_two_universe = six_two_reported_panel.loc[
        six_two_reported_panel["in_universe"]
    ].copy()

    baseline_signal_coverage = (
        research_universe["momentum"]
        .notna()
        .mean()
    )

    six_two_signal_coverage = (
        six_two_universe["momentum"]
        .notna()
        .mean()
    )

    print()
    print("SIGNAL COVERAGE — 12-2 VS 6-2")
    print("-----------------------------")
    print(
        f"12-2 coverage: "
        f"{baseline_signal_coverage:.6f}"
    )
    print(
        f"6-2 coverage:  "
        f"{six_two_signal_coverage:.6f}"
    )

    six_two_monthly_rank_ic = (
        calculate_monthly_rank_ic(
            six_two_reported_panel
        )
    )

    rank_ic_robustness = (
        build_rank_ic_robustness_comparison(
            {
                "12-2": monthly_rank_ic,
                "6-2": six_two_monthly_rank_ic,
            }
        )
    )

    print()
    print("RANK IC ROBUSTNESS — 12-2 VS 6-2")
    print("--------------------------------")
    print(
        rank_ic_robustness
        .to_string(index=False)
    )

    six_two_quintile_panel = (
        assign_momentum_quintiles(
            six_two_reported_panel
        )
    )

    six_two_quintile_returns = (
        calculate_monthly_quintile_returns(
            six_two_quintile_panel
        )
    )

    six_two_long_short = (
        calculate_monthly_long_short_returns(
            six_two_quintile_returns
        )
    )

    long_short_robustness = (
        build_long_short_robustness_comparison(
            {
                "12-2": long_short_returns,
                "6-2": six_two_long_short,
            }
        )
    )

    print()
    print("LONG-SHORT ROBUSTNESS — 12-2 VS 6-2")
    print("-----------------------------------")
    print(
        long_short_robustness
        .to_string(index=False)
    )

    top_500_panel = add_universe_membership(
        panel,
        n=500,
    )

    top_1500_panel = add_universe_membership(
        panel,
        n=1500,
    )

    top_500_reported_panel = top_500_panel.loc[
        top_500_panel["date"].between(
            REPORT_START,
            REPORT_END,
        )
    ].copy()

    top_1500_reported_panel = top_1500_panel.loc[
        top_1500_panel["date"].between(
            REPORT_START,
            REPORT_END,
        )
    ].copy()

    top_500_monthly_size = (
        top_500_reported_panel
        .groupby("date")["in_universe"]
        .sum()
    )

    top_1500_monthly_size = (
        top_1500_reported_panel
        .groupby("date")["in_universe"]
        .sum()
    )

    print()
    print("UNIVERSE-SIZE ROBUSTNESS — MONTHLY COUNTS")
    print("-----------------------------------------")
    print(
        "Top 500 range:",
        int(top_500_monthly_size.min()),
        "to",
        int(top_500_monthly_size.max()),
    )
    print(
        "Top 1000 range:",
        int(monthly_universe_size.min()),
        "to",
        int(monthly_universe_size.max()),
    )
    print(
        "Top 1500 range:",
        int(top_1500_monthly_size.min()),
        "to",
        int(top_1500_monthly_size.max()),
    )


    top_500_monthly_rank_ic = (
        calculate_monthly_rank_ic(
            top_500_reported_panel
        )
    )

    top_1500_monthly_rank_ic = (
        calculate_monthly_rank_ic(
            top_1500_reported_panel
        )
    )

    universe_rank_ic_robustness = (
        build_rank_ic_robustness_comparison(
            {
                "top_500": top_500_monthly_rank_ic,
                "top_1000": monthly_rank_ic,
                "top_1500": top_1500_monthly_rank_ic,
            }
        )
    )

    print()
    print("RANK IC ROBUSTNESS — UNIVERSE SIZE")
    print("----------------------------------")
    print(
        universe_rank_ic_robustness
        .to_string(index=False)
    )

    top_500_quintile_panel = (
        assign_momentum_quintiles(
            top_500_reported_panel
        )
    )

    top_500_quintile_returns = (
        calculate_monthly_quintile_returns(
            top_500_quintile_panel
        )
    )

    top_500_long_short = (
        calculate_monthly_long_short_returns(
            top_500_quintile_returns
        )
    )

    top_1500_quintile_panel = (
        assign_momentum_quintiles(
            top_1500_reported_panel
        )
    )

    top_1500_quintile_returns = (
        calculate_monthly_quintile_returns(
            top_1500_quintile_panel
        )
    )

    top_1500_long_short = (
        calculate_monthly_long_short_returns(
            top_1500_quintile_returns
        )
    )

    universe_long_short_robustness = (
        build_long_short_robustness_comparison(
            {
                "top_500": top_500_long_short,
                "top_1000": long_short_returns,
                "top_1500": top_1500_long_short,
            }
        )
    )

    print()
    print("LONG-SHORT ROBUSTNESS — UNIVERSE SIZE")
    print("-------------------------------------")
    print(
        universe_long_short_robustness
        .to_string(index=False)
    )

    top_500_research_universe = (
        top_500_reported_panel.loc[
            top_500_reported_panel["in_universe"]
        ]
    )

    top_1500_research_universe = (
        top_1500_reported_panel.loc[
            top_1500_reported_panel["in_universe"]
        ]
    )

    top_500_signal_coverage = (
        top_500_research_universe["momentum"]
        .notna()
        .mean()
    )

    top_1000_signal_coverage = (
        research_universe["momentum"]
        .notna()
        .mean()
    )

    top_1500_signal_coverage = (
        top_1500_research_universe["momentum"]
        .notna()
        .mean()
    )

    print()
    print("SIGNAL COVERAGE — UNIVERSE SIZE")
    print("-------------------------------")
    print(
        f"Top 500:  {top_500_signal_coverage:.6f}"
    )
    print(
        f"Top 1000: {top_1000_signal_coverage:.6f}"
    )
    print(
        f"Top 1500: {top_1500_signal_coverage:.6f}"
    )


    # ----------------------------------------
    # RESULTS SUMMARY
    # ----------------------------------------

    master_rank_ic_comparison = (
        build_rank_ic_robustness_comparison(
            {
                "12-2 EW baseline": monthly_rank_ic,
                "12-2 VW": monthly_rank_ic,
                "6-2 EW": six_two_monthly_rank_ic,
                "12-2 EW Top 500": top_500_monthly_rank_ic,
                "12-2 EW Top 1500": top_1500_monthly_rank_ic,
            }
        )
    )

    master_long_short_comparison = (
        build_long_short_robustness_comparison(
            {
                "12-2 EW baseline": long_short_returns,
                "12-2 VW": value_weighted_long_short,
                "6-2 EW": six_two_long_short,
                "12-2 EW Top 500": top_500_long_short,
                "12-2 EW Top 1500": top_1500_long_short,
            }
        )
    )

    master_results = (
        master_rank_ic_comparison
        .merge(
            master_long_short_comparison,
            on=[
                "specification",
                "n_months",
            ],
            how="inner",
        )
    )

    master_results = master_results[
        [
            "specification",
            "n_months",
            "mean_ic",
            "ic_t_stat",
            "positive_ic_fraction",
            "mean_monthly_return",
            "return_t_stat",
            "annualized_mean_return",
            "annualized_volatility",
            "positive_return_fraction",
        ]
    ]

    print()
    print("MASTER EMPIRICAL RESULTS SUMMARY")
    print("--------------------------------")
    print(
        master_results
        .to_string(index=False)
    )


    fig, ax = plt.subplots(
        figsize=(10, 5),
    )

    ax.plot(
        quintile_return_summary["momentum_quintile"],
        quintile_return_summary["mean_monthly_return"],
        marker="o",
        label="Equal-weighted",
    )

    ax.plot(
        value_weighted_quintile_summary["momentum_quintile"],
        value_weighted_quintile_summary["mean_monthly_return"],
        marker="o",
        label="Value-weighted",
    )

    ax.set_title(
        "Average Monthly Momentum Quintile Returns — 12-2 Momentum"
    )
    ax.set_xlabel("Momentum Quintile")
    ax.set_ylabel("Mean Monthly Return")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.yaxis.set_major_formatter(
        PercentFormatter(xmax=1.0, decimals=2)
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "average_quintile_returns.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)



if __name__ == "__main__":
    main()