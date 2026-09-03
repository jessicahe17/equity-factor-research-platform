import pandas as pd
import numpy as np


def _build_complete_monthly_returns(group: pd.DataFrame) -> pd.Series:
    start = group["date"].min()
    end = group["date"].max()

    full_dates = pd.date_range(
        start=start,
        end=end,
        freq="ME",
    )

    return (
        group
        .set_index("date")["total_return"]
        .reindex(full_dates)
    )


def add_momentum_signal(panel: pd.DataFrame) -> pd.DataFrame:
    """Add 12-2 momentum using total returns from months t-12 through t-2.

    The signal is labeled by outcome month t. The immediately preceding
    month t-1 is skipped, and all 11 required calendar months must exist
    with non-missing returns.
    """

    result = panel.copy()
    momentum_parts = []

    for security_id, group in result.groupby("security_id"):
        group = group.sort_values("date")
        returns = _build_complete_monthly_returns(group)

        signal_returns = returns.shift(2)
        gross_returns = 1 + signal_returns

        momentum = (
            gross_returns
            .rolling(window=11, min_periods=11)
            .apply(np.prod, raw=True)
            - 1
        )

        group_momentum = pd.DataFrame(
            {
                "security_id": security_id,
                "date": group["date"].to_numpy(),
                "momentum": momentum.reindex(group["date"]).to_numpy(),
            }
        )

        momentum_parts.append(group_momentum)

    momentum_panel = pd.concat(
        momentum_parts,
        ignore_index=True,
    )

    result = result.merge(
        momentum_panel,
        on=["security_id", "date"],
        how="left",
        validate="one_to_one",
        sort=False,
    )

    return result


def add_momentum_eligibility(panel: pd.DataFrame) -> pd.DataFrame:
    """Add whether each row has valid history for a 12-2 momentum signal."""

    result = panel.copy()
    eligibility_parts = []

    for security_id, group in result.groupby("security_id"):
        group = group.sort_values("date")
        returns = _build_complete_monthly_returns(group)
        required_returns = returns.shift(2)

        eligible = (
            required_returns
            .notna()
            .rolling(window=11, min_periods=11)
            .sum()
            .eq(11)
        )

        group_eligibility = pd.DataFrame(
            {
                "security_id": security_id,
                "date": group["date"].to_numpy(),
                "momentum_eligible": eligible.reindex(
                    group["date"]
                ).to_numpy(),
            }
        )

        eligibility_parts.append(group_eligibility)

    eligibility_panel = pd.concat(
        eligibility_parts,
        ignore_index=True,
    )

    result = result.merge(
        eligibility_panel,
        on=["security_id", "date"],
        how="left",
        validate="one_to_one",
        sort=False,
    )

    return result


def find_momentum_inconsistencies(panel: pd.DataFrame) -> pd.Series:
    """Flag rows where Momentum eligibility and signal availability disagree."""

    return (
        panel["momentum_eligible"]
        != panel["momentum"].notna()
    )


def summarize_momentum_diagnostics(panel: pd.DataFrame) -> pd.DataFrame:
    """Summarize monthly Momentum coverage and signal distribution."""

    universe = panel.loc[panel["in_universe"]].copy()

    diagnostics = (
        universe
        .groupby("date")
        .agg(
            universe_count=("security_id", "size"),
            eligible_count=("momentum_eligible", "sum"),
            momentum_mean=("momentum", "mean"),
            momentum_median=("momentum", "median"),
            momentum_std=("momentum", "std"),
            momentum_p10=("momentum", lambda x: x.quantile(0.10)),
            momentum_p90=("momentum", lambda x: x.quantile(0.90)),
        )
        .reset_index()
    )

    diagnostics["coverage"] = (
        diagnostics["eligible_count"]
        / diagnostics["universe_count"]
    )

    return diagnostics[
        [
            "date",
            "universe_count",
            "eligible_count",
            "coverage",
            "momentum_mean",
            "momentum_median",
            "momentum_std",
            "momentum_p10",
            "momentum_p90",
        ]
    ]