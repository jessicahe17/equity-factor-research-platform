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


def add_momentum_signal(
    panel: pd.DataFrame,
    start_lag: int = 12,
    end_lag: int = 2,
) -> pd.DataFrame:
    """Add Momentum using total returns from t-start_lag through t-end_lag.

    The signal is labeled by outcome month t. By default, this computes
    conventional 12-2 Momentum using returns from t-12 through t-2.
    All required calendar months must exist with non-missing returns.
    """

    for name, value in {
        "start_lag": start_lag,
        "end_lag": end_lag,
    }.items():
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer.")

    if start_lag < end_lag:
        raise ValueError("start_lag must be greater than or equal to end_lag.")

    window_size = start_lag - end_lag + 1

    result = panel.copy()
    momentum_parts = []

    for security_id, group in result.groupby("security_id"):
        group = group.sort_values("date")
        returns = _build_complete_monthly_returns(group)

        signal_returns = returns.shift(end_lag)
        gross_returns = 1 + signal_returns

        momentum = (
            gross_returns
            .rolling(
                window=window_size,
                min_periods=window_size,
            )
            .apply(np.prod, raw=True)
            - 1
        )

        group_momentum = pd.DataFrame(
            {
                "security_id": security_id,
                "date": group["date"].to_numpy(),
                "momentum": momentum.reindex(
                    group["date"]
                ).to_numpy(),
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


def add_momentum_eligibility(
    panel: pd.DataFrame,
    start_lag: int = 12,
    end_lag: int = 2,
) -> pd.DataFrame:
    """
    Add whether each row has valid history for the Momentum signal.

    By default, this checks the conventional 12-2 specification. All required
    calendar months from t-start_lag through t-end_lag must exist with
    non-missing total returns.
    """

    for name, value in {
        "start_lag": start_lag,
        "end_lag": end_lag,
    }.items():
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer.")

    if start_lag < end_lag:
        raise ValueError("start_lag must be greater than or equal to end_lag.")

    window_size = start_lag - end_lag + 1

    result = panel.copy()
    eligibility_parts = []

    for security_id, group in result.groupby("security_id"):
        group = group.sort_values("date")
        returns = _build_complete_monthly_returns(group)

        required_returns = returns.shift(end_lag)

        eligible = (
            required_returns
            .notna()
            .rolling(
                window=window_size,
                min_periods=window_size,
            )
            .sum()
            .eq(window_size)
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