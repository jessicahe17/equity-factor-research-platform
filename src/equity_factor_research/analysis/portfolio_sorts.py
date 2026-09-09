import numpy as np
import pandas as pd


def assign_momentum_quintiles(panel: pd.DataFrame) -> pd.DataFrame:
    """Assign eligible securities to monthly Momentum quintiles."""

    result = panel.copy()

    result["momentum_quintile"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int64",
    )

    sort_eligible = (
        result["in_universe"]
        & result["momentum_eligible"]
        & result["momentum"].notna()
    )

    eligible = result.loc[sort_eligible].copy()

    eligible = eligible.sort_values(
        ["date", "momentum", "security_id"]
    )

    for _, group in eligible.groupby("date", sort=False):
        if len(group) < 5:
            continue

        ranks = group["momentum"].rank(
            method="first",
            ascending=True,
        )

        quintiles = pd.qcut(
            ranks,
            q=5,
            labels=[1, 2, 3, 4, 5],
        )

        result.loc[
            group.index,
            "momentum_quintile",
        ] = quintiles.astype(int).to_numpy()

    return result


def calculate_monthly_quintile_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Calculate equal-weighted monthly returns for Momentum quintiles."""

    assigned = panel.loc[
        panel["momentum_quintile"].notna()
    ].copy()

    rows = []

    for (date, quintile), group in assigned.groupby(
        ["date", "momentum_quintile"],
        sort=True,
        observed=True,
    ):
        observed_returns = group["total_return"].dropna()

        rows.append(
            {
                "date": date,
                "momentum_quintile": int(quintile),
                "portfolio_return": observed_returns.mean(),
                "n_assigned": len(group),
                "n_obs": len(observed_returns),
            }
        )

    return pd.DataFrame(rows)


def calculate_monthly_long_short_returns(
    quintile_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate monthly Q5-minus-Q1 Momentum returns."""

    relevant = quintile_returns.loc[
        quintile_returns["momentum_quintile"].isin([1, 5]),
        [
            "date",
            "momentum_quintile",
            "portfolio_return",
        ],
    ]

    wide = relevant.pivot(
        index="date",
        columns="momentum_quintile",
        values="portfolio_return",
    )

    wide = wide.reindex(columns=[1, 5])

    result = pd.DataFrame(
        {
            "date": wide.index,
            "q1_return": wide[1],
            "q5_return": wide[5],
            "long_short_return": wide[5] - wide[1],
        }
    ).reset_index(drop=True)

    return result


def summarize_long_short_returns(
    long_short_returns: pd.DataFrame,
) -> pd.Series:
    """Summarize the monthly Momentum long-short return series."""

    returns = long_short_returns["long_short_return"].dropna()

    n_months = len(returns)

    if n_months == 0:
        return pd.Series(
            {
                "n_months": 0,
                "mean_monthly_return": np.nan,
                "median_monthly_return": np.nan,
                "std_monthly_return": np.nan,
                "positive_return_fraction": np.nan,
                "return_t_stat": np.nan,
                "annualized_mean_return": np.nan,
                "annualized_volatility": np.nan,
            }
        )

    mean_return = returns.mean()
    median_return = returns.median()
    positive_fraction = (returns > 0).mean()

    if n_months >= 2:
        std_return = returns.std()
    else:
        std_return = np.nan

    if (
        n_months >= 2
        and not np.isclose(
            std_return,
            0.0,
            rtol=0.0,
            atol=1e-12,
        )
    ):
        return_t_stat = (
            mean_return
            / (std_return / np.sqrt(n_months))
        )
    else:
        return_t_stat = np.nan

    annualized_mean_return = mean_return * 12

    annualized_volatility = (
        std_return * np.sqrt(12)
        if pd.notna(std_return)
        else np.nan
    )

    return pd.Series(
        {
            "n_months": n_months,
            "mean_monthly_return": mean_return,
            "median_monthly_return": median_return,
            "std_monthly_return": std_return,
            "positive_return_fraction": positive_fraction,
            "return_t_stat": return_t_stat,
            "annualized_mean_return": annualized_mean_return,
            "annualized_volatility": annualized_volatility,
        }
    )


def calculate_long_short_rolling_diagnostics(
    long_short_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate strict 12-month rolling diagnostics for long-short returns."""

    data = (
        long_short_returns[
            ["date", "long_short_return"]
        ]
        .sort_values("date")
        .set_index("date")
    )

    full_dates = pd.date_range(
        start=data.index.min(),
        end=data.index.max(),
        freq="ME",
    )

    data = data.reindex(full_dates)

    rolling_returns = data["long_short_return"].rolling(
        window=12,
        min_periods=12,
    )

    positive_returns = (
        data["long_short_return"]
        .gt(0)
        .where(data["long_short_return"].notna())
    )

    return pd.DataFrame(
        {
            "date": full_dates,
            "rolling_mean_return_12m": (
                rolling_returns.mean().to_numpy()
            ),
            "rolling_std_return_12m": (
                rolling_returns.std().to_numpy()
            ),
            "rolling_positive_fraction_12m": (
                positive_returns
                .rolling(
                    window=12,
                    min_periods=12,
                )
                .mean()
                .to_numpy()
            ),
        }
    )


def summarize_long_short_returns_by_subperiod(
    long_short_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize long-short returns across fixed calendar subperiods."""

    subperiods = [
        ("2001-2008", "2001-01-01", "2008-12-31"),
        ("2009-2016", "2009-01-01", "2016-12-31"),
        ("2017-2025", "2017-01-01", "2025-12-31"),
    ]

    rows = []

    for name, start, end in subperiods:
        start_date = pd.Timestamp(start)
        end_date = pd.Timestamp(end)

        mask = (
            (long_short_returns["date"] >= start_date)
            & (long_short_returns["date"] <= end_date)
        )

        period_data = long_short_returns.loc[mask]

        summary = summarize_long_short_returns(period_data)

        rows.append(
            {
                "subperiod": name,
                "start_date": start_date,
                "end_date": end_date,
                **summary.to_dict(),
            }
        )

    return pd.DataFrame(rows)


def calculate_monthly_value_weighted_quintile_returns(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate value-weighted monthly Momentum quintile returns."""

    assigned = panel.loc[panel["momentum_quintile"].notna()].copy()

    records = []

    for (date, quintile), group in assigned.groupby(
        ["date", "momentum_quintile"],
        sort=True,
    ):
        n_assigned = len(group)

        valid_weight = (
            group["lagged_market_cap"].notna()
            & (group["lagged_market_cap"] > 0)
        )

        weightable = group.loc[valid_weight]

        n_weightable = int(valid_weight.sum())

        assigned_weight = weightable["lagged_market_cap"].sum()

        observed = weightable.loc[
            weightable["total_return"].notna()
        ]

        n_obs = len(observed)

        observed_weight = observed["lagged_market_cap"].sum()

        if assigned_weight > 0:
            weight_coverage = observed_weight / assigned_weight
        else:
            weight_coverage = np.nan

        if n_obs == 0:
            portfolio_return = np.nan
        else:
            portfolio_return = (
                observed["lagged_market_cap"]
                * observed["total_return"]
            ).sum() / observed_weight

        records.append(
            {
                "date": date,
                "momentum_quintile": quintile,
                "portfolio_return": portfolio_return,
                "n_assigned": n_assigned,
                "n_weightable": n_weightable,
                "n_obs": n_obs,
                "weight_coverage": weight_coverage,
            }
        )

    columns = [
        "date",
        "momentum_quintile",
        "portfolio_return",
        "n_assigned",
        "n_weightable",
        "n_obs",
        "weight_coverage",
    ]

    return pd.DataFrame(records, columns=columns)