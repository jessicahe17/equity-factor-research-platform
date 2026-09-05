import pandas as pd
import numpy as np


def calculate_monthly_rank_ic(panel: pd.DataFrame) -> pd.DataFrame:
    """Calculate monthly Spearman Rank IC for Momentum."""

    research_sample = panel.loc[
        panel["in_universe"]
        & panel["momentum_eligible"]
    ].copy()

    rows = []

    for date, group in research_sample.groupby("date"):
        valid = group.dropna(subset=["momentum", "total_return"])

        n_obs = len(valid)

        if (
            n_obs < 2
            or valid["momentum"].nunique() < 2
            or valid["total_return"].nunique() < 2
        ):
            rank_ic = float("nan")
        else:
            rank_ic = valid["momentum"].corr(
                valid["total_return"],
                method="spearman",
            )

        rows.append(
            {
                "date": date,
                "rank_ic": rank_ic,
                "n_obs": n_obs,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("date")
        .reset_index(drop=True)
    )


def summarize_rank_ic(monthly_ic: pd.DataFrame) -> pd.Series:
    """Summarize the time series of monthly Rank IC values."""

    valid_ic = monthly_ic["rank_ic"].dropna()

    n_months = len(valid_ic)

    if n_months == 0:
        mean_ic = float("nan")
        median_ic = float("nan")
        std_ic = float("nan")
        positive_ic_fraction = float("nan")
        ic_t_stat = float("nan")

    else:
        mean_ic = valid_ic.mean()
        median_ic = valid_ic.median()
        positive_ic_fraction = (valid_ic > 0).mean()

        if n_months < 2:
            std_ic = float("nan")
            ic_t_stat = float("nan")
        else:
            std_ic = valid_ic.std()

            if np.isclose(std_ic, 0.0, rtol=0.0, atol=1e-12):
                ic_t_stat = float("nan")
            else:
                ic_t_stat = (
                    mean_ic
                    / (std_ic / np.sqrt(n_months))
                )

    return pd.Series(
        {
            "n_months": n_months,
            "mean_ic": mean_ic,
            "median_ic": median_ic,
            "std_ic": std_ic,
            "positive_ic_fraction": positive_ic_fraction,
            "ic_t_stat": ic_t_stat,
        }
    )


def summarize_rank_ic_autocorrelation(monthly_ic: pd.DataFrame) -> pd.Series:
    """Summarize lag-1 autocorrelation of monthly Rank IC values."""

    data = monthly_ic[["date", "rank_ic"]].copy()

    data = data.sort_values("date").reset_index(drop=True)

    data["previous_date"] = data["date"].shift(1)
    data["previous_ic"] = data["rank_ic"].shift(1)

    current_month = data["date"].dt.to_period("M")
    previous_month = data["previous_date"].dt.to_period("M")

    consecutive = (current_month == previous_month + 1)

    valid_pairs = data.loc[
        consecutive
        & data["rank_ic"].notna()
        & data["previous_ic"].notna()
    ]

    n_pairs = len(valid_pairs)

    if (
        n_pairs < 2
        or valid_pairs["rank_ic"].nunique() < 2
        or valid_pairs["previous_ic"].nunique() < 2
    ):
        lag1_autocorr = float("nan")
    else:
        lag1_autocorr = valid_pairs["rank_ic"].corr(
            valid_pairs["previous_ic"]
        )

    return pd.Series(
        {
            "n_pairs": n_pairs,
            "lag1_autocorr": lag1_autocorr,
        }
    )


def calculate_rank_ic_rolling_diagnostics(monthly_ic: pd.DataFrame) -> pd.DataFrame:
    """Calculate 12-month rolling diagnostics for monthly Rank IC."""

    data = (
        monthly_ic[["date", "rank_ic"]]
        .sort_values("date")
        .set_index("date")
    )

    full_dates = pd.date_range(
        start=data.index.min(),
        end=data.index.max(),
        freq="ME",
    )

    data = data.reindex(full_dates)

    rolling = data["rank_ic"].rolling(
        window=12,
        min_periods=12,
    )

    positive = (
        data["rank_ic"]
        .gt(0)
        .where(data["rank_ic"].notna())
    )

    result = pd.DataFrame(
        {
            "date": full_dates,
            "rolling_mean_ic_12m": rolling.mean().to_numpy(),
            "rolling_std_ic_12m": rolling.std().to_numpy(),
            "rolling_positive_fraction_12m": (
                positive
                .rolling(window=12, min_periods=12)
                .mean()
                .to_numpy()
            ),
        }
    )

    return result


def summarize_rank_ic_by_subperiod(monthly_ic: pd.DataFrame) -> pd.DataFrame:
    """Summarize monthly Rank IC across fixed calendar subperiods."""

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
            (monthly_ic["date"] >= start_date)
            & (monthly_ic["date"] <= end_date)
        )

        period_data = monthly_ic.loc[mask]

        summary = summarize_rank_ic(period_data)

        rows.append(
            {
                "subperiod": name,
                "start_date": start_date,
                "end_date": end_date,
                **summary.to_dict(),
            }
        )

    return pd.DataFrame(rows)