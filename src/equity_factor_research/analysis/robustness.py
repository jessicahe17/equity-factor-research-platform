import pandas as pd

from equity_factor_research.analysis.rank_ic import (
    summarize_rank_ic,
)
from equity_factor_research.analysis.portfolio_sorts import (
    summarize_long_short_returns,
)


def build_rank_ic_robustness_comparison(
    specifications: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build a compact comparison of Rank IC robustness specifications."""

    records = []

    for specification, monthly_ic in specifications.items():
        summary = summarize_rank_ic(monthly_ic)

        records.append(
            {
                "specification": specification,
                "n_months": summary["n_months"],
                "mean_ic": summary["mean_ic"],
                "ic_t_stat": summary["ic_t_stat"],
                "positive_ic_fraction": summary[
                    "positive_ic_fraction"
                ],
            }
        )

    columns = [
        "specification",
        "n_months",
        "mean_ic",
        "ic_t_stat",
        "positive_ic_fraction",
    ]

    return pd.DataFrame(records, columns=columns)


def build_long_short_robustness_comparison(
    specifications: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build a compact comparison of long-short robustness specifications."""

    records = []

    for specification, long_short_returns in specifications.items():
        summary = summarize_long_short_returns(
            long_short_returns
        )

        records.append(
            {
                "specification": specification,
                "n_months": summary["n_months"],
                "mean_monthly_return": summary["mean_monthly_return"],
                "return_t_stat": summary["return_t_stat"],
                "annualized_mean_return": summary["annualized_mean_return"],
                "annualized_volatility": summary["annualized_volatility"],
                "positive_return_fraction": summary["positive_return_fraction"],
            }
        )

    columns = [
        "specification",
        "n_months",
        "mean_monthly_return",
        "return_t_stat",
        "annualized_mean_return",
        "annualized_volatility",
        "positive_return_fraction",
    ]

    return pd.DataFrame(records, columns=columns)