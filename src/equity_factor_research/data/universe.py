import pandas as pd


VALID_EXCHANGES = ("NYSE", "AMEX", "NASDAQ")
COMMON_STOCK_TYPE = "COMMON"


def add_lagged_market_cap(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.copy()

    ordered = result.sort_values(["security_id", "date"]).copy()

    previous_market_cap = (
        ordered
        .groupby("security_id")["market_cap"]
        .shift(1)
    )

    previous_date = (
        ordered
        .groupby("security_id")["date"]
        .shift(1)
    )

    current_month = ordered["date"].dt.to_period("M")
    previous_month = previous_date.dt.to_period("M")

    consecutive_month = (previous_month == current_month - 1)

    ordered["lagged_market_cap"] = (
        previous_market_cap.where(consecutive_month)
    )

    result["lagged_market_cap"] = ordered["lagged_market_cap"]

    return result


def add_base_eligibility(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.copy()

    result["base_eligible"] = (
        result["exchange"].isin(VALID_EXCHANGES)
        & (result["security_type"] == COMMON_STOCK_TYPE)
    )

    return result


def add_universe_membership(
    panel: pd.DataFrame,
    n: int = 1000,
) -> pd.DataFrame:
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer.")

    result = panel.copy()
    result["in_universe"] = False

    candidates = result.loc[
        result["lagged_base_eligible"].eq(True)
        & result["lagged_market_cap"].notna()
    ].copy()

    candidates = candidates.sort_values(
        [
            "date",
            "lagged_market_cap",
            "security_id",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    )

    selected_indices = (
        candidates
        .groupby("date", sort=False)
        .head(n)
        .index
    )

    result.loc[selected_indices, "in_universe"] = True

    return result


def add_lagged_base_eligibility(panel: pd.DataFrame) -> pd.DataFrame:
    """Add prior-calendar-month base eligibility."""

    result = panel.copy()

    previous = result[
        [
            "security_id",
            "date",
            "base_eligible",
        ]
    ].copy()

    previous["date"] = (
        previous["date"] + pd.offsets.MonthEnd(1)
    )

    previous = previous.rename(
        columns={
            "base_eligible": "lagged_base_eligible",
        }
    )

    result = result.merge(
        previous,
        on=["security_id", "date"],
        how="left",
        validate="one_to_one",
    )

    return result