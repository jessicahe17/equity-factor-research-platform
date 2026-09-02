import pandas as pd
import numpy as np


CORE_DATA_COLUMNS = (
    "return_ex_delist",
    "total_return",
    "price",
    "shares_outstanding",
    "market_cap",
)


def find_invalid_prices(panel: pd.DataFrame) -> pd.Series:
    return panel["price"] < 0


def find_invalid_shares(panel: pd.DataFrame) -> pd.Series:
    return panel["shares_outstanding"] < 0


def find_invalid_total_returns(panel: pd.DataFrame) -> pd.Series:
    return panel["total_return"] < -1


def find_market_cap_inconsistencies(
    panel: pd.DataFrame,
    rtol: float = 1e-8,
    atol: float = 1e-8,
) -> pd.Series:
    complete = (
        panel["price"].notna()
        & panel["shares_outstanding"].notna()
        & panel["market_cap"].notna()
    )

    implied_market_cap = panel["price"] * panel["shares_outstanding"]

    consistent = pd.Series(
        np.isclose(
            panel["market_cap"],
            implied_market_cap,
            rtol=rtol,
            atol=atol,
        ),
        index=panel.index,
    )

    return complete & ~consistent


def find_zero_prices(panel: pd.DataFrame) -> pd.Series:
    return panel["price"] == 0


def find_zero_shares(panel: pd.DataFrame) -> pd.Series:
    return panel["shares_outstanding"] == 0


def find_delisting_inconsistencies(
    panel: pd.DataFrame,
) -> pd.Series:
    return (
        ~panel["delisting_event"]
        & panel["delisting_return"].notna()
    )


def find_missing_delisting_returns(
    panel: pd.DataFrame,
) -> pd.Series:
    return (
        panel["delisting_event"]
        & panel["delisting_return"].isna()
    )


def find_missing_core_fields(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    return panel.loc[:, list(CORE_DATA_COLUMNS)].isna()


def assess_data_quality(panel: pd.DataFrame) -> pd.Series:
    missing_core = find_missing_core_fields(panel)

    return pd.Series(
        {
            "invalid_prices": int(find_invalid_prices(panel).sum()),
            "invalid_shares": int(find_invalid_shares(panel).sum()),
            "invalid_total_returns": int(
                find_invalid_total_returns(panel).sum()
            ),
            "market_cap_inconsistencies": int(
                find_market_cap_inconsistencies(panel).sum()
            ),
            "zero_prices": int(find_zero_prices(panel).sum()),
            "zero_shares": int(find_zero_shares(panel).sum()),
            "delisting_inconsistencies": int(
                find_delisting_inconsistencies(panel).sum()
            ),
            "missing_delisting_returns": int(
                find_missing_delisting_returns(panel).sum()
            ),
            "missing_return_ex_delist": int(
                missing_core["return_ex_delist"].sum()
            ),
            "missing_total_return": int(
                missing_core["total_return"].sum()
            ),
            "missing_price": int(
                missing_core["price"].sum()
            ),
            "missing_shares_outstanding": int(
                missing_core["shares_outstanding"].sum()
            ),
            "missing_market_cap": int(
                missing_core["market_cap"].sum()
            ),
        },
        name="count",
    )