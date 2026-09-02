import pandas as pd
import numpy as np


SHARADAR_EXCHANGE_MAP = {
    "NYSE": "NYSE",
    "NASDAQ": "NASDAQ",
    "NYSEMKT": "AMEX",
}


def normalize_sharadar_exchange(
    exchange: pd.Series,
) -> pd.Series:
    """
    Normalizes Sharadar exchange names to standard names.
    Unsupported exchanges remain unchanged so downstream filters can reject them.
    """

    return exchange.replace(SHARADAR_EXCHANGE_MAP)


def aggregate_sharadar_prices_to_monthly(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate daily Sharadar prices to monthly observations.

    Selects the final available trading observation for each security
    and calendar month, then labels it with the calendar month-end date.
    """

    df = prices.copy()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["security_id", "date"])
    df["month_end_date"] = df["date"] + pd.offsets.MonthEnd(0)

    monthly_df = df.groupby(["security_id", "month_end_date"]).tail(1).copy()

    monthly_df["date"] = monthly_df["month_end_date"]

    monthly_df = monthly_df.rename(
        columns={
            "close": "price",
            "closeadj": "adjusted_price",
        }
    )

    monthly_df = monthly_df[["security_id", "date", "price", "adjusted_price"]]

    monthly_df = monthly_df.sort_values(["security_id", "date"]).reset_index(drop=True)

    return monthly_df


def add_sharadar_monthly_returns(
    monthly_prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add monthly returns derived from adjusted Sharadar prices.

    A return is calculated only when the previous observation for the
    same security belongs to the immediately preceding calendar month.
    """

    df = monthly_prices.copy()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["security_id", "date"])

    grouped = df.groupby("security_id")
    prev_adj_price = grouped["adjusted_price"].shift(1)
    prev_date = grouped["date"].shift(1)

    current_month = df["date"].dt.to_period("M")
    prev_month = prev_date.dt.to_period("M")
    
    is_consecutive = (prev_month == (current_month - 1))

    raw_return = df["adjusted_price"] / prev_adj_price - 1

    df["return_ex_delist"] = raw_return.where(is_consecutive)

    df = df.reset_index(drop=True)

    return df


def aggregate_sharadar_market_cap_to_monthly(
    market_caps: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate daily Sharadar market capitalization to monthly observations.

    Selects the final available observation for each security and calendar
    month, labels it with the calendar month-end date, and converts market
    capitalization from USD millions to USD.
    """

    df = market_caps.copy()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["security_id", "date"])
    df["month_end_date"] = df["date"] + pd.offsets.MonthEnd(0)

    monthly_df = df.groupby(["security_id", "month_end_date"]).tail(1).copy()
    monthly_df["date"] = monthly_df["month_end_date"]

    monthly_df["market_cap"] = monthly_df["marketcap"] * 1_000_000

    monthly_df = monthly_df[["security_id", "date", "market_cap"]]
    monthly_df = monthly_df.sort_values(["security_id", "date"]).reset_index(drop=True)

    return monthly_df


def normalize_sharadar_metadata(
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize Sharadar security metadata to the standard equity schema.

    Assumes metadata have been restricted to Sharadar's fundamentals
    universe of primary-class common stocks.
    """

    df = metadata.copy()

    df = df.rename(columns={"permaticker": "security_id"})
    df["exchange"] = normalize_sharadar_exchange(df["exchange"])
    df["security_type"] = "COMMON"

    return df[
        [
            "security_id",
            "ticker",
            "exchange",
            "security_type",
        ]
    ]


def aggregate_sharadar_delistings_to_monthly(
    delistings: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate Sharadar delisting events to monthly observations.

    Converts delisting dates to calendar month-end, keeps one event per
    security-month, and leaves delisting returns missing because Sharadar
    does not provide a CRSP-style delisting return.
    """

    df = delistings.copy()

    df["date"] = pd.to_datetime(df["date"])
    df["date"] = df["date"] + pd.offsets.MonthEnd(0)

    df = df.drop_duplicates(subset=["security_id", "date"])

    df["delisting_event"] = True
    df["delisting_return"] = np.nan

    df = df[["security_id", "date", "delisting_event", "delisting_return"]]
    df = df.sort_values(["security_id", "date"]).reset_index(drop=True)

    return df


def assemble_sharadar_monthly_panel(
    monthly_prices: pd.DataFrame,
    monthly_market_caps: pd.DataFrame,
    metadata: pd.DataFrame,
    monthly_delistings: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assemble normalized Sharadar components into a standardized monthly panel.
    """
    
    df = monthly_prices.copy()

    df = df.merge(
        monthly_market_caps,
        on=["security_id", "date"],
        how="left",
        validate="one_to_one",
    )

    df = df.merge(
        monthly_delistings,
        on=["security_id", "date"],
        how="outer",
        validate="one_to_one",
    )

    df = df.merge(
        metadata,
        on="security_id",
        how="left",
        validate="many_to_one",
    )

    df["delisting_event"] = (
        df["delisting_event"]
        .fillna(False)
        .astype(bool)
    )

    df["shares_outstanding"] = np.nan

    output_columns = [
        "security_id",
        "date",
        "ticker",
        "exchange",
        "security_type",
        "return_ex_delist",
        "delisting_event",
        "delisting_return",
        "price",
        "shares_outstanding",
        "market_cap",
    ]

    df = df[output_columns]
    df = df.sort_values(["security_id", "date"]).reset_index(drop=True)

    return df