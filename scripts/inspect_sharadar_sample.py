import pandas as pd
from equity_factor_research.data.providers.sharadar import (
    add_sharadar_monthly_returns,
    aggregate_sharadar_market_cap_to_monthly,
    aggregate_sharadar_prices_to_monthly,
    assemble_sharadar_monthly_panel,
    normalize_sharadar_metadata,
)
from equity_factor_research.data.quality import assess_data_quality
from equity_factor_research.data.schema import validate_equity_panel
from equity_factor_research.data.transforms import construct_equity_panel
from equity_factor_research.data.universe import (
    add_base_eligibility,
    add_lagged_market_cap,
    add_universe_membership,
)


BASE_URL = "https://api.sharadar.com/v1.0/data"
API_KEY = "test-api-key"

TICKERS = "AAPL,MSFT,JPM"
START_DATE = "2025-01-01"
END_DATE = "2025-12-31"


stocks_url = (
    f"{BASE_URL}/stocks"
    f"?api_key={API_KEY}"
    f"&format=csv"
    f"&ticker={TICKERS}"
    f"&from={START_DATE}"
    f"&to={END_DATE}"
    f"&fields=ticker,date,close,closeadj"
)

daily_url = (
    f"{BASE_URL}/daily"
    f"?api_key={API_KEY}"
    f"&format=csv"
    f"&ticker={TICKERS}"
    f"&from={START_DATE}"
    f"&to={END_DATE}"
    f"&fields=ticker,date,marketcap"
)

tickers_url = (
    f"{BASE_URL}/tickers"
    f"?api_key={API_KEY}"
    f"&format=csv"
    f"&table=fundamentals"
    f"&ticker={TICKERS}"
    f"&fields=table,permaticker,ticker,exchange,isdelisted,category,"
    f"firstpricedate,lastpricedate"
)

actions_url = (
    f"{BASE_URL}/actions"
    f"?api_key={API_KEY}"
    f"&format=csv"
    f"&ticker={TICKERS}"
    f"&from={START_DATE}"
    f"&to={END_DATE}"
)


stocks = pd.read_csv(stocks_url)
daily = pd.read_csv(daily_url)
tickers = pd.read_csv(tickers_url)
actions = pd.read_csv(actions_url)


print("\nSTOCKS")
print(stocks.head())
print(stocks.shape)
print(stocks.columns.tolist())
print(stocks.dtypes)

print("\nDAILY")
print(daily.head())
print(daily.shape)
print(daily.columns.tolist())
print(daily.dtypes)

print("\nTICKERS")
print(tickers)
print(tickers.shape)
print(tickers.columns.tolist())
print(tickers.dtypes)

print("\nACTIONS")
print(actions.head(20))
print(actions.shape)
print(actions.columns.tolist())
print(actions.dtypes)


print("\nTicker → permaticker counts")
print(
    tickers.groupby("ticker")["permaticker"]
    .nunique()
    .sort_values(ascending=False)
)

print("\nPermaticker → ticker counts")
print(
    tickers.groupby("permaticker")["ticker"]
    .nunique()
    .sort_values(ascending=False)
)

print("\nExchange values")
print(tickers["exchange"].value_counts(dropna=False))

print("\nCategory values")
print(tickers["category"].value_counts(dropna=False))

print("\nAction types")
print(actions["action"].value_counts(dropna=False))


id_map = (
    tickers[["ticker", "permaticker"]]
    .rename(columns={"permaticker": "security_id"})
)

stocks_with_id = stocks.merge(
    id_map,
    on="ticker",
    how="left",
    validate="many_to_one",
)

daily_with_id = daily.merge(
    id_map,
    on="ticker",
    how="left",
    validate="many_to_one",
)

assert stocks_with_id["security_id"].notna().all()
assert daily_with_id["security_id"].notna().all()

monthly_prices = aggregate_sharadar_prices_to_monthly(
    stocks_with_id
)
monthly_prices = add_sharadar_monthly_returns(
    monthly_prices
)

monthly_market_caps = (
    aggregate_sharadar_market_cap_to_monthly(
        daily_with_id
    )
)

normalized_metadata = normalize_sharadar_metadata(
    tickers
)

monthly_delistings = pd.DataFrame(
    {
        "security_id": pd.Series(dtype="int64"),
        "date": pd.Series(dtype="datetime64[ns]"),
        "delisting_event": pd.Series(dtype="bool"),
        "delisting_return": pd.Series(dtype="float64"),
    }
)

assembled_panel = assemble_sharadar_monthly_panel(
    monthly_prices,
    monthly_market_caps,
    normalized_metadata,
    monthly_delistings,
)

print("\nMONTHLY PRICES")
print(monthly_prices.head(15))
print(monthly_prices.shape)

print("\nMONTHLY MARKET CAPS")
print(monthly_market_caps.head(15))
print(monthly_market_caps.shape)

print("\nNORMALIZED METADATA")
print(normalized_metadata)

print("\nASSEMBLED PANEL")
print(assembled_panel.head(20))
print(assembled_panel.shape)
print(assembled_panel.dtypes)

print("\nROWS PER SECURITY")
print(
    assembled_panel.groupby("security_id")
    .size()
)


equity_panel = construct_equity_panel(assembled_panel)

validate_equity_panel(equity_panel)

validated_panel = add_lagged_market_cap(equity_panel)
validated_panel = add_base_eligibility(validated_panel)
validated_panel = add_universe_membership(validated_panel)
quality_summary = assess_data_quality(validated_panel)

print("\nFINAL VALIDATED PANEL")
print(validated_panel.head(20))
print(validated_panel.shape)

print("\nQUALITY SUMMARY")
print(quality_summary)

print("\nBASE ELIGIBILITY")
print(
    validated_panel["base_eligible"]
    .value_counts(dropna=False)
)

print("\nUNIVERSE MEMBERSHIP BY MONTH")
print(
    validated_panel.groupby("date")["in_universe"]
    .sum()
)

print("\nMISSING VALUES")
print(
    validated_panel[
        [
            "return_ex_delist",
            "total_return",
            "price",
            "shares_outstanding",
            "market_cap",
            "lagged_market_cap",
        ]
    ].isna().sum()
)

print("\nDUPLICATE SECURITY-MONTHS")
print(
    validated_panel.duplicated(
        subset=["security_id", "date"]
    ).sum()
)