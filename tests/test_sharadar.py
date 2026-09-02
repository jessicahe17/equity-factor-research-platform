import pandas as pd
import numpy as np
import pytest

from equity_factor_research.data.providers.sharadar import (
    normalize_sharadar_exchange,
    aggregate_sharadar_prices_to_monthly,
    add_sharadar_monthly_returns,
    aggregate_sharadar_market_cap_to_monthly,
    normalize_sharadar_metadata,
    aggregate_sharadar_delistings_to_monthly,
    assemble_sharadar_monthly_panel,
)


def test_normalize_sharadar_exchange():
    input_series = pd.Series(
        ["NYSE", "NASDAQ", "NYSEMKT", "OTC", None],
        index=[10, 20, 30, 40, 50], 
        name="exchange",
    )
    original_series = input_series.copy()

    result = normalize_sharadar_exchange(input_series)
    expected_series = pd.Series(
        ["NYSE", "NASDAQ", "AMEX", "OTC", None], 
        index=[10, 20, 30, 40, 50],
        name="exchange",
    )

    pd.testing.assert_series_equal(result, expected_series)
    pd.testing.assert_series_equal(input_series, original_series)


def test_aggregate_basic_and_leap_year_with_immutability():
    input_df = pd.DataFrame(
        {
            "security_id": [1, 1, 1, 1],
            "date": ["2020-01-30", "2020-01-31", "2020-02-27", "2020-02-28"],
            "close": [99.0, 100.0, 104.0, 105.0],
            "closeadj": [98.0, 99.0, 103.0, 104.0]
        }
    )
    
    original_df = input_df.copy()

    expected_df = pd.DataFrame(
        {
            "security_id": [1, 1],
            "date": pd.to_datetime(["2020-01-31", "2020-02-29"]),
            "price": [100.0, 105.0],
            "adjusted_price": [99.0, 104.0]
        }
    )
    result_df = aggregate_sharadar_prices_to_monthly(input_df)
    pd.testing.assert_frame_equal(result_df, expected_df)
    pd.testing.assert_frame_equal(input_df, original_df)


def test_aggregate_multiple_securities():
    input_df = pd.DataFrame(
        {
            "security_id": [1, 1, 2, 2],
            "date": ["2020-01-30", "2020-01-31", "2020-01-30", "2020-01-31"],
            "close": [10.0, 11.0, 50.0, 55.0],
            "closeadj": [9.0, 10.0, 48.0, 53.0]
        }
    )

    result_df = aggregate_sharadar_prices_to_monthly(input_df)

    expected_df = pd.DataFrame(
        {
            "security_id": [1, 2],
            "date": pd.to_datetime(["2020-01-31", "2020-01-31"]),
            "price": [11.0, 55.0],
            "adjusted_price": [10.0, 53.0]
        }
    )

    pd.testing.assert_frame_equal(result_df, expected_df)


def test_aggregate_unsorted_input():
    input_df = pd.DataFrame(
        {
            "security_id": [1, 1, 1],
            "date": ["2020-01-31", "2020-01-29", "2020-01-30"],
            "close": [100.0, 95.0, 98.0],
            "closeadj": [99.0, 94.0, 97.0]
        }
    )

    expected_df = pd.DataFrame(
        {
            "security_id": [1],
            "date": pd.to_datetime(["2020-01-31"]),
            "price": [100.0],
            "adjusted_price": [99.0]
        }
    )

    result_df = aggregate_sharadar_prices_to_monthly(input_df)
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_aggregate_nan_on_final_day():
    input_df = pd.DataFrame(
        {
            "security_id": [1, 1],
            "date": ["2020-01-30", "2020-01-31"],
            "close": [100.0, 101.0],
            "closeadj": [99.0, np.nan]
        }
    )

    expected_df = pd.DataFrame(
        {
            "security_id": [1],
            "date": pd.to_datetime(["2020-01-31"]),
            "price": [101.0],
            "adjusted_price": [np.nan]
        }
    )

    result_df = aggregate_sharadar_prices_to_monthly(input_df)
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_normal_consecutive_returns():
    input_df = pd.DataFrame(
        {
            "security_id": [1, 1, 1],
            "date": ["2020-01-31", "2020-02-29", "2020-03-31"],
            "price": [100.0, 110.0, 121.0],
            "adjusted_price": [100.0, 110.0, 121.0]
        }
    )
    
    expected_df = input_df.copy()
    expected_df["date"] = pd.to_datetime(expected_df["date"])
    expected_df["return_ex_delist"] = [np.nan, 0.10, 0.10]
    
    result_df = add_sharadar_monthly_returns(input_df)
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_missing_calendar_month():
    input_df = pd.DataFrame(
        {
            "security_id": [1, 1],
            "date": ["2020-01-31", "2020-03-31"],
            "price": [100.0, 110.0],
            "adjusted_price": [100.0, 110.0]
        }
    )
    
    expected_df = input_df.copy()
    expected_df["date"] = pd.to_datetime(expected_df["date"])
    expected_df["return_ex_delist"] = [np.nan, np.nan]
    
    result_df = add_sharadar_monthly_returns(input_df)
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_multiple_securities_no_leakage():
    input_df = pd.DataFrame(
        {
            "security_id": [1, 1, 2, 2],
            "date": ["2020-01-31", "2020-02-29", "2020-01-31", "2020-02-29"],
            "price": [100.0, 110.0, 50.0, 45.0],
            "adjusted_price": [100.0, 110.0, 50.0, 45.0]
        }
    )
    
    expected_df = input_df.copy()
    expected_df["date"] = pd.to_datetime(expected_df["date"])
    expected_df["return_ex_delist"] = [np.nan, 0.10, np.nan, -0.10]
    
    result_df = add_sharadar_monthly_returns(input_df)
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_missing_adjusted_price():
    input_df = pd.DataFrame(
        {
            "security_id": [1, 1, 1],
            "date": ["2020-01-31", "2020-02-29", "2020-03-31"],
            "price": [100.0, 110.0, 120.0],
            "adjusted_price": [100.0, np.nan, 120.0]
        }
    )
    
    expected_df = input_df.copy()
    expected_df["date"] = pd.to_datetime(expected_df["date"])
    expected_df["return_ex_delist"] = [np.nan, np.nan, np.nan]
    
    result_df = add_sharadar_monthly_returns(input_df)
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_unsorted_input_and_non_mutation():
    input_df = pd.DataFrame(
        {
            "security_id": [1, 1, 1],
            "date": ["2020-02-29", "2020-03-31", "2020-01-31"],
            "price": [110.0, 121.0, 100.0],
            "adjusted_price": [110.0, 121.0, 100.0]
        }, 
        index=[2, 0, 1]
    )
    
    original_df = input_df.copy(deep=True)
    
    expected_df = pd.DataFrame(
        {
            "security_id": [1, 1, 1],
            "date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]),
            "price": [100.0, 110.0, 121.0],
            "adjusted_price": [100.0, 110.0, 121.0],
            "return_ex_delist": [np.nan, 0.10, 0.10]
        }
    )
    
    result_df = add_sharadar_monthly_returns(input_df)
    
    pd.testing.assert_frame_equal(result_df, expected_df)
    pd.testing.assert_frame_equal(input_df, original_df)


def test_aggregate_market_cap_basic_and_unit_conversion():
    input_df = pd.DataFrame(
        {
            "security_id": [1, 1, 1, 1],
            "date": ["2020-01-30", "2020-01-31", "2020-02-27", "2020-02-28"],
            "marketcap": [2500.0, 2600.0, 2700.0, 2800.0]
        }
    )
    
    expected_df = pd.DataFrame(
        {
            "security_id": [1, 1],
            "date": pd.to_datetime(["2020-01-31", "2020-02-29"]),
            "market_cap": [2_600_000_000.0, 2_800_000_000.0]
        }
    )
    
    result_df = aggregate_sharadar_market_cap_to_monthly(input_df)
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_aggregate_market_cap_multiple_securities():
    input_df = pd.DataFrame(
        {
            "security_id": [1, 1, 2, 2],
            "date": ["2020-01-30", "2020-01-31", "2020-01-30", "2020-01-31"],
            "marketcap": [100.0, 110.0, 500.0, 550.0]
        }
    )
    
    expected_df = pd.DataFrame(
        {
            "security_id": [1, 2],
            "date": pd.to_datetime(["2020-01-31", "2020-01-31"]),
            "market_cap": [110_000_000.0, 550_000_000.0]
        }
    )
    
    result_df = aggregate_sharadar_market_cap_to_monthly(input_df)
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_aggregate_market_cap_unsorted_and_non_mutation():
    input_df = pd.DataFrame(
        {
            "security_id": [1, 1, 1],
            "date": ["2020-01-31", "2020-01-29", "2020-01-30"],
            "marketcap": [2600.0, 2400.0, 2500.0]
        }, 
        index=[5, 9, 2]
    )
    
    original_df = input_df.copy(deep=True)
    
    expected_df = pd.DataFrame(
        {
            "security_id": [1],
            "date": pd.to_datetime(["2020-01-31"]),
            "market_cap": [2_600_000_000.0]
        }
    )
    
    result_df = aggregate_sharadar_market_cap_to_monthly(input_df)
    
    pd.testing.assert_frame_equal(result_df, expected_df)
    pd.testing.assert_frame_equal(input_df, original_df)


def test_aggregate_market_cap_nan_on_final_day():
    input_df = pd.DataFrame(
        {
            "security_id": [1, 1],
            "date": ["2020-01-30", "2020-01-31"],
            "marketcap": [2500.0, np.nan]
        }
    )
    
    expected_df = pd.DataFrame(
        {
            "security_id": [1],
            "date": pd.to_datetime(["2020-01-31"]),
            "market_cap": [np.nan]
        }
    )
    
    result_df = aggregate_sharadar_market_cap_to_monthly(input_df)
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_normalize_sharadar_metadata_basic_and_immutability():
    input_df = pd.DataFrame(
        {
            "permaticker": ["100001", "100002", "100003"],
            "ticker": ["ABC", "XYZ", "DEF"],
            "exchange": ["NYSE", "NASDAQ", "NYSEMKT"]
        }
    )
    
    original_df = input_df.copy(deep=True)
    
    expected_df = pd.DataFrame(
        {
            "security_id": ["100001", "100002", "100003"],
            "ticker": ["ABC", "XYZ", "DEF"],
            "exchange": ["NYSE", "NASDAQ", "AMEX"],
            "security_type": ["COMMON", "COMMON", "COMMON"]
        }
    )
    
    result_df = normalize_sharadar_metadata(input_df)
    
    pd.testing.assert_frame_equal(result_df, expected_df)
    pd.testing.assert_frame_equal(input_df, original_df)


def test_normalize_sharadar_metadata_unsupported_exchanges():
    input_df = pd.DataFrame(
        {
            "permaticker": ["100004", "100005"],
            "ticker": ["GHI", "JKL"],
            "exchange": ["OTC", np.nan]
        }
    )
    
    expected_df = pd.DataFrame(
        {
            "security_id": ["100004", "100005"],
            "ticker": ["GHI", "JKL"],
            "exchange": ["OTC", np.nan],
            "security_type": ["COMMON", "COMMON"]
        }
    )
    
    result_df = normalize_sharadar_metadata(input_df)
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_aggregate_delistings_basic():
    input_df = pd.DataFrame(
        {
            "security_id": [1001, 1002],
            "date": ["2020-02-18", "2020-06-30"]
        }
    )
    
    expected_df = pd.DataFrame(
        {
            "security_id": [1001, 1002],
            "date": pd.to_datetime(["2020-02-29", "2020-06-30"]),
            "delisting_event": [True, True],
            "delisting_return": [np.nan, np.nan]
        }
    )
    
    result_df = aggregate_sharadar_delistings_to_monthly(input_df)
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_aggregate_delistings_duplicate_events_same_month():
    input_df = pd.DataFrame(
        {
            "security_id": [1, 1],
            "date": ["2020-05-15", "2020-05-20"]
        }
    )
    
    expected_df = pd.DataFrame(
        {
            "security_id": [1],
            "date": pd.to_datetime(["2020-05-31"]),
            "delisting_event": [True],
            "delisting_return": [np.nan]
        }
    )
    
    result_df = aggregate_sharadar_delistings_to_monthly(input_df)
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_aggregate_delistings_multiple_securities_unsorted_non_mutation():
    input_df = pd.DataFrame(
        {
            "security_id": [200, 100, 100],
            "date": ["2020-03-10", "2020-04-15", "2020-01-05"]
        }, 
        index=[5, 2, 8]
    )
    
    original_df = input_df.copy(deep=True)
    
    expected_df = pd.DataFrame(
        {
            "security_id": [100, 100, 200],
            "date": pd.to_datetime(["2020-01-31", "2020-04-30", "2020-03-31"]),
            "delisting_event": [True, True, True],
            "delisting_return": [np.nan, np.nan, np.nan]
        }
    )
    
    result_df = aggregate_sharadar_delistings_to_monthly(input_df)
    
    pd.testing.assert_frame_equal(result_df, expected_df)
    pd.testing.assert_frame_equal(input_df, original_df)


def test_assemble_sharadar_monthly_panel_basic():
    monthly_prices = pd.DataFrame(
        {
            "security_id": [1, 1],
            "date": pd.to_datetime(
                ["2020-01-31", "2020-02-29"]
            ),
            "price": [100.0, 110.0],
            "adjusted_price": [99.0, 109.0],
            "return_ex_delist": [np.nan, 0.10],
        }
    )

    monthly_market_caps = pd.DataFrame(
        {
            "security_id": [1, 1],
            "date": pd.to_datetime(
                ["2020-01-31", "2020-02-29"]
            ),
            "market_cap": [
                5_000_000_000.0,
                5_500_000_000.0,
            ],
        }
    )

    metadata = pd.DataFrame(
        {
            "security_id": [1],
            "ticker": ["ABC"],
            "exchange": ["NYSE"],
            "security_type": ["COMMON"],
        }
    )

    monthly_delistings = pd.DataFrame(
        {
            "security_id": [1],
            "date": pd.to_datetime(["2020-02-29"]),
            "delisting_event": [True],
            "delisting_return": [np.nan],
        }
    )

    original_prices = monthly_prices.copy(deep=True)
    original_market_caps = monthly_market_caps.copy(deep=True)
    original_metadata = metadata.copy(deep=True)
    original_delistings = monthly_delistings.copy(deep=True)

    expected_df = pd.DataFrame(
        {
            "security_id": [1, 1],
            "date": pd.to_datetime(
                ["2020-01-31", "2020-02-29"]
            ),
            "ticker": ["ABC", "ABC"],
            "exchange": ["NYSE", "NYSE"],
            "security_type": ["COMMON", "COMMON"],
            "return_ex_delist": [np.nan, 0.10],
            "delisting_event": [False, True],
            "delisting_return": [np.nan, np.nan],
            "price": [100.0, 110.0],
            "shares_outstanding": [np.nan, np.nan],
            "market_cap": [
                5_000_000_000.0,
                5_500_000_000.0,
            ],
        }
    )

    result_df = assemble_sharadar_monthly_panel(
        monthly_prices,
        monthly_market_caps,
        metadata,
        monthly_delistings,
    )

    pd.testing.assert_frame_equal(result_df, expected_df)
    pd.testing.assert_frame_equal(monthly_prices, original_prices)
    pd.testing.assert_frame_equal(monthly_market_caps, original_market_caps)
    pd.testing.assert_frame_equal(metadata, original_metadata)
    pd.testing.assert_frame_equal(monthly_delistings, original_delistings)


def test_assemble_preserves_delisting_month_without_price():
    monthly_prices = pd.DataFrame(
        {
            "security_id": [1],
            "date": pd.to_datetime(["2020-01-31"]),
            "price": [100.0],
            "adjusted_price": [99.0],
            "return_ex_delist": [np.nan],
        }
    )

    monthly_market_caps = pd.DataFrame(
        {
            "security_id": [1],
            "date": pd.to_datetime(["2020-01-31"]),
            "market_cap": [5_000_000_000.0],
        }
    )

    metadata = pd.DataFrame(
        {
            "security_id": [1],
            "ticker": ["ABC"],
            "exchange": ["NYSE"],
            "security_type": ["COMMON"],
        }
    )

    monthly_delistings = pd.DataFrame(
        {
            "security_id": [1],
            "date": pd.to_datetime(["2020-02-29"]),
            "delisting_event": [True],
            "delisting_return": [np.nan],
        }
    )

    result_df = assemble_sharadar_monthly_panel(
        monthly_prices,
        monthly_market_caps,
        metadata,
        monthly_delistings,
    )

    expected_df = pd.DataFrame(
        {
            "security_id": [1, 1],
            "date": pd.to_datetime(
                ["2020-01-31", "2020-02-29"]
            ),
            "ticker": ["ABC", "ABC"],
            "exchange": ["NYSE", "NYSE"],
            "security_type": ["COMMON", "COMMON"],
            "return_ex_delist": [np.nan, np.nan],
            "delisting_event": [False, True],
            "delisting_return": [np.nan, np.nan],
            "price": [100.0, np.nan],
            "shares_outstanding": [np.nan, np.nan],
            "market_cap": [5_000_000_000.0, np.nan],
        }
    )

    pd.testing.assert_frame_equal(result_df, expected_df)


def test_assemble_preserves_rows_with_missing_auxiliary_data():
    monthly_prices = pd.DataFrame(
        {
            "security_id": [1, 2],
            "date": pd.to_datetime(
                ["2020-01-31", "2020-01-31"]
            ),
            "price": [100.0, 50.0],
            "adjusted_price": [99.0, 49.0],
            "return_ex_delist": [np.nan, np.nan],
        }
    )

    monthly_market_caps = pd.DataFrame(
        {
            "security_id": [1],
            "date": pd.to_datetime(["2020-01-31"]),
            "market_cap": [5_000_000_000.0],
        }
    )

    metadata = pd.DataFrame(
        {
            "security_id": [1],
            "ticker": ["ABC"],
            "exchange": ["NYSE"],
            "security_type": ["COMMON"],
        }
    )

    monthly_delistings = pd.DataFrame(
        columns=[
            "security_id",
            "date",
            "delisting_event",
            "delisting_return",
        ]
    )

    result_df = assemble_sharadar_monthly_panel(
        monthly_prices,
        monthly_market_caps,
        metadata,
        monthly_delistings,
    )

    assert len(result_df) == 2

    security_2 = result_df.loc[
        result_df["security_id"] == 2
    ].iloc[0]

    assert pd.isna(security_2["market_cap"])
    assert pd.isna(security_2["ticker"])
    assert pd.isna(security_2["exchange"])
    assert pd.isna(security_2["security_type"])
    assert not security_2["delisting_event"]


def test_assemble_rejects_duplicate_market_cap_rows():
    monthly_prices = pd.DataFrame(
        {
            "security_id": [1],
            "date": pd.to_datetime(["2020-01-31"]),
            "price": [100.0],
            "adjusted_price": [99.0],
            "return_ex_delist": [np.nan],
        }
    )

    monthly_market_caps = pd.DataFrame(
        {
            "security_id": [1, 1],
            "date": pd.to_datetime(
                ["2020-01-31", "2020-01-31"]
            ),
            "market_cap": [
                5_000_000_000.0,
                5_100_000_000.0,
            ],
        }
    )

    metadata = pd.DataFrame(
        {
            "security_id": [1],
            "ticker": ["ABC"],
            "exchange": ["NYSE"],
            "security_type": ["COMMON"],
        }
    )

    monthly_delistings = pd.DataFrame(
        columns=[
            "security_id",
            "date",
            "delisting_event",
            "delisting_return",
        ]
    )

    with pytest.raises(pd.errors.MergeError):
        assemble_sharadar_monthly_panel(
            monthly_prices,
            monthly_market_caps,
            metadata,
            monthly_delistings,
        )