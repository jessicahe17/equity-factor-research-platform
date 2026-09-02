import pandas as pd
import numpy as np

from equity_factor_research.data.quality import (
    find_invalid_prices,
    find_invalid_shares,
    find_invalid_total_returns,
    find_market_cap_inconsistencies,
    find_zero_prices,
    find_zero_shares,
    find_delisting_inconsistencies,
    find_missing_delisting_returns,
    find_missing_core_fields,
    assess_data_quality,
)


def test_find_invalid_prices():
    panel = pd.DataFrame(
        {
            "price": [25.0, 0.0, -1.0, np.nan]
        }
    )
    result = find_invalid_prices(panel)
    expected = pd.Series([False, False, True, False], name = "price")
    pd.testing.assert_series_equal(result, expected)


def test_find_invalid_shares():
    panel = pd.DataFrame(
        {
            "shares_outstanding": [
                1_000_000, 
                0, 
                -500, 
                np.nan
            ]
        }
    )
    result = find_invalid_shares(panel)
    expected = pd.Series([False, False, True, False], name = "shares_outstanding")
    pd.testing.assert_series_equal(result, expected)


def test_find_invalid_total_returns():
    panel = pd.DataFrame(
            {
                "total_return": [
                    -1.00, 
                    -0.99, 
                    5.00,
                    -1.01, 
                    np.nan
                ]
            }
        )
    result = find_invalid_total_returns(panel)
    expected = pd.Series([False, False, False, True, False], name = "total_return")
    pd.testing.assert_series_equal(result, expected)
    

def test_find_market_cap_inconsistencies():
    panel = pd.DataFrame(
        {
            "price": [
                25.0,
                10.0,
                50.0,
                np.nan,
            ],
            "shares_outstanding": [
                1_000_000.0,
                2_000_000.0,
                500_000.0,
                1_000_000.0,
            ],
            "market_cap": [
                25_000_000.0,         # correct
                21_000_000.0,         # incorrect
                25_000_000.00000001,  # tiny numerical difference
                np.nan,               # missing, not inconsistent
            ],
        }
    )

    result = find_market_cap_inconsistencies(panel)
    expected = pd.Series([False, True, False, False])
    pd.testing.assert_series_equal(result, expected)


def test_find_zero_prices():
    panel = pd.DataFrame(
        {
            "price": [25.0, 0.0, -1.0, np.nan]
        }
    )

    result = find_zero_prices(panel)
    expected = pd.Series(
        [False, True, False, False],
        name = "price"
    )
    pd.testing.assert_series_equal(result, expected)


def test_find_zero_shares():
    panel = pd.DataFrame(
        {
            "shares_outstanding": [
                1_000_000, 
                0, 
                -500, 
                np.nan
            ]
        }
    )

    result = find_zero_shares(panel)
    expected = pd.Series(
        [False, True, False, False],
        name = "shares_outstanding"
    )
    pd.testing.assert_series_equal(result, expected)


def test_find_delisting_inconsistencies():
    panel = pd.DataFrame(
        {
            "delisting_event": [
                False,
                True,
                True,
                False,
            ],
            "delisting_return": [
                -0.80,
                -0.80,
                np.nan,
                np.nan,
            ],
        }
    )

    result = find_delisting_inconsistencies(panel)

    expected = pd.Series(
        [
            True,   # no event, but delisting return exists
            False,  # event + known delisting return
            False,  # event + missing return: incomplete, not inconsistent
            False,  # no event + no delisting return
        ]
    )

    pd.testing.assert_series_equal(result, expected)


def test_find_missing_delisting_returns():
    panel = pd.DataFrame(
        {
            "delisting_event": [
                False,
                True,
                True,
                False,
            ],
            "delisting_return": [
                np.nan,
                -0.80,
                np.nan,
                -0.50,
            ],
        }
    )

    result = find_missing_delisting_returns(panel)
    expected = pd.Series(
        [
            False,  # no delisting, missing is expected
            False,  # delisting with known return
            True,   # delisting with missing return
            False,  # inconsistent case, handled elsewhere
        ]
    )
    pd.testing.assert_series_equal(result, expected)


def test_find_missing_core_fields():
    panel = pd.DataFrame(
        {
            "return_ex_delist": [
                0.05,
                np.nan,
                0.10,
            ],
            "total_return": [
                0.05,
                np.nan,
                0.10,
            ],
            "price": [
                25.0,
                20.0,
                np.nan,
            ],
            "shares_outstanding": [
                1_000_000.0,
                2_000_000.0,
                3_000_000.0,
            ],
            "market_cap": [
                25_000_000.0,
                40_000_000.0,
                np.nan,
            ],
        }
    )

    result = find_missing_core_fields(panel)
    expected = pd.DataFrame(
        {
            "return_ex_delist": [
                False,
                True,
                False,
            ],
            "total_return": [
                False,
                True,
                False,
            ],
            "price": [
                False,
                False,
                True,
            ],
            "shares_outstanding": [
                False,
                False,
                False,
            ],
            "market_cap": [
                False,
                False,
                True,
            ],
        }
    )
    pd.testing.assert_frame_equal(result, expected)


def test_assess_data_quality():
    panel = pd.DataFrame(
        {
            "price": [
                25.0,
                -5.0,
                0.0,
                np.nan,
            ],
            "shares_outstanding": [
                1_000_000.0,
                2_000_000.0,
                0.0,
                500_000.0,
            ],
            "market_cap": [
                25_000_000.0,
                -10_000_000.0,
                1_000.0,
                np.nan,
            ],
            "return_ex_delist": [
                0.05,
                -0.20,
                0.10,
                np.nan,
            ],
            "total_return": [
                0.05,
                -1.20,
                0.10,
                np.nan,
            ],
            "delisting_event": [
                False,
                True,
                False,
                True,
            ],
            "delisting_return": [
                np.nan,
                -0.80,
                -0.50,
                np.nan,
            ],
        }
    )

    result = assess_data_quality(panel)
    expected = pd.Series(
        {
            "invalid_prices": 1,
            "invalid_shares": 0,
            "invalid_total_returns": 1,
            "market_cap_inconsistencies": 1,
            "zero_prices": 1,
            "zero_shares": 1,
            "delisting_inconsistencies": 1,
            "missing_delisting_returns": 1,
            "missing_return_ex_delist": 1,
            "missing_total_return":1,
            "missing_price": 1,
            "missing_shares_outstanding": 0,
            "missing_market_cap": 1,
        },
        name="count",
    )
    pd.testing.assert_series_equal(result, expected)