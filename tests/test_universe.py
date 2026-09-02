import pandas as pd
import numpy as np
import pytest

from equity_factor_research.data.universe import (
    add_lagged_market_cap,
    add_base_eligibility,
    add_universe_membership,
)


def test_add_lagged_market_cap():
    panel = pd.DataFrame(
        {
            "security_id": [1, 2, 1, 2, 1, 2],
            "date": pd.to_datetime(
                [
                    "2020-03-31",
                    "2020-02-29",
                    "2020-01-31",
                    "2020-03-31",
                    "2020-02-29",
                    "2020-01-31",
                ]
            ),
            "market_cap": [
                90.0,
                140.0,
                100.0,
                150.0,
                120.0,
                80.0,
            ],
        }
    )
    
    original_panel = panel.copy()

    result = add_lagged_market_cap(panel)

    expected_lagged = pd.Series(
        [
            120.0,   # Row 0: ID 1 Mar gets ID 1 Feb (120.0)
            80.0,    # Row 1: ID 2 Feb gets ID 2 Jan (80.0)
            np.nan,  # Row 2: ID 1 Jan gets NaN (first observation)
            140.0,   # Row 3: ID 2 Mar gets ID 2 Feb (140.0)
            100.0,   # Row 4: ID 1 Feb gets ID 1 Jan (100.0)
            np.nan,  # Row 5: ID 2 Jan gets NaN (first observation)
        ],
        name="lagged_market_cap"
    )

    pd.testing.assert_series_equal(result["lagged_market_cap"], expected_lagged)
    pd.testing.assert_frame_equal(panel, original_panel)
    pd.testing.assert_series_equal(result.index.to_series(), original_panel.index.to_series())


def test_add_lagged_market_cap_requires_previous_calendar_month():
    panel = pd.DataFrame(
        {
            "security_id": [1, 1, 1],
            "date": pd.to_datetime(
                [
                    "2020-01-31",
                    "2020-03-31",
                    "2020-04-30",
                ]
            ),
            "market_cap": [
                100.0,
                130.0,
                140.0,
            ],
        }
    )

    result = add_lagged_market_cap(panel)

    expected = pd.Series(
        [
            np.nan,
            np.nan,
            130.0,
        ],
        name="lagged_market_cap",
    )

    pd.testing.assert_series_equal(result["lagged_market_cap"], expected)


def test_add_base_eligibility():
    panel = pd.DataFrame(
        {
            "exchange": [
                "NYSE",
                "AMEX",
                "NASDAQ",
                "OTC",
                "NYSE",
            ],
            "security_type": [
                "COMMON",
                "COMMON",
                "COMMON",
                "COMMON",
                "ETF",
            ],
        }
    )

    result = add_base_eligibility(panel)
    expected = pd.Series(
        [
            True,
            True,
            True,
            False,
            False,
        ],
        name="base_eligible",
    )

    pd.testing.assert_series_equal(result["base_eligible"], expected)
    assert "base_eligible" not in panel.columns


def test_add_universe_membership():
    panel = pd.DataFrame(
        {
            "security_id": [10, 20, 30, 40, 50],
            "date": pd.to_datetime(
                [
                    "2020-03-31",
                    "2020-03-31",
                    "2020-03-31",
                    "2020-03-31",
                    "2020-03-31",
                ]
            ),
            "base_eligible": [
                True,
                True,
                True,
                True,
                True,
            ],
            "lagged_market_cap": [
                500.0,
                400.0,
                300.0,
                300.0,
                200.0,
            ],
        }
    )
    original_panel = panel.copy()

    result = add_universe_membership(panel, n=3)

    expected = pd.Series(
        [True, True, True, False, False,],
        name="in_universe",
    )

    pd.testing.assert_series_equal(result["in_universe"], expected)
    pd.testing.assert_frame_equal(panel, original_panel)


def test_add_universe_membership_excludes_unrankable_rows():
    panel = pd.DataFrame(
        {
            "security_id": [10, 20, 30, 40],
            "date": pd.to_datetime(
                [
                    "2020-03-31",
                    "2020-03-31",
                    "2020-03-31",
                    "2020-03-31",
                ]
            ),
            "base_eligible": [
                True,
                False,
                True,
                True,
            ],
            "lagged_market_cap": [
                500.0,
                900.0,
                np.nan,
                300.0,
            ],
        }
    )

    result = add_universe_membership(panel, n=3)

    expected = pd.Series(
        [True, False, False, True],
        name="in_universe",
    )

    pd.testing.assert_series_equal(result["in_universe"], expected)


def test_add_universe_membership_selects_separately_by_month():
    panel = pd.DataFrame(
        {
            "security_id": [10, 20, 30, 10, 20, 30],
            "date": pd.to_datetime(
                [
                    "2020-03-31",
                    "2020-03-31",
                    "2020-03-31",
                    "2020-04-30",
                    "2020-04-30",
                    "2020-04-30",
                ]
            ),
            "base_eligible": [
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            "lagged_market_cap": [
                500.0,
                400.0,
                300.0,
                200.0,
                600.0,
                700.0,
            ],
        }
    )

    result = add_universe_membership(panel, n=2)

    expected = pd.Series(
        [
            True,   # March: 500
            True,   # March: 400
            False,  # March: 300
            False,  # April: 200
            True,   # April: 600
            True,   # April: 700
        ],
        name="in_universe",
    )

    pd.testing.assert_series_equal(result["in_universe"], expected)


def test_add_universe_membership_rejects_invalid_n():
    panel = pd.DataFrame(
        {
            "security_id": [10],
            "date": pd.to_datetime(["2020-03-31"]),
            "base_eligible": [True],
            "lagged_market_cap": [500.0],
        }
    )

    with pytest.raises(ValueError):
        add_universe_membership(panel, n=0)

    with pytest.raises(ValueError):
        add_universe_membership(panel, n=-1)

    with pytest.raises(ValueError):
        add_universe_membership(panel, n=2.5)


def test_universe_construction_pipeline():
    panel = pd.DataFrame(
        {
            "security_id": [1, 2, 3, 1, 2, 3],
            "date": pd.to_datetime(
                [
                    "2020-01-31",
                    "2020-01-31",
                    "2020-01-31",
                    "2020-02-29",
                    "2020-02-29",
                    "2020-02-29",
                ]
            ),
            "exchange": [
                "NYSE",
                "NASDAQ",
                "OTC",
                "NYSE",
                "NASDAQ",
                "OTC",
            ],
            "security_type": [
                "COMMON",
                "COMMON",
                "COMMON",
                "COMMON",
                "COMMON",
                "COMMON",
            ],
            "market_cap": [
                500.0,
                400.0,
                900.0,
                550.0,
                450.0,
                950.0,
            ],
        }
    )

    result = add_lagged_market_cap(panel)
    result = add_base_eligibility(result)
    result = add_universe_membership(result, n=1)

    expected_lagged_market_cap = pd.Series(
        [np.nan, np.nan, np.nan, 500.0, 400.0, 900.0],
        name="lagged_market_cap",
    )

    expected_base_eligible = pd.Series(
        [True, True, False, True, True, False],
        name="base_eligible",
    )

    expected_in_universe = pd.Series(
        [False, False, False, True, False, False],
        name="in_universe",
    )

    pd.testing.assert_series_equal(
        result["lagged_market_cap"], 
        expected_lagged_market_cap,
    )

    pd.testing.assert_series_equal(
        result["base_eligible"],
        expected_base_eligible,
    )

    pd.testing.assert_series_equal(
        result["in_universe"],
        expected_in_universe,
    )