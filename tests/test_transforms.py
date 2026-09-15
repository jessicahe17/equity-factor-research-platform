import pandas as pd
import pytest
import numpy as np

from equity_factor_research.data.transforms import (
    construct_total_return, 
    construct_market_cap,
    construct_equity_panel,
)


def test_construct_total_return():
    return_ex_delist = pd.Series([
        0.05,
        -0.30,
        0.10,
    ])

    delisting_return = pd.Series([
        float("nan"),
        -0.80,
        -0.50,
    ])

    delisting_event = pd.Series([
        False,
        True,
        True,
    ])

    result = construct_total_return(
        return_ex_delist,
        delisting_return,
        delisting_event,
    )

    expected = pd.Series([
        0.05,
        -0.86,
        -0.45,
    ])

    pd.testing.assert_series_equal(result, expected)


def test_construct_total_return_with_missing_returns():
    return_ex_delist = pd.Series([
        -0.20,
        float("nan"),
        float("nan"),
    ])

    delisting_return = pd.Series([
        float("nan"),
        -0.80,
        float("nan"),
    ])

    delisting_event = pd.Series([
        True,
        True,
        False,
    ])

    result = construct_total_return(
        return_ex_delist,
        delisting_return,
        delisting_event,
    )

    expected = pd.Series([
        float("nan"),
        float("nan"),
        float("nan"),
    ])

    pd.testing.assert_series_equal(result, expected)


def test_construct_total_return_rejects_missing_delisting_event():
    return_ex_delist = pd.Series([0.05])
    delisting_return = pd.Series([float("nan")])
    delisting_event = pd.Series([float("nan")])

    with pytest.raises(ValueError, match="delisting_event must not contain missing values"):
        construct_total_return(
            return_ex_delist,
            delisting_return,
            delisting_event,
        )


def test_construct_market_cap():
    price = pd.Series([
        25.0,
        100.0,
        7.5,
    ])

    shares_outstanding = pd.Series([
        1_000_000.0,
        500_000.0,
        2_000_000.0,
    ])

    result = construct_market_cap(
        price,
        shares_outstanding,
    )

    expected = pd.Series([
        25_000_000.0,
        50_000_000.0,
        15_000_000.0,
    ])

    pd.testing.assert_series_equal(result, expected)


def test_construct_market_cap_with_missing_inputs():
    price = pd.Series([
        float("nan"),
        25.0,
    ])

    shares_outstanding = pd.Series([
        1_000_000.0,
        float("nan"),
    ])

    result = construct_market_cap(
        price,
        shares_outstanding,
    )

    expected = pd.Series([
        float("nan"),
        float("nan"),
    ])

    pd.testing.assert_series_equal(result, expected)


def test_construct_equity_panel():
    panel = pd.DataFrame(
        {
            "return_ex_delist": [0.05, -0.30],
            "delisting_event": [False, True],
            "delisting_return": [float("nan"), -0.80],
            "price": [25, 10],
            "shares_outstanding": [1_000_000, 2_000_000],
        }
    )

    constructed_panel = construct_equity_panel(panel)

    np.testing.assert_allclose(
        constructed_panel["total_return"],
        [0.05, -0.86],
    )

    np.testing.assert_allclose(
        constructed_panel["market_cap"],
        [25_000_000, 20_000_000],
    )

    assert "total_return" not in panel.columns
    assert "market_cap" not in panel.columns


def test_supplied_market_cap_preserved():
    panel = pd.DataFrame(
        {
            "return_ex_delist": [0.05],
            "delisting_event": [False],
            "delisting_return": [float("nan")],
            "price": [100.0],
            "shares_outstanding": [np.nan],
            "market_cap": [5_000_000_000.0],
        }
    )

    original_panel = panel.copy(deep=True)
    
    constructed_panel = construct_equity_panel(panel)
    np.testing.assert_allclose(
        constructed_panel["market_cap"],
        [5_000_000_000.0],
    )
    pd.testing.assert_frame_equal(panel, original_panel)


def test_construct_total_return_handles_nullable_delisting_return():
    return_ex_delist = pd.Series(
        [0.10, -0.40, 0.20, 0.05],
        dtype="float64",
    )

    delisting_return = pd.Series(
        [pd.NA, -0.30, -0.30, -0.30],
        dtype="Float64",
    )

    delisting_event = pd.Series(
        [False, True, True, True],
        dtype=bool,
    )

    result = construct_total_return(
        return_ex_delist,
        delisting_return,
        delisting_event,
    )

    expected = pd.Series(
        [
            0.10,
            -0.58,
            -0.16,
            -0.265,
        ],
        dtype="float64",
    )

    pd.testing.assert_series_equal(
        result,
        expected,
    )


def test_construct_total_return_handles_large_nullable_delisting_return():
    n = 4_000

    return_ex_delist = pd.Series(
        np.full(n, -0.40),
        dtype="float64",
    )

    delisting_return = pd.Series(
        pd.array(
            np.full(n, -0.30),
            dtype="Float64",
        )
    )

    delisting_event = pd.Series(
        np.ones(n, dtype=bool)
    )

    result = construct_total_return(
        return_ex_delist,
        delisting_return,
        delisting_event,
    )

    expected = pd.Series(
        np.full(n, -0.58),
        dtype="float64",
    )

    pd.testing.assert_series_equal(
        result,
        expected,
    )