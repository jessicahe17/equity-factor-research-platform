import pandas as pd
import pytest

from equity_factor_research.data.schema import validate_equity_panel


def make_valid_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "security_id": [10001, 10002, 10001],
            "date": pd.to_datetime(
                ["2020-01-31", "2020-01-31", "2020-02-29"]
            ),
            "ticker": ["AAA", "BBB", "AAA"],
            "exchange": ["NYSE", "NASDAQ", "NYSE"],
            "security_type": ["common_stock", "common_stock", "common_stock"],
            "return_ex_delist": [0.05, -0.02, 0.03],
            "delisting_return": [None, None, None],
            "total_return": [0.05, -0.02, 0.03],
            "price": [50.0, 25.0, 51.5],
            "shares_outstanding": [100_000_000, 80_000_000, 100_000_000],
            "market_cap": [
                5_000_000_000,
                2_000_000_000,
                5_150_000_000,
            ],
        }
    )


def test_valid_panel():
    panel = make_valid_panel()
    validate_equity_panel(panel)


def test_non_dataframe():
    with pytest.raises(TypeError):
        validate_equity_panel([1, 2, 3])


def test_missing_required_column():
    panel = make_valid_panel()
    panel = panel.drop(columns="market_cap")
    with pytest.raises(ValueError, match="market_cap"):
        validate_equity_panel(panel)


def test_missing_security_id():
    panel = make_valid_panel()
    panel.loc[0, "security_id"] = None
    with pytest.raises(ValueError):
        validate_equity_panel(panel)


def test_missing_date():
    panel = make_valid_panel()
    panel.loc[0, "date"] = None
    with pytest.raises(ValueError):
        validate_equity_panel(panel)


def test_duplicate_composite_key():
    panel = make_valid_panel()
    panel.loc[2, "date"] = pd.to_datetime("2020-01-31")
    with pytest.raises(ValueError):
        validate_equity_panel(panel)


def test_wrong_date_dtype():
    panel = make_valid_panel()
    panel["date"] = panel["date"].dt.strftime("%Y-%m-%d")
    with pytest.raises(ValueError):
        validate_equity_panel(panel)


def test_extra_column_allowed():
    panel = make_valid_panel()
    panel["some_extra_field"] = 1
    validate_equity_panel(panel)