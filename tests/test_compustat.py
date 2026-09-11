import pandas as pd
import pytest
import numpy as np

from equity_factor_research.data.providers.compustat import (
    normalize_compustat_secm,
    normalize_compustat_sec_history,
    add_compustat_historical_exchange,
    normalize_compustat_security,
    add_compustat_inactivation_metadata,
    handle_compustat_terminal_stubs,
    add_compustat_delisting_adjustment,
    assemble_compustat_monthly_panel,
)
from equity_factor_research.data.transforms import (
    construct_equity_panel,
)


def test_normalize_compustat_secm():
    secm = pd.DataFrame(
        {
            "gvkey": ["001690", "012141"],
            "iid": ["01", "01"],
            "datadate": ["2025-01-31", "2025-01-31"],
            "tic": ["AAPL", "MSFT"],
            "tpci": ["0", "0"],
            "trt1m": [-5.7583, -1.5346],
            "prccm": [236.00, 415.06],
            "cshom": [15_037_874_000, 7_432_000_000],
        }
    )

    result = normalize_compustat_secm(secm)

    assert result["security_id"].tolist() == [
        "001690_01",
        "012141_01",
    ]
    assert result["ticker"].tolist() == [
        "AAPL",
        "MSFT",
    ]
    assert result["security_type"].tolist() == [
        "COMMON",
        "COMMON",
    ]

    pd.testing.assert_series_equal(
        result["return_ex_delist"],
        pd.Series(
            [-0.057583, -0.015346],
            name="return_ex_delist",
        ),
    )

    assert pd.api.types.is_datetime64_any_dtype(
        result["date"]
    )
    assert result["price"].tolist() == [
        236.00,
        415.06,
    ]
    assert result["shares_outstanding"].tolist() == [
        15_037_874_000,
        7_432_000_000,
    ]


def test_normalize_compustat_secm_preserves_non_common_security_type():
    secm = pd.DataFrame(
        {
            "gvkey": ["001690"],
            "iid": ["02"],
            "datadate": ["2025-01-31"],
            "tic": ["TEST"],
            "tpci": ["1"],
            "trt1m": [2.5],
            "prccm": [50.0],
            "cshom": [1_000_000],
        }
    )

    result = normalize_compustat_secm(secm)
    assert result.loc[0, "security_type"] == "1"


def test_normalize_compustat_secm_preserves_missing_numeric_values():
    secm = pd.DataFrame(
        {
            "gvkey": ["001690"],
            "iid": ["01"],
            "datadate": ["2025-01-31"],
            "tic": ["AAPL"],
            "tpci": ["0"],
            "trt1m": [float("nan")],
            "prccm": [float("nan")],
            "cshom": [float("nan")],
        }
    )

    result = normalize_compustat_secm(secm)

    assert pd.isna(result.loc[0, "return_ex_delist"])
    assert pd.isna(result.loc[0, "price"])
    assert pd.isna(result.loc[0, "shares_outstanding"])


def test_normalize_compustat_secm_raises_for_missing_required_column():
    secm = pd.DataFrame(
        {
            "gvkey": ["001690"],
            "iid": ["01"],
            "datadate": ["2025-01-31"],
            "tic": ["AAPL"],
            "tpci": ["0"],
            # trt1m deliberately missing
            "prccm": [236.0],
            "cshom": [15_037_874_000],
        }
    )

    with pytest.raises(
        ValueError,
        match="missing required",
    ):
        normalize_compustat_secm(secm)


def test_normalize_compustat_secm_does_not_mutate_input():
    secm = pd.DataFrame(
        {
            "gvkey": ["001690"],
            "iid": ["01"],
            "datadate": ["2025-01-31"],
            "tic": ["AAPL"],
            "tpci": ["0"],
            "trt1m": [-5.7583],
            "prccm": [236.0],
            "cshom": [15_037_874_000],
        }
    )

    original = secm.copy(deep=True)
    normalize_compustat_secm(secm)

    pd.testing.assert_frame_equal(secm, original)


def test_normalize_compustat_secm_handles_numeric_common_stock_code():
    secm = pd.DataFrame(
        {
            "gvkey": ["001690"],
            "iid": ["01"],
            "datadate": ["2025-01-31"],
            "tic": ["AAPL"],
            "tpci": [0],
            "trt1m": [-5.7583],
            "prccm": [236.0],
            "cshom": [15_037_874_000],
        }
    )

    result = normalize_compustat_secm(secm)

    assert result.loc[0, "security_type"] == "COMMON"


def test_normalize_compustat_secm_handles_float_common_stock_code():
    secm = pd.DataFrame(
        {
            "gvkey": ["001690", "012141"],
            "iid": ["01", "01"],
            "datadate": [
                "2025-01-31",
                "2025-01-31",
            ],
            "tic": ["AAPL", "MSFT"],
            "tpci": [0.0, None],
            "trt1m": [-5.7583, -1.5346],
            "prccm": [236.0, 415.06],
            "cshom": [
                15_037_874_000,
                7_432_000_000,
            ],
        }
    )

    result = normalize_compustat_secm(secm)

    assert result.loc[0, "security_type"] == "COMMON"
    assert pd.isna(result.loc[1, "security_type"])


def test_normalize_compustat_sec_history():
    sec_history = pd.DataFrame(
        {
            "gvkey": ["001690", "012141"],
            "iid": ["01", "01"],
            "effdate": [
                "1998-04-01",
                "1998-04-01",
            ],
            "thrudate": [
                "2020-12-31",
                None,
            ],
            "item": [
                "EXCHG",
                "EXCHG",
            ],
            "itemvalue": [
                "14",
                "14",
            ],
        }
    )

    result = normalize_compustat_sec_history(
        sec_history
    )

    assert result["security_id"].tolist() == [
        "001690_01",
        "012141_01",
    ]

    assert pd.api.types.is_datetime64_any_dtype(result["effdate"])
    assert pd.api.types.is_datetime64_any_dtype(result["thrudate"])

    assert result.loc[0, "item"] == "EXCHG"
    assert result.loc[0, "itemvalue"] == "14"

    assert pd.isna(result.loc[1, "thrudate"])


def test_normalize_compustat_sec_history_raises_for_missing_required_column():
    sec_history = pd.DataFrame(
        {
            "gvkey": ["001690"],
            "iid": ["01"],
            "effdate": ["1998-04-01"],
            "thrudate": [None],
            "item": ["EXCHG"],
            # itemvalue deliberately missing
        }
    )

    with pytest.raises(ValueError, match="missing required"):
        normalize_compustat_sec_history(sec_history)


def test_normalize_compustat_sec_history_does_not_mutate_input():
    sec_history = pd.DataFrame(
        {
            "gvkey": ["001690"],
            "iid": ["01"],
            "effdate": ["1998-04-01"],
            "thrudate": [None],
            "item": ["EXCHG"],
            "itemvalue": ["14"],
        }
    )

    original = sec_history.copy(deep=True)

    normalize_compustat_sec_history(sec_history)

    pd.testing.assert_frame_equal(sec_history, original)


def test_add_compustat_historical_exchange_handles_exchange_transition():
    secm = pd.DataFrame(
        {
            "security_id": ["123456_01", "123456_01"],
            "date": pd.to_datetime(
                [
                    "2004-06-30",
                    "2004-07-31",
                ]
            ),
        }
    )

    sec_history = pd.DataFrame(
        {
            "security_id": ["123456_01", "123456_01"],
            "effdate": pd.to_datetime(
                [
                    "2000-01-01",
                    "2004-07-01",
                ]
            ),
            "thrudate": pd.to_datetime(
                [
                    "2004-06-30",
                    None,
                ]
            ),
            "item": ["EXCHG", "EXCHG"],
            "itemvalue": ["11", "14"],
        }
    )

    result = add_compustat_historical_exchange(secm, sec_history)

    assert result["exchange"].tolist() == ["NYSE", "NASDAQ"]


def test_add_compustat_historical_exchange_maps_amex():
    secm = pd.DataFrame(
        {
            "security_id": ["123456_01"],
            "date": pd.to_datetime(
                ["2005-01-31"]
            ),
        }
    )

    sec_history = pd.DataFrame(
        {
            "security_id": ["123456_01"],
            "effdate": pd.to_datetime(
                ["2000-01-01"]
            ),
            "thrudate": pd.to_datetime(
                [None]
            ),
            "item": ["EXCHG"],
            "itemvalue": ["12"],
        }
    )

    result = add_compustat_historical_exchange(secm, sec_history)

    assert result.loc[0, "exchange"] == "AMEX"


def test_add_compustat_historical_exchange_respects_expired_interval():
    secm = pd.DataFrame(
        {
            "security_id": ["123456_01"],
            "date": pd.to_datetime(
                ["2004-08-31"]
            ),
        }
    )

    sec_history = pd.DataFrame(
        {
            "security_id": ["123456_01"],
            "effdate": pd.to_datetime(
                ["2000-01-01"]
            ),
            "thrudate": pd.to_datetime(
                ["2004-06-30"]
            ),
            "item": ["EXCHG"],
            "itemvalue": ["11"],
        }
    )

    result = add_compustat_historical_exchange(secm, sec_history)

    assert pd.isna(result.loc[0, "exchange"])


def test_normalize_compustat_security():
    security = pd.DataFrame(
        {
            "gvkey": [
                "149738",
                "011818",
                "007267",
            ],
            "iid": [
                "01",
                "01",
                "01",
            ],
            "dldtei": [
                "2009-12-10",
                "2008-06-02",
                "2009-01-02",
            ],
            "dlrsni": [
                "20",
                "01",
                "01",
            ],
        }
    )

    result = normalize_compustat_security(security)

    assert result["security_id"].tolist() == [
        "149738_01",
        "011818_01",
        "007267_01",
    ]

    assert pd.api.types.is_datetime64_any_dtype(
        result["dldtei"]
    )

    assert result["inactivation_code"].tolist() == [
        "20",
        "01",
        "01",
    ]


def test_normalize_compustat_security_handles_numeric_inactivation_codes():
    security = pd.DataFrame(
        {
            "gvkey": [
                "000001",
                "000002",
                "000003",
            ],
            "iid": [
                "01",
                "01",
                "01",
            ],
            "dldtei": [
                "2008-01-31",
                "2008-02-29",
                None,
            ],
            "dlrsni": [
                2.0,
                3.0,
                None,
            ],
        }
    )

    result = normalize_compustat_security(security)

    assert result.loc[0, "inactivation_code"] == "02"
    assert result.loc[1, "inactivation_code"] == "03"
    assert pd.isna(result.loc[2, "inactivation_code"])
    assert pd.isna(result.loc[2, "dldtei"])


def test_normalize_compustat_security_does_not_mutate_input():
    security = pd.DataFrame(
        {
            "gvkey": ["149738"],
            "iid": ["01"],
            "dldtei": ["2009-12-10"],
            "dlrsni": ["20"],
        }
    )

    original = security.copy(deep=True)
    normalize_compustat_security(security)

    pd.testing.assert_frame_equal(security, original)


def test_normalize_compustat_security_raises_for_missing_required_column():
    security = pd.DataFrame(
        {
            "gvkey": ["149738"],
            "iid": ["01"],
            "dldtei": ["2009-12-10"],
            # dlrsni deliberately missing
        }
    )

    with pytest.raises(ValueError, match="missing required"):
        normalize_compustat_security(security)


def test_add_compustat_inactivation_metadata_matches_by_security_id():
    secm = pd.DataFrame(
        {
            "security_id": [
                "149738_01",
                "149738_01",
                "149738_06",
            ],
            "date": pd.to_datetime(
                [
                    "2009-11-30",
                    "2009-12-31",
                    "2010-01-31",
                ]
            ),
        }
    )

    security = pd.DataFrame(
        {
            "security_id": [
                "149738_01",
                "149738_06",
            ],
            "dldtei": pd.to_datetime(
                [
                    "2009-12-10",
                    None,
                ]
            ),
            "inactivation_code": [
                "20",
                pd.NA,
            ],
        }
    )

    result = add_compustat_inactivation_metadata(secm, security)

    assert result.loc[0, "inactivation_code"] == "20"
    assert result.loc[1, "inactivation_code"] == "20"
    assert pd.isna(result.loc[2, "inactivation_code"])

    assert result.loc[
        1, "inactivation_month"
    ] == pd.Timestamp("2009-12-31")


def test_add_compustat_inactivation_metadata_allows_missing_metadata():
    secm = pd.DataFrame(
        {
            "security_id": ["999999_01"],
            "date": pd.to_datetime(["2025-01-31"]),
        }
    )

    security = pd.DataFrame(
        {
            "security_id": ["001690_01"],
            "dldtei": pd.to_datetime([None]),
            "inactivation_code": [pd.NA],
        }
    )

    result = add_compustat_inactivation_metadata(secm, security)

    assert pd.isna(result.loc[0, "dldtei"])
    assert pd.isna(result.loc[0, "inactivation_code"])
    assert pd.isna(result.loc[0, "inactivation_month"])


def test_add_compustat_inactivation_metadata_rejects_duplicate_security_metadata():
    secm = pd.DataFrame(
        {
            "security_id": ["149738_01"],
            "date": pd.to_datetime(["2009-12-31"]),
        }
    )

    security = pd.DataFrame(
        {
            "security_id": [
                "149738_01",
                "149738_01",
            ],
            "dldtei": pd.to_datetime(
                [
                    "2009-12-10",
                    "2009-12-10",
                ]
            ),
            "inactivation_code": [
                "20",
                "20",
            ],
        }
    )

    with pytest.raises(pd.errors.MergeError):
        add_compustat_inactivation_metadata(secm, security)


def test_add_compustat_inactivation_metadata_does_not_mutate_inputs():
    secm = pd.DataFrame(
        {
            "security_id": ["149738_01"],
            "date": pd.to_datetime(["2009-12-31"]),
        }
    )

    security = pd.DataFrame(
        {
            "security_id": ["149738_01"],
            "dldtei": pd.to_datetime(["2009-12-10"]),
            "inactivation_code": ["20"],
        }
    )

    original_secm = secm.copy(deep=True)
    original_security = security.copy(deep=True)

    add_compustat_inactivation_metadata(secm, security)

    pd.testing.assert_frame_equal(secm, original_secm)
    pd.testing.assert_frame_equal(security, original_security)


def test_handle_compustat_terminal_stubs_nullifies_stub_return():
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2009-01-31"]),
            "inactivation_month": pd.to_datetime(
                ["2009-01-31"]
            ),
            "return_ex_delist": [0.0],
            "price": [float("nan")],
            "shares_outstanding": [float("nan")],
        }
    )

    result = handle_compustat_terminal_stubs(panel)

    assert result.loc[0, "terminal_stub"]
    assert pd.isna(result.loc[0, "return_ex_delist"])


def test_handle_compustat_terminal_stubs_preserves_genuine_zero_return():
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2009-01-31"]),
            "inactivation_month": pd.to_datetime(
                ["2009-01-31"]
            ),
            "return_ex_delist": [0.0],
            "price": [20.0],
            "shares_outstanding": [1_000_000],
        }
    )

    result = handle_compustat_terminal_stubs(panel)

    assert not result.loc[0, "terminal_stub"]
    assert result.loc[0, "return_ex_delist"] == 0.0


def test_handle_compustat_terminal_stubs_requires_inactivation_month():
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2008-12-31"]),
            "inactivation_month": pd.to_datetime(
                ["2009-01-31"]
            ),
            "return_ex_delist": [0.0],
            "price": [float("nan")],
            "shares_outstanding": [float("nan")],
        }
    )

    result = handle_compustat_terminal_stubs(panel)

    assert not result.loc[0, "terminal_stub"]
    assert result.loc[0, "return_ex_delist"] == 0.0


def test_handle_compustat_terminal_stubs_preserves_nonzero_terminal_return():
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2009-12-31"]),
            "inactivation_month": pd.to_datetime(
                ["2009-12-31"]
            ),
            "return_ex_delist": [-0.689655],
            "price": [0.045],
            "shares_outstanding": [100_000_000],
        }
    )

    result = handle_compustat_terminal_stubs(panel)

    assert not result.loc[0, "terminal_stub"]
    assert result.loc[0, "return_ex_delist"] == -0.689655


def test_handle_compustat_terminal_stubs_does_not_mutate_input():
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2009-01-31"]),
            "inactivation_month": pd.to_datetime(
                ["2009-01-31"]
            ),
            "return_ex_delist": [0.0],
            "price": [float("nan")],
            "shares_outstanding": [float("nan")],
        }
    )

    original = panel.copy(deep=True)
    handle_compustat_terminal_stubs(panel)

    pd.testing.assert_frame_equal(panel, original)


def test_handle_compustat_terminal_stubs_raises_for_missing_required_column():
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2009-01-31"]),
            "inactivation_month": pd.to_datetime(
                ["2009-01-31"]
            ),
            "return_ex_delist": [0.0],
            "price": [float("nan")],
            # shares_outstanding deliberately missing
        }
    )

    with pytest.raises(ValueError, match="missing required"):
        handle_compustat_terminal_stubs(panel)


def test_add_compustat_delisting_adjustment_imputes_bankruptcy():
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2009-01-31"]),
            "inactivation_month": pd.to_datetime(
                ["2009-01-31"]
            ),
            "inactivation_code": ["02"],
            "terminal_stub": [True],
            "return_ex_delist": [float("nan")],
        }
    )

    result = add_compustat_delisting_adjustment(panel)

    assert result.loc[0, "delisting_event"]
    assert result.loc[0, "delisting_return"] == -0.30
    assert result.loc[0, "return_ex_delist"] == 0.0


def test_add_compustat_delisting_adjustment_imputes_liquidation():
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2009-01-31"]),
            "inactivation_month": pd.to_datetime(
                ["2009-01-31"]
            ),
            "inactivation_code": ["03"],
            "terminal_stub": [True],
            "return_ex_delist": [float("nan")],
        }
    )

    result = add_compustat_delisting_adjustment(panel)

    assert result.loc[0, "delisting_event"]
    assert result.loc[0, "delisting_return"] == -0.30


def test_add_compustat_delisting_adjustment_does_not_impute_acquisition():
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2009-01-31"]),
            "inactivation_month": pd.to_datetime(
                ["2009-01-31"]
            ),
            "inactivation_code": ["01"],
            "terminal_stub": [True],
            "return_ex_delist": [float("nan")],
        }
    )

    result = add_compustat_delisting_adjustment(panel)

    assert not result.loc[0, "delisting_event"]
    assert pd.isna(result.loc[0, "delisting_return"])
    assert pd.isna(result.loc[0, "return_ex_delist"])


def test_add_compustat_delisting_adjustment_preserves_observed_return():
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2009-01-31"]),
            "inactivation_month": pd.to_datetime(
                ["2009-01-31"]
            ),
            "inactivation_code": ["02"],
            "terminal_stub": [False],
            "return_ex_delist": [-0.40],
        }
    )

    result = add_compustat_delisting_adjustment(panel)

    assert result.loc[0, "delisting_event"]
    assert result.loc[0, "return_ex_delist"] == -0.40
    assert result.loc[0, "delisting_return"] == -0.30


def test_add_compustat_delisting_adjustment_requires_inactivation_month():
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2008-12-31"]),
            "inactivation_month": pd.to_datetime(
                ["2009-01-31"]
            ),
            "inactivation_code": ["02"],
            "terminal_stub": [False],
            "return_ex_delist": [-0.20],
        }
    )

    result = add_compustat_delisting_adjustment(panel)

    assert not result.loc[0, "delisting_event"]
    assert pd.isna(result.loc[0, "delisting_return"])
    assert result.loc[0, "return_ex_delist"] == -0.20


def test_add_compustat_delisting_adjustment_does_not_mutate_input():
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2009-01-31"]),
            "inactivation_month": pd.to_datetime(
                ["2009-01-31"]
            ),
            "inactivation_code": ["02"],
            "terminal_stub": [True],
            "return_ex_delist": [float("nan")],
        }
    )

    original = panel.copy(deep=True)

    add_compustat_delisting_adjustment(panel)

    pd.testing.assert_frame_equal(panel, original)


def test_add_compustat_delisting_adjustment_raises_for_missing_required_column():
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2009-01-31"]),
            "inactivation_month": pd.to_datetime(
                ["2009-01-31"]
            ),
            "inactivation_code": ["02"],
            "terminal_stub": [True],
            # return_ex_delist deliberately missing
        }
    )

    with pytest.raises(ValueError, match="missing required"):
        add_compustat_delisting_adjustment(panel)


def test_assemble_compustat_monthly_panel():
    secm = pd.DataFrame(
        {
            "gvkey": ["001690"],
            "iid": ["01"],
            "datadate": ["2025-01-31"],
            "tic": ["AAPL"],
            "tpci": ["0"],
            "trt1m": [5.0],
            "prccm": [200.0],
            "cshom": [15_000_000_000],
        }
    )

    sec_history = pd.DataFrame(
        {
            "gvkey": ["001690"],
            "iid": ["01"],
            "effdate": ["2000-01-01"],
            "thrudate": [None],
            "item": ["EXCHG"],
            "itemvalue": ["14"],
        }
    )

    security = pd.DataFrame(
        {
            "gvkey": ["001690"],
            "iid": ["01"],
            "dldtei": [None],
            "dlrsni": [None],
        }
    )

    result = assemble_compustat_monthly_panel(
        secm,
        sec_history,
        security,
    )

    assert len(result) == 1

    row = result.iloc[0]

    assert row["security_id"] == "001690_01"
    assert row["date"] == pd.Timestamp("2025-01-31")
    assert row["ticker"] == "AAPL"
    assert row["exchange"] == "NASDAQ"
    assert row["security_type"] == "COMMON"

    assert row["return_ex_delist"] == pytest.approx(0.05)
    assert row["price"] == pytest.approx(200.0)
    assert row["shares_outstanding"] == pytest.approx(15_000_000_000)

    assert not row["delisting_event"]
    assert pd.isna(row["delisting_return"])
    assert not row["terminal_stub"]


def test_assemble_compustat_monthly_panel_handles_performance_delisting():
    secm = pd.DataFrame(
        {
            "gvkey": ["123456"],
            "iid": ["01"],
            "datadate": ["2009-12-31"],
            "tic": ["TEST"],
            "tpci": ["0"],
            "trt1m": [0.0],
            "prccm": [np.nan],
            "cshom": [np.nan],
        }
    )

    sec_history = pd.DataFrame(
        {
            "gvkey": ["123456"],
            "iid": ["01"],
            "effdate": ["2000-01-01"],
            "thrudate": [None],
            "item": ["EXCHG"],
            "itemvalue": ["11"],
        }
    )

    security = pd.DataFrame(
        {
            "gvkey": ["123456"],
            "iid": ["01"],
            "dldtei": ["2009-12-10"],
            "dlrsni": ["02"],
        }
    )

    result = assemble_compustat_monthly_panel(
        secm,
        sec_history,
        security,
    )

    assert len(result) == 1

    row = result.iloc[0]

    assert row["security_id"] == "123456_01"
    assert row["exchange"] == "NYSE"

    assert row["terminal_stub"]
    assert row["delisting_event"]

    assert row["return_ex_delist"] == pytest.approx(0.0)
    assert row["delisting_return"] == pytest.approx(-0.30)

    assert row["inactivation_code"] == "02"
    assert row["inactivation_month"] == pd.Timestamp(
        "2009-12-31"
    )


def test_assemble_compustat_monthly_panel_does_not_mutate_inputs():
    secm = pd.DataFrame(
        {
            "gvkey": ["001690"],
            "iid": ["01"],
            "datadate": ["2025-01-31"],
            "tic": ["AAPL"],
            "tpci": ["0"],
            "trt1m": [5.0],
            "prccm": [200.0],
            "cshom": [15_000_000_000],
        }
    )

    sec_history = pd.DataFrame(
        {
            "gvkey": ["001690"],
            "iid": ["01"],
            "effdate": ["2000-01-01"],
            "thrudate": [None],
            "item": ["EXCHG"],
            "itemvalue": ["14"],
        }
    )

    security = pd.DataFrame(
        {
            "gvkey": ["001690"],
            "iid": ["01"],
            "dldtei": [None],
            "dlrsni": [None],
        }
    )

    original_secm = secm.copy(deep=True)
    original_sec_history = sec_history.copy(deep=True)
    original_security = security.copy(deep=True)

    assemble_compustat_monthly_panel(
        secm,
        sec_history,
        security,
    )

    pd.testing.assert_frame_equal(secm, original_secm)
    pd.testing.assert_frame_equal(sec_history, original_sec_history)
    pd.testing.assert_frame_equal(security, original_security)


def test_compustat_panel_integrates_with_construct_equity_panel():
    secm = pd.DataFrame(
        {
            "gvkey": ["001690"],
            "iid": ["01"],
            "datadate": ["2025-01-31"],
            "tic": ["AAPL"],
            "tpci": ["0"],
            "trt1m": [5.0],
            "prccm": [200.0],
            "cshom": [15_000_000_000],
        }
    )

    sec_history = pd.DataFrame(
        {
            "gvkey": ["001690"],
            "iid": ["01"],
            "effdate": ["2000-01-01"],
            "thrudate": [None],
            "item": ["EXCHG"],
            "itemvalue": ["14"],
        }
    )

    security = pd.DataFrame(
        {
            "gvkey": ["001690"],
            "iid": ["01"],
            "dldtei": [None],
            "dlrsni": [None],
        }
    )

    compustat_panel = assemble_compustat_monthly_panel(
        secm,
        sec_history,
        security,
    )

    result = construct_equity_panel(compustat_panel)

    assert len(result) == 1

    row = result.iloc[0]

    assert row["security_id"] == "001690_01"
    assert row["exchange"] == "NASDAQ"
    assert row["security_type"] == "COMMON"

    assert row["return_ex_delist"] == pytest.approx(0.05)
    assert row["total_return"] == pytest.approx(0.05)

    assert row["price"] == pytest.approx(200.0)
    assert row["shares_outstanding"] == pytest.approx(15_000_000_000)
    assert row["market_cap"] == pytest.approx(3_000_000_000_000)


def test_compustat_delisting_integrates_with_construct_equity_panel():
    secm = pd.DataFrame(
        {
            "gvkey": ["123456"],
            "iid": ["01"],
            "datadate": ["2009-12-31"],
            "tic": ["TEST"],
            "tpci": ["0"],
            "trt1m": [-40.0],
            "prccm": [10.0],
            "cshom": [1_000_000],
        }
    )

    sec_history = pd.DataFrame(
        {
            "gvkey": ["123456"],
            "iid": ["01"],
            "effdate": ["2000-01-01"],
            "thrudate": [None],
            "item": ["EXCHG"],
            "itemvalue": ["11"],
        }
    )

    security = pd.DataFrame(
        {
            "gvkey": ["123456"],
            "iid": ["01"],
            "dldtei": ["2009-12-10"],
            "dlrsni": ["02"],
        }
    )

    compustat_panel = assemble_compustat_monthly_panel(
        secm,
        sec_history,
        security,
    )

    result = construct_equity_panel(compustat_panel)

    assert len(result) == 1

    row = result.iloc[0]

    assert row["return_ex_delist"] == pytest.approx(-0.40)
    assert row["delisting_event"]
    assert row["delisting_return"] == pytest.approx(-0.30)
    assert row["total_return"] == pytest.approx(-0.58)
    assert row["market_cap"] == pytest.approx(10_000_000)
    assert not row["terminal_stub"]