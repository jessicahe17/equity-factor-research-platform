from pathlib import Path
import pandas as pd

from equity_factor_research.data.providers.compustat import (
    assemble_compustat_monthly_panel,
)
from equity_factor_research.data.transforms import (
    construct_equity_panel,
)
from equity_factor_research.data.schema import (
    validate_equity_panel,
)
from equity_factor_research.data.universe import (
    add_lagged_market_cap,
    add_base_eligibility,
    add_universe_membership,
    add_lagged_base_eligibility,
)
from equity_factor_research.data.quality import (
    assess_data_quality,
)


columns = [
    "security_id",
    "date",
    "ticker",
    "exchange",
    "security_type",
    "return_ex_delist",
    "price",
    "shares_outstanding",
    "delisting_event",
    "delisting_return",
    "terminal_stub",
]


downloads = Path.home() / "Downloads"

secm = pd.read_csv(
    downloads / "compustat_secm_test.csv",
    dtype={
        "gvkey": "string",
        "iid": "string",
        "tpci": "string",
    },
)

sec_history = pd.read_csv(
    downloads / "compustat_sec_history_test.csv",
    dtype={
        "gvkey": "string",
        "iid": "string",
        "item": "string",
        "itemvalue": "string",
    },
)

security = pd.read_csv(
    downloads / "compustat_inactive_security_test.csv",
    dtype={
        "gvkey": "string",
        "iid": "string",
        "dlrsni": "string",
    },
)

print("\nRAW DATA SHAPES")
print("secm:", secm.shape)
print("sec_history:", sec_history.shape)
print("security:", security.shape)

print("\nRAW IDENTIFIER CHECK")
print(secm[["gvkey", "iid"]].head())
print(sec_history[["gvkey", "iid"]].head())
print(security[["gvkey", "iid"]].head())

print("\nSECURITY DUPLICATE IDS")
print(
    security.duplicated(
        subset=["gvkey", "iid"],
        keep=False,
    ).sum()
)

print("\nHISTORICAL EXCHANGE VALUES")
print(
    sec_history.loc[
        sec_history["item"] == "EXCHG",
        "itemvalue",
    ].value_counts(dropna=False)
)

print("\n" + "=" * 60)
print("REAL TERMINAL / DELISTING SAMPLE")
print("=" * 60)

terminal = pd.read_csv(
    downloads / "compustat_terminal_return_test.csv",
    dtype={
        "gvkey": "string",
        "iid": "string",
        "tpci": "string",
    },
)

print("\nTERMINAL SAMPLE SHAPE")
print(terminal.shape)

print("\nTERMINAL SAMPLE COLUMNS")
print(terminal.columns.tolist())

print("\nTERMINAL SAMPLE")
print(
    terminal[
        [
            "gvkey",
            "iid",
            "datadate",
            "tic",
            "trt1m",
            "prccm",
            "cshom",
        ]
    ].sort_values(["gvkey", "iid", "datadate"])
)

terminal_ids = terminal[
    ["gvkey", "iid"]
].drop_duplicates()

terminal_security = security.merge(
    terminal_ids,
    on=["gvkey", "iid"],
    how="inner",
)

print("\nMATCHING SECURITY METADATA")
print(
    terminal_security[
        [
            "gvkey",
            "iid",
            "dldtei",
            "dlrsni",
        ]
    ]
)

terminal_sec_history = pd.read_csv(
    downloads / "compustat_terminal_sec_history_test.csv",
    dtype={
        "gvkey": "string",
        "iid": "string",
        "item": "string",
        "itemvalue": "string",
    },
)

terminal_sec_history = terminal_sec_history.merge(
    terminal_ids,
    on=["gvkey", "iid"],
    how="inner",
)

print("\nTERMINAL SECURITY HISTORY")
print(
    terminal_sec_history[
        [
            "gvkey",
            "iid",
            "effdate",
            "thrudate",
            "item",
            "itemvalue",
        ]
    ].sort_values(
        ["gvkey", "iid", "effdate"]
    )
)

terminal_provider_panel = assemble_compustat_monthly_panel(
    terminal,
    terminal_sec_history,
    terminal_security,
)

terminal_panel = construct_equity_panel(terminal_provider_panel)

terminal_panel = add_lagged_market_cap(terminal_panel)

terminal_panel = add_base_eligibility(terminal_panel)

terminal_panel = add_lagged_base_eligibility(terminal_panel)

terminal_panel = add_universe_membership(
    terminal_panel,
    n=1000,
)

print("\nTERMINAL UNIVERSE CHECK")
print(
    terminal_panel.loc[
        terminal_panel["date"]
        == terminal_panel["inactivation_month"],
        [
            "security_id",
            "date",
            "exchange",
            "base_eligible",
            "lagged_base_eligible",
            "market_cap",
            "lagged_market_cap",
            "terminal_stub",
            "total_return",
            "in_universe",
        ],
    ]
)

print("\nTERMINAL PROVIDER PANEL")
print(
    terminal_provider_panel[
        [
            "security_id",
            "date",
            "ticker",
            "exchange",
            "security_type",
            "return_ex_delist",
            "price",
            "shares_outstanding",
            "dldtei",
            "inactivation_code",
            "inactivation_month",
            "terminal_stub",
            "delisting_event",
            "delisting_return",
        ]
    ].sort_values(
        ["security_id", "date"]
    )
)

print("\nTERMINAL / INACTIVATION ROWS")
print(
    terminal_provider_panel.loc[
        terminal_provider_panel["date"]
        == terminal_provider_panel["inactivation_month"],
        [
            "security_id",
            "date",
            "ticker",
            "exchange",
            "return_ex_delist",
            "price",
            "shares_outstanding",
            "inactivation_code",
            "terminal_stub",
            "delisting_event",
            "delisting_return",
        ],
    ]
)


provider_panel = assemble_compustat_monthly_panel(
    secm,
    sec_history,
    security,
)

print("\nPROVIDER PANEL")
print(
    provider_panel[columns]
    .sort_values(["security_id", "date"])
)

panel = construct_equity_panel(provider_panel)

validate_equity_panel(panel)

print("\nCONSTRUCTED EQUITY PANEL")
print(
    panel[
        [
            "security_id",
            "date",
            "ticker",
            "exchange",
            "security_type",
            "return_ex_delist",
            "total_return",
            "price",
            "shares_outstanding",
            "market_cap",
        ]
    ].sort_values(["security_id", "date"])
)

panel = add_lagged_market_cap(panel)

print(
    panel[
        [
            "security_id",
            "date",
            "market_cap",
            "lagged_market_cap",
        ]
    ].sort_values(["security_id", "date"])
)

panel = add_base_eligibility(panel)

print(
    panel[
        [
            "security_id",
            "date",
            "exchange",
            "security_type",
            "base_eligible",
        ]
    ]
)

panel = add_lagged_base_eligibility(panel)

panel = add_universe_membership(panel, n=1000)

print(
    panel[
        [
            "security_id",
            "date",
            "base_eligible",
            "lagged_base_eligible",
            "lagged_market_cap",
            "in_universe",
        ]
    ].sort_values(["date", "security_id"])
)


quality = assess_data_quality(panel)

print("\nDATA QUALITY")
print(quality)

print("\nDUPLICATE SECURITY-MONTHS")
print(
    panel.duplicated(
        subset=["security_id", "date"],
        keep=False,
    ).sum()
)

print("\nPANEL SIZE")
print(panel.shape)

print("\nDTYPES")
print(panel.dtypes)

print("\nMISSINGNESS")
print(
    panel[
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