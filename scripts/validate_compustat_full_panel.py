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
    add_lagged_base_eligibility,
    add_universe_membership,
)
from equity_factor_research.data.quality import (
    assess_data_quality,
)


secm = pd.read_csv(
    "~/Downloads/compustat_secm_2000_2025.csv.gz",
    dtype={
        "gvkey": "string",
        "iid": "string",
        "tic": "string",
        "tpci": "string",
    },
    usecols=[
        "gvkey",
        "iid",
        "tic",
        "datadate",
        "tpci",
        "prccm",
        "trt1m",
        "cshom",
    ],
)

sec_history = pd.read_csv(
    "~/Downloads/compustat_sec_history_exchg.csv.gz",
    dtype={
        "gvkey": "string",
        "iid": "string",
        "item": "string",
        "itemvalue": "string",
    },
)

study_start = pd.Timestamp("2000-01-01")
study_end = pd.Timestamp("2025-12-31")

sec_history["effdate"] = pd.to_datetime(
    sec_history["effdate"]
)
sec_history["thrudate"] = pd.to_datetime(
    sec_history["thrudate"]
)

sec_history["security_id"] = (
    sec_history["gvkey"].str.zfill(6)
    + "_"
    + sec_history["iid"].str.zfill(2)
)

sec_history_overlap = sec_history[
    (sec_history["effdate"] <= study_end)
    & (
        sec_history["thrudate"].isna()
        | (
            sec_history["thrudate"]
            >= study_start
        )
    )
].copy()

history_ids = set(
    sec_history_overlap["security_id"].unique()
)

security = pd.read_csv(
    "~/Downloads/compustat_security.csv.gz",
    dtype={
        "gvkey": "string",
        "iid": "string",
        "dlrsni": "string",
    },
)


provider_panel = assemble_compustat_monthly_panel(
    secm,
    sec_history,
    security,
)

print("\nRETURN-CONSTRUCTION DTYPES:")
print(
    provider_panel[
        [
            "return_ex_delist",
            "delisting_return",
            "delisting_event",
        ]
    ].dtypes
)

print("\nDELISTING EVENT COUNT:")
print(provider_panel["delisting_event"].sum())


panel = construct_equity_panel(provider_panel)

validate_equity_panel(panel)

panel = add_lagged_market_cap(panel)
panel = add_base_eligibility(panel)
panel = add_lagged_base_eligibility(panel)
panel = add_universe_membership(
    panel,
    n=1000,
)

quality = assess_data_quality(panel)

print("RAW SECM ROWS:")
print(len(secm))

print("\nPROVIDER PANEL ROWS:")
print(len(provider_panel))

print("\nFINAL PANEL ROWS:")
print(len(panel))

print("\nUNIQUE SECURITIES:")
print(panel["security_id"].nunique())

print("\nDATE RANGE:")
print(panel["date"].min())
print(panel["date"].max())

print("\nDUPLICATE SECURITY-MONTHS:")
print(
    panel.duplicated(
        subset=["security_id", "date"]
    ).sum()
)

print("\nQUALITY DIAGNOSTICS:")
print(quality)

print("\nTERMINAL STUBS:")
print(panel["terminal_stub"].sum())

print("\nTERMINAL STUBS WITH MISSING RETURN:")
print(
    (
        panel["terminal_stub"]
        & panel["return_ex_delist"].isna()
    ).sum()
)

print("\nTERMINAL STUBS WITH NONMISSING RETURN:")
print(
    (
        panel["terminal_stub"]
        & panel["return_ex_delist"].notna()
    ).sum()
)

print("\nDELISTING EVENTS:")
print(panel["delisting_event"].sum())

print("\nDELISTING EVENTS BY INACTIVATION CODE:")
print(
    panel.loc[
        panel["delisting_event"],
        "inactivation_code",
    ].value_counts(dropna=False)
)

print("\nDELISTING RETURN DISTRIBUTION:")
print(
    panel.loc[
        panel["delisting_event"],
        "delisting_return",
    ].value_counts(dropna=False)
)


performance_terminal = (
    panel["terminal_stub"]
    & panel["delisting_event"]
)

print("\nPERFORMANCE-DELISTING TERMINAL STUBS:")
print(performance_terminal.sum())

print("\nRETURN_EX_DELIST FOR THOSE ROWS:")
print(
    panel.loc[
        performance_terminal,
        "return_ex_delist",
    ].value_counts(dropna=False)
)

print("\nTOTAL RETURN FOR PERFORMANCE TERMINAL STUBS:")
print(
    panel.loc[
        performance_terminal,
        "total_return",
    ].value_counts(dropna=False)
)

monthly_universe = (
    panel.groupby("date")
    .agg(
        n_rows=("security_id", "size"),
        n_base_eligible=("base_eligible", "sum"),
        n_lagged_eligible=("lagged_base_eligible", "sum"),
        n_in_universe=("in_universe", "sum"),
    )
)

print("\nMONTHLY UNIVERSE — FIRST 15 MONTHS:")
print(monthly_universe.head(15))

print("\nMONTHLY UNIVERSE — LAST 12 MONTHS:")
print(monthly_universe.tail(12))

print("\nMONTHLY UNIVERSE SUMMARY:")
print(monthly_universe.describe())


reported_sample = monthly_universe.loc[
    "2001-01-31":"2025-12-31"
]

print("\nREPORTED-SAMPLE UNIVERSE SIZE:")
print(
    reported_sample["n_in_universe"].describe()
)

print("\nMONTHS WITH FEWER THAN 1000 STOCKS:")
print(
    reported_sample[
        reported_sample["n_in_universe"] < 1000
    ]
)

print("\nBASE-ELIGIBLE EXCHANGE DISTRIBUTION:")
print(
    panel.loc[
        panel["base_eligible"],
        "exchange",
    ].value_counts(dropna=False)
)

print("\nIN-UNIVERSE SECURITY-TYPE DISTRIBUTION:")
print(
    panel.loc[
        panel["in_universe"],
        "security_type",
    ].value_counts(dropna=False)
)

print("\nIN-UNIVERSE ROWS WITH MISSING TOTAL RETURN:")
print(
    (
        panel["in_universe"]
        & panel["total_return"].isna()
    ).sum()
)

print("\nIN-UNIVERSE TERMINAL STUBS WITH MISSING TOTAL RETURN:")
print(
    (
        panel["in_universe"]
        & panel["terminal_stub"]
        & panel["total_return"].isna()
    ).sum()
)

missing_in_universe_return = (
    panel["in_universe"]
    & panel["total_return"].isna()
)

missing_nonterminal_return = (
    missing_in_universe_return
    & ~panel["terminal_stub"]
)

print("\nMISSING IN-UNIVERSE RETURNS BY MONTH:")
missing_by_month = (
    panel.loc[missing_in_universe_return]
    .groupby("date")
    .size()
)

print(missing_by_month)

print("\nMISSING IN-UNIVERSE RETURN SUMMARY:")
print(missing_by_month.describe())

print("\nNON-TERMINAL MISSING IN-UNIVERSE RETURNS:")
print(
    panel.loc[
        missing_nonterminal_return,
        [
            "security_id",
            "date",
            "ticker",
            "exchange",
            "security_type",
            "return_ex_delist",
            "delisting_event",
            "terminal_stub",
            "lagged_market_cap",
        ],
    ]
    .sort_values(["date", "security_id"])
    .to_string(index=False)
)


monthly_cutoff = (
    panel.loc[
        panel["in_universe"]
    ]
    .groupby("date")["lagged_market_cap"]
    .min()
    .rename("top_1000_cutoff")
)

print("\nTOP-1000 CUTOFF SUMMARY — REPORTED SAMPLE:")
print(
    monthly_cutoff.loc[
        "2001-01-31":"2025-12-31"
    ].describe()
)

print("\nTOP-1000 CUTOFF — FIRST 12 REPORTED MONTHS:")
print(
    monthly_cutoff.loc[
        "2001-01-31":"2001-12-31"
    ]
)

print("\nTOP-1000 CUTOFF — LAST 12 MONTHS:")
print(
    monthly_cutoff.loc[
        "2025-01-31":"2025-12-31"
    ]
)


ordered = panel.sort_values(
    ["security_id", "date"]
).copy()

grouped = ordered.groupby(
    "security_id",
    sort=False,
)

ordered["previous_date"] = (
    grouped["date"].shift(1)
)

ordered["lagged_exchange_diagnostic"] = (
    grouped["exchange"].shift(1)
)

ordered["lagged_security_type_diagnostic"] = (
    grouped["security_type"].shift(1)
)

expected_previous_date = (
    ordered["date"]
    - pd.offsets.MonthEnd(1)
)

adjacent_month = (
    ordered["previous_date"]
    == expected_previous_date
)

ordered.loc[
    ~adjacent_month,
    [
        "lagged_exchange_diagnostic",
        "lagged_security_type_diagnostic",
    ],
] = pd.NA

ordered = ordered.merge(
    monthly_cutoff,
    left_on="date",
    right_index=True,
    how="left",
    validate="many_to_one",
)

potential_missing_exchange_impact = (
    ordered["lagged_market_cap"].notna()
    & (
        ordered["lagged_security_type_diagnostic"]
        == "COMMON"
    )
    & ordered["lagged_exchange_diagnostic"].isna()
    & ordered["top_1000_cutoff"].notna()
    & (
        ordered["lagged_market_cap"]
        >= ordered["top_1000_cutoff"]
    )
    & (
        ordered["date"]
        >= pd.Timestamp("2001-01-31")
    )
)

print(
    "\nPOTENTIALLY MATERIAL UNKNOWN-EXCHANGE SECURITY-MONTHS:"
)
print(
    potential_missing_exchange_impact.sum()
)

print(
    "\nUNIQUE SECURITIES:"
)
print(
    ordered.loc[
        potential_missing_exchange_impact,
        "security_id",
    ].nunique()
)

print(
    "\nAFFECTED MONTHS:"
)
print(
    ordered.loc[
        potential_missing_exchange_impact,
        "date",
    ].nunique()
)

print(
    "\nLARGEST POTENTIALLY MATERIAL CASES:"
)
print(
    ordered.loc[
        potential_missing_exchange_impact,
        [
            "security_id",
            "date",
            "ticker",
            "lagged_market_cap",
            "top_1000_cutoff",
        ],
    ]
    .sort_values(
        "lagged_market_cap",
        ascending=False,
    )
    .head(30)
    .to_string(index=False)
)

print("\nEXAMPLE PERFORMANCE DELISTINGS:")
print(
    panel.loc[
        panel["delisting_event"],
        [
            "security_id",
            "date",
            "ticker",
            "inactivation_code",
            "terminal_stub",
            "return_ex_delist",
            "delisting_return",
            "total_return",
            "in_universe",
        ],
    ]
    .sort_values(["date", "security_id"])
    .head(20)
    .to_string(index=False)
)

candidate_rows = (
    ordered.loc[
        potential_missing_exchange_impact,
        [
            "security_id",
            "date",
            "ticker",
            "lagged_market_cap",
            "top_1000_cutoff",
        ],
    ]
    .copy()
    .reset_index(drop=True)
)

candidate_rows["candidate_id"] = candidate_rows.index

candidate_rows["formation_date"] = (
    candidate_rows["date"]
    - pd.offsets.MonthEnd(1)
)

candidate_history = candidate_rows.merge(
    sec_history[
        [
            "security_id",
            "itemvalue",
            "effdate",
            "thrudate",
        ]
    ],
    on="security_id",
    how="left",
)

effective_history = candidate_history[
    candidate_history["effdate"].isna()
    | (
        candidate_history["effdate"]
        <= candidate_history["formation_date"]
    )
].copy()

latest_history = (
    effective_history
    .sort_values(
        [
            "candidate_id",
            "effdate",
        ]
    )
    .groupby(
        "candidate_id",
        as_index=False,
        dropna=False,
    )
    .tail(1)
)

latest_history["history_valid"] = (
    latest_history["effdate"].notna()
    & (
        latest_history["thrudate"].isna()
        | (
            latest_history["thrudate"]
            >= latest_history["formation_date"]
        )
    )
)

candidate_classification = candidate_rows.merge(
    latest_history[
        [
            "candidate_id",
            "itemvalue",
            "effdate",
            "thrudate",
            "history_valid",
        ]
    ],
    on="candidate_id",
    how="left",
    validate="one_to_one",
)

candidate_classification["history_valid"] = (
    candidate_classification[
        "history_valid"
    ]
    .fillna(False)
    .astype(bool)
)

eligible_exchange_codes = {
    "11",
    "12",
    "14",
}

candidate_classification["known_noneligible_exchange"] = (
    candidate_classification["history_valid"]
    & ~candidate_classification[
        "itemvalue"
    ].isin(
        eligible_exchange_codes
    )
)

candidate_classification["unexpected_eligible_exchange"] = (
    candidate_classification["history_valid"]
    & candidate_classification[
        "itemvalue"
    ].isin(
        eligible_exchange_codes
    )
)

candidate_classification["true_unknown_exchange"] = (
    ~candidate_classification["history_valid"]
)

print(
    "\nBROAD UNKNOWN-EXCHANGE ABOVE-CUTOFF CANDIDATES:"
)
print(len(candidate_classification))

print(
    "\nKNOWN NON-ELIGIBLE EXCHANGE:"
)
print(
    candidate_classification[
        "known_noneligible_exchange"
    ].sum()
)

print(
    "\nTRUE UNKNOWN EXCHANGE:"
)
print(
    candidate_classification[
        "true_unknown_exchange"
    ].sum()
)

print(
    "\nUNEXPECTED ELIGIBLE EXCHANGE:"
)
print(
    candidate_classification[
        "unexpected_eligible_exchange"
    ].sum()
)

print(
    "\nLARGEST KNOWN NON-ELIGIBLE CASES:"
)
print(
    candidate_classification.loc[
        candidate_classification[
            "known_noneligible_exchange"
        ],
        [
            "security_id",
            "date",
            "ticker",
            "formation_date",
            "itemvalue",
            "lagged_market_cap",
            "top_1000_cutoff",
        ],
    ]
    .sort_values(
        "lagged_market_cap",
        ascending=False,
    )
    .head(20)
    .to_string(index=False)
)

true_unknown = candidate_classification[
    candidate_classification[
        "true_unknown_exchange"
    ]
].copy()

true_unknown["has_any_study_history"] = (
    true_unknown["security_id"].isin(
        history_ids
    )
)

print(
    "\nTRUE UNKNOWN — NO STUDY-PERIOD HISTORY:"
)
print(
    (
        ~true_unknown[
            "has_any_study_history"
        ]
    ).sum()
)

print("UNIQUE SECURITIES:")
print(
    true_unknown.loc[
        ~true_unknown[
            "has_any_study_history"
        ],
        "security_id",
    ].nunique()
)

print(
    "\nTRUE UNKNOWN — GAP IN EXISTING HISTORY:"
)
print(
    true_unknown[
        "has_any_study_history"
    ].sum()
)

print("UNIQUE SECURITIES:")
print(
    true_unknown.loc[
        true_unknown[
            "has_any_study_history"
        ],
        "security_id",
    ].nunique()
)

print(
    "\nLARGEST TRUE UNKNOWN CASES:"
)
print(
    true_unknown[
        [
            "security_id",
            "date",
            "ticker",
            "formation_date",
            "lagged_market_cap",
            "top_1000_cutoff",
        ]
    ]
    .sort_values(
        "lagged_market_cap",
        ascending=False,
    )
    .head(30)
    .to_string(index=False)
)


missing_nonterminal = panel.loc[
    missing_nonterminal_return
].copy()

print(
    "\nNON-TERMINAL MISSING RETURNS BY "
    "INACTIVATION CODE:"
)
print(
    missing_nonterminal[
        "inactivation_code"
    ].value_counts(dropna=False)
)

print(
    "\nSAME MONTH AS INACTIVATION:"
)
print(
    (
        missing_nonterminal["date"]
        == missing_nonterminal[
            "inactivation_month"
        ]
    ).value_counts(dropna=False)
)

gap_candidates = true_unknown[
    true_unknown["has_any_study_history"]
].copy()

gap_candidates = gap_candidates.rename(
    columns={
        "itemvalue": "previous_exchange_code",
        "effdate": "previous_effdate",
        "thrudate": "previous_thrudate",
    }
)

future_history = gap_candidates[
    [
        "candidate_id",
        "security_id",
        "formation_date",
    ]
].merge(
    sec_history_overlap[
        [
            "security_id",
            "itemvalue",
            "effdate",
            "thrudate",
        ]
    ],
    on="security_id",
    how="left",
)

future_history = future_history[
    future_history["effdate"]
    > future_history["formation_date"]
].copy()

next_history = (
    future_history
    .sort_values(
        [
            "candidate_id",
            "effdate",
        ]
    )
    .groupby(
        "candidate_id",
        as_index=False,
    )
    .head(1)
    .rename(
        columns={
            "itemvalue": "next_exchange_code",
            "effdate": "next_effdate",
            "thrudate": "next_thrudate",
        }
    )
)

gap_candidates = gap_candidates.merge(
    next_history[
        [
            "candidate_id",
            "next_exchange_code",
            "next_effdate",
            "next_thrudate",
        ]
    ],
    on="candidate_id",
    how="left",
    validate="one_to_one",
)

gap_candidates["previous_is_eligible"] = (
    gap_candidates[
        "previous_exchange_code"
    ].isin(
        eligible_exchange_codes
    )
)

gap_candidates["next_is_eligible"] = (
    gap_candidates[
        "next_exchange_code"
    ].isin(
        eligible_exchange_codes
    )
)

gap_candidates["eligible_on_either_side"] = (
    gap_candidates["previous_is_eligible"]
    | gap_candidates["next_is_eligible"]
)

print(
    "\nHISTORY-GAP CANDIDATES:"
)
print(len(gap_candidates))

print(
    "\nGAPS WITH ELIGIBLE EXCHANGE ON EITHER SIDE:"
)
print(
    gap_candidates[
        "eligible_on_either_side"
    ].sum()
)

print(
    "\nUNIQUE SECURITIES WITH ELIGIBLE EXCHANGE "
    "ON EITHER SIDE:"
)
print(
    gap_candidates.loc[
        gap_candidates[
            "eligible_on_either_side"
        ],
        "security_id",
    ].nunique()
)

print(
    "\nPREVIOUS / NEXT EXCHANGE-CODE PAIRS:"
)
print(
    gap_candidates.groupby(
        [
            "previous_exchange_code",
            "next_exchange_code",
        ],
        dropna=False,
    )
    .size()
    .sort_values(
        ascending=False
    )
    .head(30)
)

print(
    "\nLARGEST GAPS WITH ELIGIBLE EXCHANGE "
    "ON EITHER SIDE:"
)

print(
    gap_candidates.loc[
        gap_candidates[
            "eligible_on_either_side"
        ],
        [
            "security_id",
            "ticker",
            "date",
            "formation_date",
            "previous_exchange_code",
            "previous_thrudate",
            "next_exchange_code",
            "next_effdate",
            "lagged_market_cap",
            "top_1000_cutoff",
        ],
    ]
    .sort_values(
        "lagged_market_cap",
        ascending=False,
    )
    .head(30)
    .to_string(index=False)
)