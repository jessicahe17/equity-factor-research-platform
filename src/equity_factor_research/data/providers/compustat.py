import pandas as pd


COMPUSTAT_SECM_REQUIRED_COLUMNS = (
    "gvkey",
    "iid",
    "datadate",
    "tic",
    "tpci",
    "trt1m",
    "prccm",
    "cshom",
)

COMPUSTAT_SEC_HISTORY_REQUIRED_COLUMNS = (
    "gvkey",
    "iid",
    "effdate",
    "thrudate",
    "item",
    "itemvalue",
)


COMPUSTAT_EXCHANGE_MAP = {
    11: "NYSE",
    12: "AMEX",
    14: "NASDAQ",
}

COMPUSTAT_SECURITY_REQUIRED_COLUMNS = (
    "gvkey",
    "iid",
    "dldtei",
    "dlrsni",
)

COMPUSTAT_TERMINAL_STUB_REQUIRED_COLUMNS = (
    "date",
    "inactivation_month",
    "return_ex_delist",
    "price",
    "shares_outstanding",
)

COMPUSTAT_PERFORMANCE_INACTIVATION_CODES = {
    "02",
    "03",
}

COMPUSTAT_DELISTING_REQUIRED_COLUMNS = (
    "date",
    "inactivation_month",
    "inactivation_code",
    "terminal_stub",
    "return_ex_delist",
)


def normalize_compustat_secm(secm: pd.DataFrame) -> pd.DataFrame:
    """Normalize Compustat Security Monthly data."""

    missing_columns = [
        column
        for column in COMPUSTAT_SECM_REQUIRED_COLUMNS
        if column not in secm.columns
    ]

    if missing_columns:
        raise ValueError(
            "Compustat Security Monthly data is missing required "
            f"columns: {missing_columns}."
        )

    result = secm.copy()

    result["gvkey"] = result["gvkey"].astype("string")
    result["iid"] = result["iid"].astype("string")

    common_stock = pd.to_numeric(
        result["tpci"],
        errors="coerce",
    ).eq(0)

    result["tpci"] = result["tpci"].astype("string")

    result["security_id"] = (
        result["gvkey"] + "_" + result["iid"]
    )

    result["datadate"] = pd.to_datetime(
        result["datadate"]
    )

    result["trt1m"] = pd.to_numeric(
        result["trt1m"],
        errors="raise",
    )
    result["prccm"] = pd.to_numeric(
        result["prccm"],
        errors="raise",
    )
    result["cshom"] = pd.to_numeric(
        result["cshom"],
        errors="raise",
    )

    result["security_type"] = result["tpci"].copy()
    result.loc[
        common_stock,
        "security_type",
    ] = "COMMON"

    result["return_ex_delist"] = (
        result["trt1m"] / 100
    )

    result = result.rename(
        columns={
            "datadate": "date",
            "tic": "ticker",
            "prccm": "price",
            "cshom": "shares_outstanding",
        }
    )

    return result


def normalize_compustat_sec_history(sec_history: pd.DataFrame) -> pd.DataFrame:
    """Normalize Compustat Security Historical Identifiers data."""

    missing_columns = [
        column
        for column in COMPUSTAT_SEC_HISTORY_REQUIRED_COLUMNS
        if column not in sec_history.columns
    ]

    if missing_columns:
        raise ValueError(
            "Compustat Security Historical Identifiers data is "
            f"missing required columns: {missing_columns}."
        )

    result = sec_history.copy()

    result["gvkey"] = result["gvkey"].astype("string")
    result["iid"] = result["iid"].astype("string")

    result["security_id"] = (
        result["gvkey"] + "_" + result["iid"]
    )

    result["effdate"] = pd.to_datetime(
        result["effdate"]
    )

    result["thrudate"] = pd.to_datetime(
        result["thrudate"]
    )

    result["item"] = result["item"].astype("string")
    result["itemvalue"] = result["itemvalue"].astype("string")

    return result


def add_compustat_historical_exchange(
    secm: pd.DataFrame,
    sec_history: pd.DataFrame,
) -> pd.DataFrame:
    """Attach point-in-time historical exchange to Compustat monthly data."""

    result = secm.copy()

    exchange_history = sec_history.loc[
        sec_history["item"].eq("EXCHG"),
        [
            "security_id",
            "effdate",
            "thrudate",
            "itemvalue",
        ],
    ].copy()

    exchange_history["exchange_code"] = pd.to_numeric(
        exchange_history["itemvalue"],
        errors="coerce",
    )

    result = result.sort_values(
        ["date", "security_id"]
    )

    exchange_history = exchange_history.sort_values(
        ["effdate", "security_id"]
    )

    result = pd.merge_asof(
        result,
        exchange_history[
            [
                "security_id",
                "effdate",
                "thrudate",
                "exchange_code",
            ]
        ],
        left_on="date",
        right_on="effdate",
        by="security_id",
        direction="backward",
        allow_exact_matches=True,
    )

    valid_interval = (
        result["effdate"].notna()
        & (
            result["thrudate"].isna()
            | result["date"].le(result["thrudate"])
        )
    )

    result.loc[
        ~valid_interval,
        "exchange_code",
    ] = pd.NA

    result["exchange"] = (
        result["exchange_code"]
        .map(COMPUSTAT_EXCHANGE_MAP)
        .astype("string")
    )

    return result


def normalize_compustat_security(security: pd.DataFrame) -> pd.DataFrame:
    """Normalize Compustat security-level inactivation metadata."""

    missing_columns = [
        column
        for column in COMPUSTAT_SECURITY_REQUIRED_COLUMNS
        if column not in security.columns
    ]

    if missing_columns:
        raise ValueError(
            "Compustat Security data is missing required "
            f"columns: {missing_columns}."
        )

    result = security.copy()

    result["gvkey"] = result["gvkey"].astype("string")
    result["iid"] = result["iid"].astype("string")

    result["security_id"] = (
        result["gvkey"] + "_" + result["iid"]
    )

    result["dldtei"] = pd.to_datetime(
        result["dldtei"]
    )

    result["dlrsni"] = result["dlrsni"].astype("string")

    result["inactivation_code"] = (
        pd.to_numeric(
            result["dlrsni"],
            errors="coerce",
        )
        .astype("Int64")
        .astype("string")
        .str.zfill(2)
    )

    return result


def add_compustat_inactivation_metadata(
    secm: pd.DataFrame,
    security: pd.DataFrame,
) -> pd.DataFrame:
    """Attach security-level inactivation metadata to monthly Compustat data."""

    result = secm.copy()

    metadata = security[
        [
            "security_id",
            "dldtei",
            "inactivation_code",
        ]
    ].copy()

    result = result.merge(
        metadata,
        on="security_id",
        how="left",
        validate="many_to_one",
    )

    result["inactivation_month"] = (
        result["dldtei"]
        .dt.to_period("M")
        .dt.to_timestamp("M")
    )

    return result


def handle_compustat_terminal_stubs(panel: pd.DataFrame) -> pd.DataFrame:
    """Identify and nullify Compustat terminal stub returns."""

    missing_columns = [
        column
        for column in COMPUSTAT_TERMINAL_STUB_REQUIRED_COLUMNS
        if column not in panel.columns
    ]

    if missing_columns:
        raise ValueError(
            "Compustat monthly data is missing required "
            f"columns: {missing_columns}."
        )

    result = panel.copy()

    same_month = (
        result["date"].dt.to_period("M")
        == result["inactivation_month"].dt.to_period("M")
    )

    terminal_stub = (
        result["inactivation_month"].notna()
        & same_month
        & result["price"].isna()
        & result["shares_outstanding"].isna()
        & result["return_ex_delist"].eq(0)
    )

    result["terminal_stub"] = terminal_stub

    result.loc[
        terminal_stub,
        "return_ex_delist",
    ] = pd.NA

    return result


def add_compustat_delisting_adjustment(
    panel: pd.DataFrame,
    performance_delisting_return: float = -0.30,
) -> pd.DataFrame:
    """Add Compustat performance-related delisting adjustments."""

    missing_columns = [
        column
        for column in COMPUSTAT_DELISTING_REQUIRED_COLUMNS
        if column not in panel.columns
    ]

    if missing_columns:
        raise ValueError(
            "Compustat monthly data is missing required "
            f"columns: {missing_columns}."
        )

    if not -1 <= performance_delisting_return <= 0:
        raise ValueError(
            "performance_delisting_return must be between -1 and 0."
        )

    result = panel.copy()

    same_month = (
        result["date"].dt.to_period("M")
        == result["inactivation_month"].dt.to_period("M")
    )

    performance_related = (
        result["inactivation_code"]
        .isin(COMPUSTAT_PERFORMANCE_INACTIVATION_CODES)
    )

    adjustment_event = (
        result["inactivation_month"].notna()
        & same_month
        & performance_related
    )

    result["delisting_event"] = adjustment_event

    result["delisting_return"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Float64",
    )

    result.loc[
        adjustment_event,
        "delisting_return",
    ] = performance_delisting_return

    # If the performance-related inactivation month is a
    # terminal stub, use a neutral ordinary-return component
    # so the imputed delisting return becomes the month's
    # economic return.
    terminal_adjustment = (
        adjustment_event
        & result["terminal_stub"]
    )

    result.loc[terminal_adjustment, "return_ex_delist"] = 0.0

    return result


def assemble_compustat_monthly_panel(
    secm: pd.DataFrame,
    sec_history: pd.DataFrame,
    security: pd.DataFrame,
    performance_delisting_return: float = -0.30,
) -> pd.DataFrame:
    """Assemble standardized Compustat monthly construction inputs."""

    normalized_secm = normalize_compustat_secm(secm)
    normalized_history = normalize_compustat_sec_history(sec_history)

    result = add_compustat_historical_exchange(
        normalized_secm,
        normalized_history,
    )

    normalized_security = normalize_compustat_security(security)

    result = add_compustat_inactivation_metadata(
        result,
        normalized_security,
    )

    result = handle_compustat_terminal_stubs(result)

    result = add_compustat_delisting_adjustment(
        result,
        performance_delisting_return=performance_delisting_return,
    )

    return result