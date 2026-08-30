import pandas as pd


REQUIRED_COLUMNS = (
    "security_id",
    "date",
    "ticker",
    "exchange",
    "security_type",
    "return_ex_delist",
    "delisting_return",
    "total_return",
    "price",
    "shares_outstanding",
    "market_cap",
)

KEY_COLUMNS = ("security_id", "date")


def validate_equity_panel(panel: pd.DataFrame) -> None:
    """
    Validate the basic structure of a standardized monthly equity panel.

    Parameters
    ----------
    panel
        DataFrame containing standardized monthly equity data.

    Raises
    ------
    TypeError
        If panel is not a pandas DataFrame.
    ValueError
        If required columns are missing, key columns contain missing
        values, duplicate security-month observations exist, or the
        date column does not have a datetime dtype.
    """

    if not isinstance(panel, pd.DataFrame):
        raise TypeError("Input must be a pd.DataFrame.")

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in panel.columns]
    if missing_cols:
        raise ValueError(f"Panel is missing required columns: {missing_cols}.")

    for col in KEY_COLUMNS:
        if panel[col].isna().any():
            raise ValueError(f"Column '{col}' contains missing values.")

    if panel.duplicated(subset=list(KEY_COLUMNS)).any():
        raise ValueError("Duplicate records found for the composite key (security_id, date).")

    if not pd.api.types.is_datetime64_any_dtype(panel["date"].dtype):
        raise ValueError("Column 'date' must have a datetime dtype.")