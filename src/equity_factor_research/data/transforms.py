import pandas as pd


def construct_total_return(
    return_ex_delist: pd.Series,
    delisting_return: pd.Series,
    delisting_event: pd.Series,
) -> pd.Series:

    if delisting_event.isna().any():
        raise ValueError("delisting_event must not contain missing values.")

    total_return = return_ex_delist.copy()

    compounded_return = (1 + return_ex_delist) * (1 + delisting_return) - 1

    total_return.loc[delisting_event] = compounded_return.loc[delisting_event]

    return total_return


def construct_market_cap(
    price: pd.Series,
    shares_outstanding: pd.Series,
) -> pd.Series:

    return price * shares_outstanding


def construct_equity_panel(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.copy()

    result["total_return"] = construct_total_return(
        result["return_ex_delist"],
        result["delisting_return"],
        result["delisting_event"],
    )

    if "market_cap" not in result.columns:
        result["market_cap"] = construct_market_cap(
            result["price"],
            result["shares_outstanding"],
        )

    return result