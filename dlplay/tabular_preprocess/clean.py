from typing import Callabel, Dict

import pandas as pd


def replace_invalids_with_na(df: pd.DataFrame, col_pred_map: Dict[str, Callable[[Any], bool]]) -> pd.DataFrame:
    df = df.copy()
    for col, pred in col_pred_map.items():
        if col not in df.columns:
            continue
        mask = df[col].apply(lambda v: bool(pred(v)))
        df.loc[mask, col] = pd.NA
    return df
