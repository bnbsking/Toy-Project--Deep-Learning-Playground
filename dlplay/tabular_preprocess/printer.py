from typing import Optional
import pandas as pd


def print_df_info(df: pd.DataFrame, show_rows: int = 5, show_cols: Optional[int] = None) -> None:
    print("\n===DataFrame Info===")
    df.info()

    print("\n===Missing Percentage===")
    print(df.isna().sum() / len(df) * 100)

    print("\n===Number of Unique Values in Each Column===")
    print(df.nunique())

    print("\n===Duplicate Rows===")
    print(df.duplicated().sum())

    with pd.option_context('display.max_columns', show_cols):
        print("\n===DataFrame Description===")
        print(df.describe(include='all'))
        
        print(f"\n===First {show_rows} Rows===")
        print(df.head(show_rows))
