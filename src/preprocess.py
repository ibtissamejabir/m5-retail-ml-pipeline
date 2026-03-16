import pandas as pd


def merge_with_calendar(sales_long_df, calendar_df):
    merged_df = sales_long_df.merge(calendar_df, on="d", how="left")
    return merged_df


def merge_with_prices(merged_df, prices_df):
    merged_df = merged_df.merge(
        prices_df,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left"
    )
    return merged_df


def basic_cleaning(df):
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store_id", "item_id", "date"])

    return df


def preprocess_m5_data(sales_long_df, calendar_df, prices_df):
    df = merge_with_calendar(sales_long_df, calendar_df)
    df = merge_with_prices(df, prices_df)
    df = basic_cleaning(df)

    return df