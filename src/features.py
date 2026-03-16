import pandas as pd


def add_date_features(df):
    df = df.copy()

    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["day"] = df["date"].dt.day
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    return df


def add_lag_features(df):
    df = df.copy()

    df = df.sort_values(["store_id", "item_id", "date"])

    df["lag_7"] = df.groupby(["store_id", "item_id"])["sales"].shift(7)
    df["lag_28"] = df.groupby(["store_id", "item_id"])["sales"].shift(28)

    return df


def add_rolling_features(df):
    df = df.copy()

    df = df.sort_values(["store_id", "item_id", "date"])

    df["rolling_mean_7"] = (
        df.groupby(["store_id", "item_id"])["sales"]
        .transform(lambda x: x.shift(1).rolling(7).mean())
    )

    df["rolling_mean_28"] = (
        df.groupby(["store_id", "item_id"])["sales"]
        .transform(lambda x: x.shift(1).rolling(28).mean())
    )

    return df


def build_features(df):
    df = add_date_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    return df