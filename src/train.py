from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os


FEATURE_COLUMNS = [
    "sell_price",
    "day_of_week",
    "month",
    "year",
    "day",
    "is_weekend",
    "lag_7",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
]

TARGET_COLUMN = "sales"


def prepare_training_data(df):
    df = df.copy()

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN, "date"]
    df = df[required_columns]

    df = df.dropna()

    return df


def split_train_test(df, train_ratio=0.8):
    df = df.sort_values("date")

    split_index = int(len(df) * train_ratio)

    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }

    return metrics


def save_model(model, output_path="models/random_forest_model.pkl"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)