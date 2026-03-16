import joblib
import pandas as pd
import os


def load_saved_model(model_path="models/random_forest_model.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not found")

    model = joblib.load(model_path)
    return model


def make_predictions(model, X_test):
    predictions = model.predict(X_test)
    return predictions


def create_prediction_dataframe(X_test, y_test, predictions):
    results_df = X_test.copy()
    results_df["actual_sales"] = y_test.values
    results_df["predicted_sales"] = predictions
    results_df["prediction_error"] = results_df["actual_sales"] - results_df["predicted_sales"]

    return results_df