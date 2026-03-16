from src.ingest import (
    load_m5_data,
    sample_sales_rows,
    reshape_sales_to_long,
    save_dataframe,
)
from src.preprocess import preprocess_m5_data
from src.features import build_features
from src.train import (
    prepare_training_data,
    split_train_test,
    train_model,
    evaluate_model,
    save_model,
)
from src.predict import (
    load_saved_model,
    make_predictions,
    create_prediction_dataframe,
)


def main():
    sales_path = "data/raw/sales_train_validation.csv"
    calendar_path = "data/raw/calendar.csv"
    prices_path = "data/raw/sell_prices.csv"

    sales_df, calendar_df, prices_df = load_m5_data(
        sales_path,
        calendar_path,
        prices_path,
    )

    print("Original sales shape:", sales_df.shape)

    sampled_sales_df = sample_sales_rows(sales_df, n_rows=200)
    print("Sampled sales shape:", sampled_sales_df.shape)

    long_sales_df = reshape_sales_to_long(sampled_sales_df)
    print("Long sales shape:", long_sales_df.shape)

    merged_df = preprocess_m5_data(long_sales_df, calendar_df, prices_df)
    print("Merged data shape:", merged_df.shape)

    feature_df = build_features(merged_df)
    print("Feature data shape:", feature_df.shape)

    training_df = prepare_training_data(feature_df)
    print("Training data shape after dropping NaNs:", training_df.shape)

    X_train, X_test, y_train, y_test = split_train_test(training_df)
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    print("\nModel evaluation:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    save_model(model)

    loaded_model = load_saved_model()
    predictions = make_predictions(loaded_model, X_test)

    prediction_df = create_prediction_dataframe(X_test, y_test, predictions)

    save_dataframe(training_df, "data/processed/m5_training_sample.csv")
    save_dataframe(prediction_df, "data/processed/m5_predictions.csv")

    print("\nPrediction preview:")
    print(prediction_df.head())

    print("\nStep 6 completed successfully.")
    print("Saved model: models/random_forest_model.pkl")
    print("Saved training data: data/processed/m5_training_sample.csv")
    print("Saved predictions: data/processed/m5_predictions.csv")


if __name__ == "__main__":
    main()