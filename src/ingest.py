import os
import pandas as pd


ID_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def load_m5_data(sales_path, calendar_path, prices_path):
    """Load the M5 dataset files."""
    if not os.path.exists(sales_path):
        raise FileNotFoundError(f"{sales_path} not found")

    if not os.path.exists(calendar_path):
        raise FileNotFoundError(f"{calendar_path} not found")

    if not os.path.exists(prices_path):
        raise FileNotFoundError(f"{prices_path} not found")

    sales_df = pd.read_csv(sales_path)
    calendar_df = pd.read_csv(calendar_path)
    prices_df = pd.read_csv(prices_path)

    return sales_df, calendar_df, prices_df


def sample_sales_rows(sales_df, n_rows=200):
    """Take a smaller sample of product-store rows for faster development."""
    n_rows = min(n_rows, len(sales_df))
    return sales_df.sample(n=n_rows, random_state=42).copy()


def reshape_sales_to_long(sales_df):
    """Convert wide daily columns (d_1, d_2, ...) into long format."""
    day_columns = [col for col in sales_df.columns if col.startswith("d_")]

    long_df = sales_df.melt(
        id_vars=ID_COLUMNS,
        value_vars=day_columns,
        var_name="d",
        value_name="sales"
    )

    return long_df


def save_dataframe(df, output_path):
    """Save a DataFrame to CSV, creating folders if needed."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)