import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt


st.set_page_config(page_title="M5 Forecasting Dashboard", layout="wide")

st.title("M5 Retail Forecasting Dashboard")
st.write("End-to-End ML Pipeline Demo with the M5 dataset")


predictions_path = "data/processed/m5_predictions.csv"

if not os.path.exists(predictions_path):
    st.error("Prediction file not found. Please run pipeline.py first.")
else:
    df = pd.read_csv(predictions_path)

    st.subheader("Prediction Results")
    st.dataframe(df.head(20))

    if {"actual_sales", "predicted_sales"}.issubset(df.columns):
        st.subheader("Actual vs Predicted Sales")

        chart_df = df[["actual_sales", "predicted_sales"]].head(100).reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(chart_df.index, chart_df["actual_sales"], label="Actual Sales")
        ax.plot(chart_df.index, chart_df["predicted_sales"], label="Predicted Sales")
        ax.set_xlabel("Row Index")
        ax.set_ylabel("Sales")
        ax.set_title("Actual vs Predicted Sales")
        ax.legend()

        st.pyplot(fig)

    st.subheader("Basic Summary")
    st.write("Number of prediction rows:", len(df))
    st.write("Average actual sales:", round(df["actual_sales"].mean(), 2))
    st.write("Average predicted sales:", round(df["predicted_sales"].mean(), 2))