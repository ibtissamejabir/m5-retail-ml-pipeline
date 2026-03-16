# M5 Retail Forecasting: End-to-End ML Pipeline

This project implements an end-to-end machine learning pipeline for retail demand forecasting using the **M5 Forecasting dataset**. The system demonstrates the full lifecycle of an ML workflow: data ingestion, preprocessing, feature engineering, model training, prediction generation, visualization, and containerized deployment.

---

## Project Overview

Retail demand forecasting helps businesses predict how many items will sell in the future so they can optimize inventory, logistics, and supply chain planning.

This project builds a structured ML pipeline that processes historical retail data and generates sales predictions using machine learning.

The pipeline includes:

- Data ingestion from multiple sources
- Data preprocessing and merging
- Time-series feature engineering
- Machine learning model training
- Prediction generation
- Visualization using Streamlit
- Containerization with Docker

---

## Pipeline Architecture

The workflow of the system is:

Raw Data  
↓  
Data Ingestion  
↓  
Data Preprocessing  
↓  
Feature Engineering  
↓  
Model Training  
↓  
Prediction Generation  
↓  
Streamlit Dashboard  
↓  
Docker Deployment

---

## Tech Stack

- Python  
- Pandas / NumPy  
- Scikit-Learn  
- Streamlit  
- Docker  
- Git / GitHub  

---

## Project Structure

```
m5-retail-ml-pipeline/

src/
    ingest.py
    preprocess.py
    features.py
    train.py
    predict.py

pipeline.py
streamlit_app.py
Dockerfile
requirements.txt
```

---

## Key Components

### Data Pipeline
The pipeline loads and merges multiple datasets from the M5 competition:

- Sales data
- Calendar data
- Price data

These datasets are combined into a structured dataset suitable for machine learning.

### Feature Engineering

The system creates time-series features including:

- Day of week
- Month
- Year
- Weekend indicator
- Lag features (sales from previous days)
- Rolling averages

These features allow the model to capture temporal demand patterns.

### Model Training

A **Random Forest Regressor** is trained to predict sales.

Evaluation metrics include:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

### Prediction Generation

The trained model generates sales predictions which are stored and later visualized in the dashboard.

### Streamlit Dashboard

The dashboard displays:

- Prediction results
- Actual vs predicted sales
- Summary statistics

This provides a simple way to explore model performance.

### Docker Deployment

The project is containerized with Docker so the entire application can run in a reproducible environment.

---

## Running the Project

### Run the Pipeline

```bash
python pipeline.py
```

### Run the Dashboard

```bash
streamlit run streamlit_app.py
```

Then open:

```
http://localhost:8501
```

---

### Run with Docker

Build the container:

```bash
docker build -t m5-retail-dashboard .
```

Run the container:

```bash
docker run -p 8501:8501 m5-retail-dashboard
```

Then open:

```
http://localhost:8501
```

---

## Dataset

This project uses the **M5 Forecasting Accuracy Dataset** from Kaggle.

https://www.kaggle.com/competitions/m5-forecasting-accuracy

Due to GitHub file size limits, the dataset is not included in the repository.

---

## Future Improvements

Possible future improvements include:

- Using LightGBM or XGBoost for better forecasting performance
- Adding future demand forecasting
- Deploying the dashboard to the cloud
- Adding experiment tracking

---

## Author

This project was built to practice building structured machine learning pipelines and deploying ML systems beyond notebook experimentation.
