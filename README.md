# AI-Based Agricultural Demand and Supply Forecasting System

## Project Overview

Agriculture is one of the most important sectors contributing to food production and economic development. Farmers often face challenges in selecting suitable crops, predicting future crop yield, estimating market demand, and managing supply due to changing weather conditions and environmental factors.

This project presents an AI-Based Agricultural Demand and Supply Forecasting System that uses Machine Learning and Deep Learning techniques to predict crop yield, forecast future yield, estimate agricultural demand and supply, recommend suitable crops, and suggest alternative crops based on weather and agricultural conditions.

The system is developed using Python, Streamlit, XGBoost, and LSTM models and provides an interactive dashboard for users.

---

# Problem Statement

Traditional farming decisions are often based on experience and assumptions, which may lead to lower productivity and financial losses.

The objective of this project is to:

* Predict crop yield accurately.
* Forecast future crop yield trends.
* Predict agricultural demand.
* Predict agricultural supply.
* Recommend suitable crops.
* Suggest alternative crops when yield is low.
* Support farmers with data-driven decision making.

---

# Project Objectives

* Predict crop yield using Machine Learning.
* Analyze weather and agricultural factors affecting crop production.
* Predict agricultural demand.
* Predict agricultural supply.
* Generate 7-Day Yield Forecast.
* Generate 30-Day Yield Forecast.
* Recommend suitable crops.
* Suggest alternative crops.
* Compare Machine Learning model performance.
* Build an interactive dashboard using Streamlit.

---

# Dataset Information

## Agricultural Dataset

Contains crop production details such as:

* State
* District
* Crop
* Area
* Production
* Yield

## Weather Dataset

Contains weather information such as:

* Rainfall
* Average Temperature
* Minimum Temperature
* Maximum Temperature
* Wind Speed
* Air Pressure

## Enhanced Dataset

The final enhanced dataset contains:

* Agricultural production data
* Weather data
* Engineered features
* Lag features
* Rolling mean features
* Production growth
* Yield growth
* Temperature range
* Encoded categorical variables

---

# Data Preprocessing

The following preprocessing steps were performed:

## Data Cleaning

* Missing value handling
* Duplicate record removal
* Data type correction
* Null value treatment

## Feature Engineering

* Yield-related features
* Weather-based features
* Forecasting features
* Encoded categorical variables

### Additional Engineered Features

* Rainfall Lag
* Rainfall Rolling Mean
* Yield Lag Features
* Production Lag Features
* Production Growth
* Yield Growth
* Temperature Range

## Data Transformation

* Label Encoding
* Feature Scaling
* Data Standardization

---

# Technologies Used

## Programming Language

* Python

## Libraries

* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* TensorFlow
* Keras
* Joblib
* Streamlit
* Matplotlib
* Plotly

## Development Tools

* VS Code
* Git
* GitHub

---

# Machine Learning Models

## XGBoost

Used for Crop Yield Prediction.

Advantages:

* High prediction accuracy
* Faster training
* Better performance on structured datasets
* Reduced overfitting

## LSTM (Long Short-Term Memory)

Used for:

* 7-Day Yield Forecasting
* 30-Day Yield Forecasting

## Demand Prediction Model

Predicts future agricultural demand using machine learning.

## Supply Prediction Model

Predicts future agricultural supply using machine learning.

---

# Project Features

### 1. Crop Yield Prediction

Predicts crop yield based on:

* Crop
* State
* District
* Area
* Rainfall
* Average Temperature
* Minimum Temperature
* Maximum Temperature
* Wind Speed
* Air Pressure

### 2. Demand Prediction

Predicts future agricultural demand.

### 3. Supply Prediction

Predicts future agricultural supply.

### 4. Crop Recommendation

Provides suitable crop recommendations.

### 5. Alternative Crop Recommendation

Suggests alternative crops when expected yield is low.

### 6. 7-Day Yield Forecast

Forecasts crop yield for the next 7 days using LSTM.

### 7. 30-Day Yield Forecast

Forecasts crop yield for the next 30 days using LSTM.

### 8. Model Comparison

Compares XGBoost and LSTM model performance.

---

# Model Evaluation Metrics

The model performance is evaluated using:

* MAE (Mean Absolute Error)
* MSE (Mean Squared Error)
* RMSE (Root Mean Squared Error)
* R² Score

---

# Project Structure

```text
agri_project/
│── app.py
│── requirements.txt
│── README.md
│── final_dataset_enhanced.csv
│── crop_recommendation_table.csv
│── alternative_crop.csv
│── crop_comparison_result.csv
│── demand_model.pkl
│── supply_model.pkl
│── xgb_yield_model.pkl
│── lstm_yield_model.keras
│── scaler_x.pkl
│── scaler_y.pkl
│── le_crop.pkl
│── le_state.pkl
│── le_district.pkl
```

---

# How to Run The Project

## Step 1

Clone the repository.

## Step 2

Install dependencies.

```bash
pip install -r requirements.txt
```

## Step 3

Run the Streamlit application.

```bash
streamlit run app.py
```

---

# Expected Outputs

* Crop Yield Prediction
* Demand Prediction
* Supply Prediction
* Crop Recommendation
* Alternative Crop Recommendation
* 7-Day Yield Forecast
* 30-Day Yield Forecast
* Model Comparison Results

---

# Future Enhancements

* Real-Time Weather API Integration
* Soil Nutrient Analysis
* Crop Price Prediction
* Mobile Application Development
* Satellite Data Integration
* Live Demand and Supply Analytics

---

# Project Outcome

The system helps farmers and agricultural stakeholders make informed decisions by predicting crop yield, demand, supply, future yield trends, and recommending suitable crops using Artificial Intelligence and Machine Learning techniques.
