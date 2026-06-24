AI-Based Agricultural Demand and Supply Forecasting System

Project Overview

Agriculture is one of the most important sectors contributing to food production and economic development. Farmers often face challenges in selecting suitable crops and predicting future crop yield due to changing weather conditions and environmental factors.

This project presents an AI-Based Agricultural Demand and Supply Forecasting System that uses Machine Learning techniques to predict crop yield, generate future yield forecasts, recommend suitable crops, and suggest alternative crops based on weather and agricultural conditions.

The system is developed using Python, Streamlit, XGBoost, and LSTM models and provides an interactive dashboard for users.

---

Problem Statement

Traditional farming decisions are often based on experience and assumptions, which may lead to lower productivity and financial losses.

The objective of this project is to:

- Predict crop yield accurately.
- Forecast future crop yield trends.
- Recommend suitable crops.
- Suggest alternative crops when yield is low.
- Support farmers with data-driven decision making.

---

Project Objectives

- Predict crop yield using Machine Learning.
- Analyze weather and agricultural factors affecting crop production.
- Generate 7-Day Yield Forecast.
- Generate 30-Day Yield Forecast.
- Recommend suitable crops.
- Suggest alternative crops when yield is low.
- Compare machine learning model performance.
- Build an interactive dashboard using Streamlit.

---

Dataset Information

Agricultural Dataset

Contains crop production details such as:

- State
- District
- Crop
- Area
- Production
- Yield

Weather Dataset

Contains weather information such as:

- Rainfall
- Average Temperature
- Minimum Temperature
- Maximum Temperature
- Wind Speed
- Air Pressure

---

Data Preprocessing

The following preprocessing steps were performed:

Data Cleaning

- Missing value handling
- Duplicate record removal
- Data type correction
- Null value treatment

Feature Engineering

- Yield-related features
- Weather-based features
- Forecasting features
- Encoded categorical variables

Data Transformation

- Label Encoding
- Feature Scaling
- Data Standardization

---

Technologies Used

Programming Language

- Python

Libraries

- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- TensorFlow
- Joblib
- Streamlit

Development Tools

- VS Code
- Git
- GitHub

---

Machine Learning Models

XGBoost

Used for Crop Yield Prediction.

Advantages:

- High prediction accuracy
- Faster training
- Better performance on structured datasets
- Reduced overfitting

LSTM (Long Short-Term Memory)

Used for:

- 7-Day Yield Forecasting
- 30-Day Yield Forecasting

LSTM was selected because it can learn long-term dependencies and historical patterns from time-series agricultural data.

---

Project Features

1. Crop Yield Prediction

Predicts crop yield based on:

- Crop
- State
- District
- Area
- Rainfall
- Average Temperature
- Minimum Temperature
- Maximum Temperature
- Wind Speed
- Air Pressure

2. Crop Recommendation

Provides suitable crop recommendations based on environmental conditions and agricultural factors.

3. Alternative Crop Recommendation

Suggests alternative crops when the selected crop is expected to provide lower yield.

4. 7-Day Yield Forecasting

Uses LSTM to forecast crop yield for the next 7 days.

5. 30-Day Yield Forecasting

Uses LSTM to forecast crop yield for the next 30 days.

6. Model Comparison

Compares machine learning models using evaluation metrics.

---

Model Evaluation Metrics

The model performance is evaluated using:

MAE (Mean Absolute Error)

Measures average prediction error.

MSE (Mean Squared Error)

Measures squared prediction error.

RMSE (Root Mean Squared Error)

Measures overall prediction accuracy.

R² Score

Measures goodness of fit.

---

Project Structure

agri_project/

- app.py
- requirements.txt
- final_dataset.csv
- recommendation_df.csv
- alternative_crop.csv
- crop_comparison_result.csv
- xgb_yield_model.pkl
- lstm_yield_model.keras
- scaler_x.pkl
- scaler_y.pkl
- le_crop.pkl
- le_state.pkl
- le_district.pkl
- README.md

---

How To Run The Project

Step 1

Clone the repository

Step 2

Install dependencies

pip install -r requirements.txt

Step 3

Run the Streamlit application

streamlit run app.py

---

Expected Outputs

Yield Prediction

Displays predicted crop yield in Ton/Hectare.

Crop Recommendation

Displays recommended crop suggestions.

Alternative Crop Recommendation

Displays alternative crops with better expected performance.

7-Day Forecast

Displays forecasted yield trends for the next seven days.

30-Day Forecast

Displays forecasted yield trends for the next thirty days.

Model Comparison

Displays model evaluation metrics and comparison results.

---

Future Enhancements

- Real-Time Weather API Integration
- Soil Nutrient Analysis
- Crop Price Prediction
- Mobile Application Development
- Satellite Data Integration

---

Project Outcome

The system helps farmers and agricultural stakeholders make informed decisions by predicting crop yield, forecasting future yield trends, and recommending suitable crops using Artificial Intelligence and Machine Learning techniques.

