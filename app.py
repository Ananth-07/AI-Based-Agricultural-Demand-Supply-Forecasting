import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="AI-Based Agricultural Demand and Supply Forecasting",
    layout="wide"
)

st.title("🌾 AI-Based Agricultural Demand and Supply Forecasting System")

# =====================================================
# LOAD MODELS
# =====================================================
@st.cache_resource
def load_models():

    xgb_model = joblib.load(
        r"D:\xgb_yield_model.pkl"
    )

    lstm_model = load_model(
        r"D:\lstm_yield_model.keras"
    )

    scaler_x = joblib.load(
        r"D:\scaler_x.pkl"
    )

    scaler_y = joblib.load(
        r"D:\scaler_y.pkl"
    )

    le_state = joblib.load(
        r"D:\le_state.pkl"
    )

    le_district = joblib.load(
        r"D:\le_district.pkl"
    )

    le_crop = joblib.load(
        r"D:\le_crop.pkl"
    )

    return (
        xgb_model,
        lstm_model,
        scaler_x,
        scaler_y,
        le_state,
        le_district,
        le_crop
    )


# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():

    final_dataset = pd.read_csv(
        r"D:\final_dataset.csv"
    )

    recommendation_df = pd.read_csv(
        r"D:\recommendation_df.csv"
    )

    alternative_crop = pd.read_csv(
        r"D:\alternative_crop.csv"
    )

    comparison_df = pd.read_csv(
        r"D:\crop_comparison_result.csv"
    )

    merged_df = pd.read_csv(
        r"D:\merged_feature_engineered_data1.csv"
    )

    return (
        final_dataset,
        recommendation_df,
        alternative_crop,
        comparison_df,
        merged_df
    )


# =====================================================
# CALL FUNCTIONS
# =====================================================
(
    xgb_model,
    lstm_model,
    scaler_x,
    scaler_y,
    le_state,
    le_district,
    le_crop
) = load_models()

(
    final_dataset,
    recommendation_df,
    alternative_crop,
    comparison_df,
    merged_df
) = load_data()

# =====================================================
# SESSION STATE
# =====================================================
if "crop" not in st.session_state:
    st.session_state["crop"] = None

if "predicted_yield" not in st.session_state:
    st.session_state["predicted_yield"] = None

if "recommended_crop" not in st.session_state:
    st.session_state["recommended_crop"] = None

if "forecast_crop" not in st.session_state:
    st.session_state["forecast_crop"] = None   

if "area" not in st.session_state:
    st.session_state["area"] = None

if "rainfall" not in st.session_state:
    st.session_state["rainfall"] = None

if "avg_temp" not in st.session_state:
    st.session_state["avg_temp"] = None

if "min_temp" not in st.session_state:
    st.session_state["min_temp"] = None

if "max_temp" not in st.session_state:
    st.session_state["max_temp"] = None

if "wind_speed" not in st.session_state:
    st.session_state["wind_speed"] = None

if "air_pressure" not in st.session_state:
    st.session_state["air_pressure"] = None


# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("Navigation")

page = st.sidebar.selectbox(
    "Select Menu",
    [
        "Home",
        "Dataset",
        "Yield Prediction",
        "Crop Recommendation",
        "Alternative Crop",
        "7 Day Forecast",
        "30 Day Forecast",
        "Model Comparison"
    ]
)


# =====================================================
# HOME PAGE
# =====================================================
if page == "Home":

    st.header(
        "AI-Based Agricultural Demand and Supply Forecasting System"
    )

    st.write("""
This project predicts crop yield using XGBoost and LSTM models.

### Features

✅ Yield Prediction

✅ Crop Recommendation

✅ Alternative Crop Suggestion

✅ Model Comparison

✅ 7 Day Forecast

✅ 30 Day Forecast

✅ Demand and Supply Analysis

✅ Download Forecast Report
""")

    st.subheader("Dataset Used")

    st.write(
        """
1. Crop Production Dataset

2. Weather Dataset

3. Market Price Dataset

4. Feature Engineered Dataset
        """
    )


# =====================================================
# DATASET PAGE
# =====================================================
elif page == "Dataset":

    st.header("Final Dataset")

    col1, col2 = st.columns(2)

    col1.metric(
        "Rows",
        final_dataset.shape[0]
    )

    col2.metric(
        "Columns",
        final_dataset.shape[1]
    )

    st.subheader("Dataset Preview")

    st.dataframe(
        final_dataset.head(100)
    )

    st.subheader("Columns")

    st.write(
        final_dataset.columns.tolist()
    )

    # =====================================================
# YIELD PREDICTION
# =====================================================
elif page == "Yield Prediction":

    st.header("🌾 Yield Prediction")

    # --------------------------
    # State
    # --------------------------
    state = st.selectbox(
        "Select State",
        sorted(
            merged_df["State"].unique()
        )
    )

    # --------------------------
    # District
    # --------------------------
    district = st.selectbox(
        "Select District",
        sorted(
            merged_df[
                merged_df["State"] == state
            ]["District"].unique()
        )
    )

    # --------------------------
    # Crop
    # --------------------------
    crop = st.selectbox(
        "Select Crop",
        sorted(
            merged_df[
                (merged_df["State"] == state)
                &
                (merged_df["District"] == district)
            ]["Crop"].unique()
        )
    )

    # --------------------------
    # User Inputs
    # --------------------------
    crop_year = st.number_input(
        "Crop Year",
        value=2025
    )

    area = st.number_input(
        "Area",
        value=1000.0
    )

    rainfall = st.number_input(
        "Rainfall",
        value=1000.0
    )

    avg_temp = st.number_input(
        "Average Temperature",
        value=25.0
    )

    min_temp = st.number_input(
        "Minimum Temperature",
        value=20.0
    )

    max_temp = st.number_input(
        "Maximum Temperature",
        value=32.0
    )

    wind_speed = st.number_input(
        "Wind Speed",
        value=8.0
    )

    air_pressure = st.number_input(
        "Air Pressure",
        value=1010.0
    )

    # =====================================================
    # PREDICT BUTTON
    # =====================================================
    if st.button("Predict Yield"):

        # --------------------------------
        # Encode categorical columns
        # --------------------------------
        state_enc = le_state.transform(
            [state]
        )[0]

        district_enc = le_district.transform(
            [district]
        )[0]

        crop_enc = le_crop.transform(
            [crop]
        )[0]

        # --------------------------------
        # Historical Crop Data
        # --------------------------------
        crop_hist = merged_df[
            (merged_df["State"] == state)
            &
            (merged_df["District"] == district)
            &
            (merged_df["Crop"] == crop)
        ].sort_values(
            "Crop_Year"
        )

        # --------------------------------
        # Production
        # --------------------------------
        if len(crop_hist) > 0:

            production = crop_hist[
                "Production"
            ].mean()

        else:

            production = 0

        # --------------------------------
        # Lag Features
        # --------------------------------
        if len(crop_hist) >= 3:

            yield_lag1 = crop_hist[
                "Yield"
            ].iloc[-1]

            yield_lag3 = crop_hist[
                "Yield"
            ].iloc[-3]

            production_lag1 = crop_hist[
                "Production"
            ].iloc[-1]

            production_lag3 = crop_hist[
                "Production"
            ].iloc[-3]

        else:

            yield_lag1 = 0
            yield_lag3 = 0
            production_lag1 = production
            production_lag3 = production

        # --------------------------------
        # Derived Features
        # --------------------------------
        temp_range = (
            max_temp - min_temp
        )

        production_growth = 0

        yield_growth = 0

        rainfall_lag1 = rainfall

        rainfall_roll3 = rainfall

        # --------------------------------
        # Create Input Data
        # --------------------------------
        input_data = pd.DataFrame([{

            "State": state_enc,
            "District": district_enc,
            "Crop_Year": crop_year,
            "Crop": crop_enc,
            "Area": area,
            "Production": production,
            "rainfall": rainfall,
            "avg_temp": avg_temp,
            "min_temp": min_temp,
            "max_temp": max_temp,
            "wind_speed": wind_speed,
            "air_pressure": air_pressure,
            "temp_range": temp_range,
            "production_growth": production_growth,
            "yield_growth": yield_growth,
            "rainfall_lag1": rainfall_lag1,
            "rainfall_roll3": rainfall_roll3,
            "yield_lag1": yield_lag1,
            "yield_lag3": yield_lag3,
            "production_lag1": production_lag1,
            "production_lag3": production_lag3

        }])

        # --------------------------------
        # Feature Order
        # --------------------------------
        input_data = input_data[
            xgb_model.feature_names_in_
        ]

        # --------------------------------
        # Prediction
        # --------------------------------
        prediction = xgb_model.predict(input_data)[0]

        if prediction <= 0:

            crop_data = merged_df[
                (merged_df["Crop"] == crop)
    ]

            if len(crop_data) > 0:
                prediction = max(crop_data["Yield"].median(),0.1)
            else:
                prediction = 0.1

        prediction = round(prediction, 2)
        # Save values for all pages
        st.session_state["crop"] = crop
        st.session_state["predicted_yield"] = round(prediction, 2)

        st.session_state["area"] = area
        st.session_state["rainfall"] = rainfall
        st.session_state["avg_temp"] = avg_temp
        st.session_state["min_temp"] = min_temp
        st.session_state["max_temp"] = max_temp
        st.session_state["wind_speed"] = wind_speed
        st.session_state["air_pressure"] = air_pressure
        # --------------------------------
        # Output
        # --------------------------------
        st.success(
            f"Predicted Yield = {prediction:.2f}Ton/Hectare"
        )

        st.metric(
            "Predicted Yield",
            f"{prediction:.2f}Ton/Hectare",
        )

# =====================================================
# CROP CATEGORY
# =====================================================
crop_category = {

    "Cereals": [
        "Rice", "Wheat", "Maize","Bajra","Ragi", "Jowar","Barley","Small millets","Other Cereals"
    ],

    "Vegetables": [
         "Onion","Potato","Sweet potato",
        "Garlic","Dry chillies","Tapioca"
    ],

    "Fruits": [
        "Banana", "Papaya", "Mango", "Guava"
    ],

    "Plantation": [
       "Arecanut","Coconut","Cashewnut",
        "Cardamom","Coffee"
    ],

    "Commercial": [
        "Sugarcane","Cotton(lint)",
        "Jute","Tobacco","Mesta"
    ],
    
    "Pulses":["Arhar/Tur","Urad","Gram","Horse-gram",
        "Cowpea(Lobia)","Moong(Green Gram)",
        "Masoor","Khesari","Moth",
        "Peas & beans (Pulses)",
        "Other Kharif pulses",
        "Other Rabi pulses",
        "Other Summer Pulses"

    ],

    "Oil seeds":["Groundnut","Sesamum","Sunflower",
        "Rapeseed &Mustard","Castor seed",
        "Linseed","Safflower","Soyabean",
        "Niger seed","other oilseeds",
        "Oilseeds total"
    ],
     
     "Spices": [
        "Black pepper","Ginger",
        "Turmeric","Coriander"
    ],

     "Others": [
        "Guar seed","Sannhamp"
    ]

}


# =====================================================
# CROP RECOMMENDATION
# =====================================================
if page == "Crop Recommendation":

    st.header("🌱 Crop Recommendation")

    # Get crop from Yield Prediction page
crop = st.session_state["crop"]

if crop is None:
    st.warning("Please run Yield Prediction first.")
    st.stop()

# Find category automatically
category = None

for key, value in crop_category.items():
    if crop in value:
        category = key
        break

st.success(f"Selected Crop : {crop}")
st.info(f"Category : {category}")
if category is None:
    st.error("Category not found for selected crop.")
    st.stop()

    category_crops = crop_category[category]

    temp_df = recommendation_df[
        recommendation_df["Crop"].isin(
            category_crops
        )
    ].copy()

    # Coconut dominance fix
    temp_df["Yield_log"] = np.log1p(
        temp_df["Yield"]
    )

    temp_df["Yield_norm"] = (
        temp_df["Yield_log"]
        /
        temp_df["Yield_log"].max()
    )

    temp_df["Demand_norm"] = (
        temp_df["Demand_Score"]
        /
        temp_df["Demand_Score"].max()
    )

    temp_df["Supply_norm"] = (
        temp_df["Supply_Score"]
        /
        temp_df["Supply_Score"].max()
    )

    temp_df["Final_Score"] = (

        0.10 * temp_df["Yield_norm"]

        +

        0.45 * temp_df["Demand_norm"]

        +

        0.45 * temp_df["Supply_norm"]

    )

    temp_df = temp_df.sort_values(
        "Final_Score",
        ascending=False
    )
    top_crop = temp_df.iloc[0]["Crop"]

# avoid same crop recommendation
    if top_crop == crop and len(temp_df) > 1:
        top_crop = temp_df.iloc[1]["Crop"]

    st.session_state["recommended_crop"] = top_crop
    st.session_state["recommended_crop"] = top_crop

    st.success(
    f"Recommended Crop : {top_crop}"
)

    st.subheader(
        "Top Recommended Crops"
    )

    st.dataframe(

        temp_df[
            [
                "Crop",
                "Yield",
                "Demand_Score",
                "Supply_Score",
                "Final_Score"
            ]
        ]

    )


# =====================================================
# ALTERNATIVE CROP
# =====================================================
elif page == "Alternative Crop":

    st.header(
        "🌾 Alternative Crop Recommendation"
    )

    current_crop = st.session_state.get(
    "recommended_crop"
    )

    if current_crop is None:
        current_crop = st.session_state.get(
            "crop"
        )

    if current_crop is None:
        current_crop = sorted(
            recommendation_df["Crop"].unique()
        )[0]

    st.success(
        f"Current Crop : {current_crop}"
    )

    selected_category = None

    for cat, crops in crop_category.items():

        if current_crop.lower() in [x.lower() for x in crops]:

            selected_category = cat

            break
    if selected_category is None:
        selected_category = list(
            crop_category.keys()
        )[0]   

    category_df = recommendation_df[
        recommendation_df["Crop"].isin(
            crop_category[selected_category]
        )
    ].copy()

    category_df = category_df[
        category_df["Crop"] != current_crop
    ]

    category_df["Yield_log"] = np.log1p(
        category_df["Yield"]
    )

    category_df["Yield_norm"] = (
        category_df["Yield_log"]
        /
        category_df["Yield_log"].max()
    )

    category_df["Demand_norm"] = (
        category_df["Demand_Score"]
        /
        category_df["Demand_Score"].max()
    )

    category_df["Supply_norm"] = (
        category_df["Supply_Score"]
        /
        category_df["Supply_Score"].max()
    )

    category_df["Final_Score"] = (

        0.10 * category_df["Yield_norm"]

        +

        0.45 * category_df["Demand_norm"]

        +

        0.45 * category_df["Supply_norm"]

    )

    category_df = category_df.sort_values(
        "Final_Score",
        ascending=False
    )
    if len(category_df) == 0:
        category_df = recommendation_df.copy()
    
    best_crop = category_df.iloc[0]["Crop"]

    if best_crop == current_crop and len(category_df) > 1:
        best_crop = category_df.iloc[1]["Crop"]

    st.session_state["forecast_crop"] = best_crop

    st.success(
       f"Recommended Alternative Crop : {best_crop}"
)

    st.subheader(
        "Top Alternative Crops"
    )

    st.dataframe(

        category_df[
            [
                "Crop",
                "Yield",
                "Demand_Score",
                "Supply_Score",
                "Final_Score"
            ]
        ].head(5)

    )

   


# =====================================================
# MODEL COMPARISON
# =====================================================
elif page == "Model Comparison":

    st.header(
        "📊 Model Comparison"
    )

    st.dataframe(
        comparison_df
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(

        "Best MAE",

        round(
            comparison_df["MAE"].min(),
            2
        )

    )

    col2.metric(

        "Best RMSE",

        round(
            comparison_df["RMSE"].min(),
            2
        )

    )

    col3.metric(

        "Best R² Score",

        round(
            comparison_df["R2 Score"].max(),
            3
        )

    )

    st.subheader(
        "R² Score Comparison"
    )

    fig, ax = plt.subplots(
        figsize=(8,5)
    )

    ax.bar(

        comparison_df["Model"],

        comparison_df["R2 Score"]

    )

    ax.set_ylabel(
        "R² Score"
    )

    st.pyplot(fig)
    # =====================================================
# FORECAST FUNCTION
# =====================================================
def forecast_future(crop_df, days):

    crop_df = crop_df.sort_values(
        "Crop_Year"
    ).copy()

    # Encode categorical columns
    crop_df["State"] = le_state.transform(
        crop_df["State"]
    )

    crop_df["District"] = le_district.transform(
        crop_df["District"]
    )

    crop_df["Crop"] = le_crop.transform(
        crop_df["Crop"]
    )

    # Missing lag features
    crop_df["yield_lag1"] = (
        crop_df["Yield"].shift(1)
    )

    crop_df["yield_lag3"] = (
        crop_df["Yield"]
        .rolling(3)
        .mean()
    )

    crop_df["production_lag1"] = (
        crop_df["Production"]
        .shift(1)
    )

    crop_df["production_lag3"] = (
        crop_df["Production"]
        .rolling(3)
        .mean()
    )

    crop_df.fillna(
        method="bfill",
        inplace=True
    )

    feature_cols = [

        'State',
        'District',
        'Crop_Year',
        'Crop',
        'Area',
        'Production',
        'rainfall',
        'avg_temp',
        'min_temp',
        'max_temp',
        'wind_speed',
        'air_pressure',
        'temp_range',
        'production_growth',
        'yield_growth',
        'rainfall_lag1',
        'rainfall_roll3',
        'yield_lag1',
        'yield_lag3',
        'production_lag1',
        'production_lag3'

    ]

    X_group = scaler_x.transform(
        crop_df[feature_cols]
    )

    current_seq = X_group[-3:]

    predictions = []

    for i in range(days):

        pred_scaled = lstm_model.predict(
            current_seq.reshape(
                1,
                3,
                21
            ),
            verbose=0
        )

        pred = scaler_y.inverse_transform(
            pred_scaled
        )[0][0]

        predictions.append(pred)

        next_row = current_seq[-1].copy()

        current_seq = np.vstack(
            [
                current_seq[1:],
                next_row
            ]
        )

    return predictions


# =====================================================
# 7 DAY FORECAST
# =====================================================
if page == "7 Day Forecast":

    st.header(
        "📈 7 Day Forecast"
    )

    forecast_crop = st.session_state.get(
        "forecast_crop"
    )

    if forecast_crop is None:
        forecast_crop = st.session_state.get(
            "recommended_crop"
        )

    if forecast_crop is None:
        forecast_crop = st.session_state.get(
            "crop"
        )

    if forecast_crop is None:
        forecast_crop = sorted(
            merged_df["Crop"].unique()
        )[0]

    st.success(
        f"Forecast Crop : {forecast_crop}"
    )

    crop_df = merged_df[
        merged_df["Crop"].astype(str)
        .str.strip()
        .str.lower()
        ==
        forecast_crop.strip().lower()
    ]

# Safety check
    if crop_df.empty:

        st.warning(
            "Crop data not found. Using entire dataset."
        )

        crop_df = merged_df.copy()

    st.write("Forecast Crop :", forecast_crop)
    st.write("Crop Data Shape :", crop_df.shape)

    pred_7 = forecast_future(
        crop_df,
        7
    )

    forecast_df = pd.DataFrame({
        "Day": range(1, 8),
        "Predicted Yield": pred_7
    })

    st.dataframe(
        forecast_df
    )

    st.line_chart(
        forecast_df.set_index(
            "Day"
        )
    )

    st.download_button(
        "Download CSV",
        forecast_df.to_csv(index=False),
        "7_day_forecast.csv"
    )


# =====================================================
# 30 DAY FORECAST
# =====================================================
elif page == "30 Day Forecast":

    st.header(
        "📈 30 Day Forecast"
    )

    forecast_crop = st.session_state.get(
        "forecast_crop"
    )

    if forecast_crop is None:
        forecast_crop = st.session_state.get(
            "recommended_crop"
        )

    if forecast_crop is None:
        forecast_crop = st.session_state.get(
            "crop"
        )

    if forecast_crop is None:
        forecast_crop = sorted(
            merged_df["Crop"].unique()
        )[0]

    st.success(
        f"Forecast Crop : {forecast_crop}"
    )

    crop_df = merged_df[
        merged_df["Crop"].astype(str)
        .str.strip()
        .str.lower()
        ==
        forecast_crop.strip().lower()
    ]

# Safety check
    if crop_df.empty:

        st.warning(
            "Crop data not found. Using entire dataset."
        )

        crop_df = merged_df.copy()
    st.write("Forecast Crop :", forecast_crop)
    st.write("Crop Data Shape :", crop_df.shape)    

    pred_30 = forecast_future(
        crop_df,
        30
    )

    forecast_df = pd.DataFrame({
        "Day": range(1, 31),
        "Predicted Yield": pred_30
    })

    st.dataframe(
        forecast_df
    )

    st.line_chart(
        forecast_df.set_index(
            "Day"
        )
    )

    st.download_button(
        "Download CSV",
        forecast_df.to_csv(index=False),
        "30_day_forecast.csv"
    )

    