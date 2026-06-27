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
# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
<style>

/* ==========================
   APP
========================== */

.stApp{
    background:#f4f8f5;
    color:#1f2937;
}

.block-container{
    padding-top:1rem;
    padding-bottom:2rem;
}

/* ==========================
   SIDEBAR
========================== */

section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#14532d,#166534);
}

section[data-testid="stSidebar"] *{
    color:white;
}
/* Sidebar Title */

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3{
    color:white !important;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label{
    color:white !important;
}

section[data-testid="stSidebar"]{
    border-right: none !important;
    box-shadow: none !important;
}

section[data-testid="stSidebar"] > div{
    border-right: none !important;
    box-shadow: none !important;
}  
[data-testid="stSidebar"]{
    border: none !important;
}

[data-testid="stSidebar"] > div:first-child{
    border-right: none !important;
}            
                                  

/* ==========================
   HEADINGS
========================== */

h1,h2,h3,h4,h5,h6{
    color:#14532d !important;
    font-weight:700;
}

p,label{
    color:#1f2937 !important;
}

/* ==========================
   METRICS
========================== */

div[data-testid="stMetric"]{
    background:white;
    border-radius:15px;
    padding:18px;
    border-left:6px solid #16a34a;
    box-shadow:0 4px 10px rgba(0,0,0,.12);
}

div[data-testid="stMetricValue"]{
    color:#14532d !important;
    font-size:28px;
    font-weight:bold;
}

div[data-testid="stMetricLabel"]{
    color:#374151 !important;
}

/* ==========================
   BUTTON
========================== */

.stButton button,
.stDownloadButton button{

    background:#16a34a !important;
    color:white !important;

    border:none;
    border-radius:10px;
    font-weight:bold;
}

.stButton button:hover,
.stDownloadButton button:hover{

    background:#15803d !important;
}

/* ==========================
   SELECTBOX FIX
========================== */

div[data-baseweb="select"]{

    background:white !important;

    border-radius:10px;

}

/* selected value */

div[data-baseweb="select"] span{

    color:#111827 !important;

    -webkit-text-fill-color:#111827 !important;

}

div[data-baseweb="select"] input{

    color:#111827 !important;

    -webkit-text-fill-color:#111827 !important;

    caret-color: transparent !important;

}

/* arrow */

div[data-baseweb="select"] svg{

    color:#111827 !important;

    fill:#111827 !important;

}

/* dropdown */

ul[role="listbox"]{

    background:white !important;

}

ul[role="listbox"] li{

    color:#111827 !important;

    background:white !important;

}

ul[role="listbox"] li:hover{

    background:#dbeafe !important;

}

/* ==========================
   INPUT
========================== */

input{

    color:#111827 !important;

    background:white !important;

}
            
div[data-baseweb="select"] input{
    caret-color: transparent !important;
}            

textarea{

    color:#111827 !important;

}

/* Number Input Style */

div[data-testid="stNumberInput"] input{
    background:#1f2937 !important;
    color:white !important;
    -webkit-text-fill-color:white !important;
    border:1px solid #374151 !important;
}

div[data-testid="stNumberInput"] button{
    background:#1f2937 !important;
    color:white !important;
}
                          
/* ==========================
   DATAFRAME
========================== */

[data-testid="stDataFrame"]{
    border-radius:12px;
    overflow:hidden;
}

/* Table */

table{
    color:#111827 !important;
}

/* Header */

thead th{

    background:#14532d !important;

    color:white !important;

    text-align:left !important;

}

/* Body */

tbody td{

    color:#111827 !important;

    background:white !important;

    text-align:left !important;

}

/* ==========================
   CODE BLOCK
========================== */

pre{

    color:white !important;

}

code{

    color:white !important;

}

/* ==========================
   ALERTS
========================== */

div[data-testid="stAlert"]{

    border-radius:12px;

}

/* ==========================
   MARKDOWN
========================== */

div[data-testid="stMarkdownContainer"]{

    color:#111827 !important;

}

/* ==========================
   STREAMLIT TOOLBAR HIDE
========================== */

#MainMenu{
    visibility:hidden;
}

header{
    visibility:hidden;
}

footer{
    visibility:hidden;
}

[data-testid="stToolbar"]{
    display:none !important;
}

[data-testid="stDecoration"]{
    display:none !important;
}

[data-testid="stStatusWidget"]{
    display:none !important;
}

/* ==========================
   FIX SELECTED VALUE
========================== */

.stSelectbox div[data-baseweb="select"] > div{
    background:#1f2937 !important;
}

.stSelectbox div[data-baseweb="select"] > div span{
    color:white !important;
    -webkit-text-fill-color:white !important;
}

.stSelectbox div[data-baseweb="select"] input{
    color:white !important;
    -webkit-text-fill-color:white !important;
    caret-color: transparent !important;
    outline:none !important;
    color: transparent !important;
    -webkit-text-fill-color: transparent !important;
}

.stSelectbox div[data-baseweb="select"] svg{
    fill:white !important;
    color:white !important;
}

ul[role="listbox"]{
    background:#111827 !important;
}

ul[role="listbox"] li{
    color:white !important;
    background:#111827 !important;
}

ul[role="listbox"] li:hover{
    background:#374151 !important;
}

/* ==========================
   PLACEHOLDER TEXT
========================== */

::placeholder{

    color:#6b7280 !important;

}                                    
            
</style>
""", unsafe_allow_html=True)

st.markdown("""
# 🌾 AI-Based Agricultural Demand and Supply Forecasting System

### AI Powered Crop Yield, Demand & Supply Forecasting Dashboard
""")
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
    demand_model = joblib.load(
    r"D:\demand_model.pkl"
    )

    supply_model = joblib.load(
    r"D:\supply_model.pkl"
    )

    return (
        xgb_model,
        lstm_model,
        scaler_x,
        scaler_y,
        le_state,
        le_district,
        le_crop,
        demand_model,
        supply_model
    )


# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():

    final_dataset = pd.read_csv(
        r"D:\final_dataset_enhanced.csv"
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
    le_crop,
    demand_model,
    supply_model

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
st.sidebar.image(
"https://img.icons8.com/color/96/agriculture.png",
width=80
)

st.sidebar.title("🌾 Navigation")
st.sidebar.markdown("---")

page = st.sidebar.selectbox(
    "Select Menu",
    [
        "🏠 Home",
        "📂 Dataset",
        "🌾 Yield Prediction",
        "🌱 Crop Recommendation",
        "🌿 Alternative Crop",
        "📈 7 Day Forecast",
        "📅 30 Day Forecast",
        "📊 Demand and Supply Analysis",
        "🤖 Model Comparison"
]
)

# =====================================================
# HOME PAGE
# =====================================================
if page == "🏠 Home":

    st.markdown("## 🌾 Smart Agriculture Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "🌾 Crops",
        final_dataset["Crop"].nunique()
    )

    col2.metric(
        "🏛 States",
        final_dataset["State"].nunique()
    )

    col3.metric(
        "📍 Districts",
        final_dataset["District"].nunique()
    )

    col4.metric(
        "📄 Records",
        len(final_dataset)
    )

    st.divider()

    left, right = st.columns([2,1])

    with left:

        st.subheader("📘 Project Overview")

        st.info("""

This application predicts crop yield, demand and supply
using Artificial Intelligence and Machine Learning.

### Features

✅ Yield Prediction

✅ Crop Recommendation

✅ Alternative Crop Recommendation

✅ Demand & Supply Analysis

✅ 7 Day Forecast

✅ 30 Day Forecast

✅ Model Comparison

""")

    with right:

        st.subheader("🤖 Models Used")

        st.success("""

✔ XGBoost

✔ LSTM

✔ Demand Model

✔ Supply Model

""")

    st.divider()

    st.subheader("📊 Dataset Statistics")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Average Yield",
        round(final_dataset["Yield"].mean(),2)
    )

    c2.metric(
        "Average Rainfall",
        round(final_dataset["rainfall"].mean(),2)
    )

    c3.metric(
        "Average Temperature",
        round(final_dataset["avg_temp"].mean(),2)
    )

    st.divider()

    st.subheader("🌾 Top 10 Crops")

    crop_chart = (
        final_dataset["Crop"]
        .value_counts()
        .head(10)
    )

    st.bar_chart(crop_chart)

    st.divider()

    st.subheader("🏛 Top States")

    state_chart = (
        final_dataset["State"]
        .value_counts()
        .head(10)
    )

    st.bar_chart(state_chart)

# =====================================================
# DATASET PAGE
# =====================================================
elif page == "📂 Dataset":

    st.header("📂 Final Dataset")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "📄 Total Rows",
        final_dataset.shape[0]
    )

    col2.metric(
        "📑 Total Columns",
        final_dataset.shape[1]
    )

    col3.metric(
        "🌾 Unique Crops",
        final_dataset["Crop"].nunique()
    )

    st.divider()

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        final_dataset.head(100),
        use_container_width=True
    )

    st.divider()

    st.subheader("📌 Column Names")

    st.dataframe(
        pd.DataFrame({"Column Names": final_dataset.columns}),
        use_container_width=True
    )

    st.divider()

    st.subheader("📊 Missing Values")

    missing = final_dataset.isnull().sum()

    missing = missing[missing > 0]

    if len(missing) == 0:
        st.success("✅ No Missing Values Found")
    else:
        st.dataframe(
            missing.reset_index().rename(
                columns={
                    "index":"Column",
                    0:"Missing Values"
                }
            ),
            use_container_width=True
        )

    st.divider()

    st.subheader("📈 Dataset Statistics")

    st.dataframe(
        final_dataset.describe(),
        use_container_width=True
    )

    st.divider()

    csv = final_dataset.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇ Download Dataset",
        csv,
        "final_dataset_enhanced.csv",
        "text/csv"
    )
    # =====================================================
# YIELD PREDICTION
# =====================================================
elif page == "🌾 Yield Prediction":

    st.header("🌾 AI Crop Yield Prediction")

    st.markdown("""
    Predict crop yield using **XGBoost** and **LSTM** models based on crop, weather and production details.
    """)

    st.divider()

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
    if st.button("🚀 Predict Yield"):

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

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "🌾 Predicted Yield",
                f"{prediction:.2f} Ton/Hectare"
            )

        with col2:
            st.metric(
                 "🤖 Prediction Model",
                "AI Model"
            )

        st.progress(100)

        chart = pd.DataFrame(
            {
                "Yield": [prediction]
            },
            index=["Prediction"]
        )

        st.bar_chart(chart)

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
if page == "🌱 Crop Recommendation":

    st.header("🌱 AI Crop Recommendation")

    st.markdown("""
    Get the best crop recommendation based on
    predicted yield, weather conditions and AI analysis.
    """)

    st.divider()

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
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "🌱 Recommended Crop",
                top_crop
            )

        with col2:
            st.metric(
                "⭐ Status",
                "Best Choice"
            )

        st.info(f"""
        ### 🌾 AI Recommendation

        **Recommended Crop:** {top_crop}

        This crop is recommended based on
        Yield Prediction,
        Demand Score,
        Supply Score,
        and Final AI Score.
        """)

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
        report = f"""
        Crop Recommendation Report

        Recommended Crop : {top_crop}

        Generated by
        AI-Based Agricultural Demand & Supply Forecasting System
        """

        st.download_button(
            label="📄 Download Recommendation Report",
            data=report,
            file_name="crop_recommendation_report.txt",
            mime="text/plain"
        )


# =====================================================
# ALTERNATIVE CROP
# =====================================================
elif page == "🌿 Alternative Crop":

    st.header("🌿 Alternative Crop Suggestion")

    st.markdown("""
    If the predicted crop is not suitable,
    AI suggests the best alternative crop.
    """)

    st.divider()

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

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "🌿 Alternative Crop",
            best_crop
        )

    with col2:
        st.metric(
            "⭐ Recommendation",
            "Suitable"
        )

    st.info(f"""
    ### 🌱 AI Suggestion

    **Alternative Crop:** {best_crop}

    This crop is suggested because it has
    better yield potential and favorable
    demand & supply conditions.
    """)    

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
    report = f"""
    Alternative Crop Report

    Suggested Crop : {best_crop}

    Generated by
    AI-Based Agricultural Demand & Supply Forecasting System
    """

    st.download_button(
        label="📄 Download Alternative Crop Report",
        data=report,
        file_name="alternative_crop_report.txt",
        mime="text/plain"
    )

# =====================================================
# MODEL COMPARISON
# =====================================================
elif page == "🤖 Model Comparison":

    st.header("🧠 AI Model Comparison")

    st.markdown("""
        Compare the performance of machine learning and deep learning models
        using standard evaluation metrics.
    """)

    st.divider()

    st.info("📊 Performance comparison between XGBoost and LSTM models.")

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
    st.success("✅ Model Comparison Completed Successfully")

    st.info(f"""
    ### 📋 Model Summary

    🏆 Best Model : XGBoost

    📊 Compared Metrics:
    • MAE
    • RMSE
    • R² Score

    🤖 XGBoost achieved the highest R² Score and provides the best prediction performance based on the evaluation metrics.
    """)

    st.caption("🌾 AI-Based Agricultural Demand & Supply Forecasting System")

# =====================================================
# Demand & Supply Analysis
# =====================================================
elif page == "📊 Demand and Supply Analysis":

    st.header("📊 AI Demand & Supply Analysis")

    st.markdown("""
    Analyze the predicted market demand and supply
    using AI-generated scores.
    """)

    st.divider()

    st.info("🤖 AI is evaluating the demand and supply of the selected crop.")
    analysis_df = final_dataset[
        [
            "Crop",
            "Demand_Score",
            "Demand_Level",
            "Supply_Score",
            "Supply_Level",
            "Market_Status",
            "Recommendation"
        ]
    ].head(100)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🌾 Total Crops", len(analysis_df))

    with col2:
        st.metric(
            "📈 Avg Demand",
            round(analysis_df["Demand_Score"].mean(), 2)
        )

    with col3:
        st.metric(
            "📦 Avg Supply",
            round(analysis_df["Supply_Score"].mean(), 2)
        )

    st.dataframe(analysis_df)
    st.subheader("📊 Demand vs Supply")

    chart_df = analysis_df.set_index("Crop")

    st.bar_chart(
        chart_df[
            [
                "Demand_Score",
                "Supply_Score"
            ]
        ]
    
    )
    st.subheader("📈 Demand Trend")

    st.area_chart(
        chart_df[
            ["Demand_Score"]
        ]
    )
    st.success("✅ Demand & Supply Analysis Completed")

    st.info(f"""
    ### 📋 Market Summary

    🌾 Records : {len(analysis_df)}

    📈 Average Demand :
    {round(analysis_df['Demand_Score'].mean(),2)}

    📦 Average Supply :
    {round(analysis_df['Supply_Score'].mean(),2)}

    🤖 AI analysis completed successfully.
    """)
    csv = analysis_df.to_csv(index=False)

    st.download_button(
        label="📄 Download Demand & Supply Report",
        data=csv,
        file_name="demand_supply_analysis.csv",
        mime="text/csv"
    )
    st.divider()

    st.caption(
        "🌾 AI-Based Agricultural Demand & Supply Forecasting System"
    )

    
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
if page == "📈 7 Day Forecast":

    st.header("📅 AI 7-Day Yield Forecast")

    st.markdown("""
    View the AI-generated crop yield forecast for the next **7 days**.
    """)

    st.divider()

    st.info("🤖 AI is forecasting the crop yield for the next 7 days.")

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

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📅 Days", "7")

    with col2:
        st.metric(
            "📊 Avg Yield",
            round(forecast_df["Predicted Yield"].mean(), 2)
        )

    with col3:
        st.metric(
            "📈 Max Yield",
            round(forecast_df["Predicted Yield"].max(), 2)
        )

    st.dataframe(
        forecast_df
    )
    st.subheader("📈 Yield Trend")

    st.line_chart(
        forecast_df.set_index(
            "Day"
        )
    )
    st.subheader("📊 Yield Comparison")

    st.bar_chart(
        forecast_df.set_index("Day")
    )

    st.success("✅ 7-Day Forecast Generated Successfully")

    st.info(f"""
    ### Forecast Summary

    🌱 Crop : {forecast_crop}

    📅 Forecast Days : 7

    🤖 Model : LSTM
    """)

    csv = forecast_df.to_csv(index=False)

    st.download_button(
        label="📄 Download 7-Day Forecast Report",
        data=csv,
        file_name="7_day_forecast.csv",
        mime="text/csv"
    )
    st.divider()

    st.caption("🌾 AI-Based Agricultural Demand & Supply Forecasting System")


# =====================================================
# 30 DAY FORECAST
# =====================================================
elif page == "📅 30 Day Forecast":

    st.header("📅 AI 30-Day Yield Forecast")

    st.markdown("""
    View the AI-generated crop yield forecast for the next **30 days**.
    """)

    st.divider()

    st.info("🤖 AI is forecasting the crop yield for the next 30 days.")

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

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "📅 Days",
            "30"
        )

    with col2:
        st.metric(
            "📊 Avg Yield",
            round(forecast_df["Predicted Yield"].mean(), 2)
        )

    with col3:
        st.metric(
            "📈 Max Yield",
            round(forecast_df["Predicted Yield"].max(), 2)
        )

    st.dataframe(
        forecast_df
    )

    st.subheader("📈 Yield Trend")

    st.line_chart(
        forecast_df.set_index(
            "Day"
        )
    )
    st.subheader("📊 Yield Comparison")

    st.bar_chart(
        forecast_df.set_index("Day")
    )

    st.success("✅ 30-Day Forecast Generated Successfully")

    st.info(f"""
    ### Forecast Summary

    🌱 Crop : {forecast_crop}

    📅 Forecast Days : 30

    🤖 Model : LSTM
    """)

    csv = forecast_df.to_csv(index=False)

    st.download_button(
        label="📄 Download 30-Day Forecast Report",
        data=csv,
        file_name="30_day_forecast.csv",
        mime="text/csv"
    )
    st.divider()

    st.caption(
        "🌾 AI-Based Agricultural Demand & Supply Forecasting System"
    )

    