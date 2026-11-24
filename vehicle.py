
import streamlit as st
import pandas as pd
from pathlib import Path
from vehicle.predictor import predict_vehicle_price

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AutoValue AI | Vehicle Valuation",
    page_icon="🚗",
    layout="wide"
)

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "vehicle" / "data" / "vehicle_cleaned.csv"

# --------------------------------------------------
# LOAD VEHICLE DATA
# --------------------------------------------------

@st.cache_data
def load_vehicle_data():
    return pd.read_csv(DATA_PATH)

df = load_vehicle_data()

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.subheader("Vehicle Valuation")

st.write(
    "Estimate the market value of a vehicle using "
    "machine learning-powered price prediction."
)

st.divider()

# --------------------------------------------------
# VEHICLE VALUATION FORM
# --------------------------------------------------

st.header("Enter Vehicle Information")

with st.form("vehicle_valuation_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        make = st.selectbox(
            "Vehicle Make",
            sorted(df["Make"].dropna().unique())
        )

        model_options = sorted(
            df.loc[df["Make"] == make, "Model"].dropna().unique()
        )

        model = st.selectbox("Vehicle Model", model_options)

        year = st.number_input(
            "Year of Manufacture",
            min_value=1980,
            max_value=2026,
            value=2018
        )

        colour = st.selectbox(
            "Colour",
            sorted(df["Colour"].dropna().unique())
        )

        condition = st.selectbox(
            "Vehicle Condition",
            sorted(df["Condition"].dropna().unique())
        )

        fuel_type = st.selectbox(
            "Fuel Type",
            sorted(df["fuel type"].dropna().unique())
        )

    with col2:
        gear_type = st.selectbox(
            "Gear Type",
            sorted(df["gear type"].dropna().unique())
        )

        mileage = st.number_input(
            "Mileage (km)",
            min_value=0,
            value=50000
        )

        engine_size = st.number_input(
            "Engine Size",
            min_value=0.1,
            value=2.0,
            step=0.1
        )

        horsepower = st.number_input(
            "Horse Power",
            min_value=1,
            value=150
        )

        cylinders = st.number_input(
            "Number of Cylinders",
            min_value=1,
            value=4
        )

        seats = st.number_input(
            "Number of Seats",
            min_value=1,
            value=5
        )

    with col3:
        car_type = st.selectbox(
            "Vehicle Type",
            sorted(df["car"].dropna().unique())
        )

        trim = st.selectbox(
            "Trim",
            sorted(df["Trim"].dropna().unique())
        )

        drivetrain = st.selectbox(
            "Drivetrain",
            sorted(df["Drivetrain"].dropna().unique())
        )

        selling_condition = st.selectbox(
            "Selling Condition",
            sorted(df["Selling Condition"].dropna().unique())
        )

        bought_condition = st.selectbox(
            "Bought Condition",
            sorted(df["Bought Condition"].dropna().unique())
        )

        registered_city = st.selectbox(
            "Registered City",
            sorted(df["Registered city"].dropna().unique())
        )

    submitted = st.form_submit_button(
        "Estimate Vehicle Value",
        use_container_width=True
    )

# --------------------------------------------------
# PRICE PREDICTION
# --------------------------------------------------

if submitted:

    vehicle_age = 2026 - year

    mileage_per_year = (
        mileage / vehicle_age if vehicle_age > 0 else mileage
    )

    make_model = f"{make}_{model}"

    vehicle_data = {
        "fuel type": fuel_type,
        "gear type": gear_type,
        "Make": make,
        "Model": model,
        "Year of manufacture": year,
        "Colour": colour,
        "Condition": condition,
        "Mileage": mileage,
        "Engine Size": engine_size,
        "Selling Condition": selling_condition,
        "Bought Condition": bought_condition,
        "car": car_type,
        "Trim": trim,
        "Drivetrain": drivetrain,
        "Seats": seats,
        "Number of Cylinders": cylinders,
        "Horse Power": horsepower,
        "Registered city": registered_city,
        "Vehicle Age": vehicle_age,
        "Mileage per year": mileage_per_year,
        "Make_Model": make_model
    }

    try:
        prediction = predict_vehicle_price(vehicle_data)

        st.divider()
        st.header("Estimated Vehicle Value")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Lower Estimate (Q10)",
            f"₦{prediction['lower_estimate']:,.0f}"
        )

        col2.metric(
            "Fair Market Estimate (Q50)",
            f"₦{prediction['median_estimate']:,.0f}"
        )

        col3.metric(
            "Upper Estimate (Q90)",
            f"₦{prediction['upper_estimate']:,.0f}"
        )

        st.info(
            "These are machine-learning estimates, not guaranteed "
            "selling prices. Actual market prices may vary."
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")
