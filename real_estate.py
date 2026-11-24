import streamlit as st
import pandas as pd
from pathlib import Path
from real_estate.predictor import predict_property_price

# --------------------------------------------------
# LOAD REAL ESTATE DATASET
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR / "real_estate" / "data" / "nigeria_houses_cleaned.csv"
)

@st.cache_data
def load_real_estate_data():
    return pd.read_csv(DATA_PATH)

df = load_real_estate_data()

# --------------------------------------------------
# PREPARE DROPDOWN OPTIONS
# --------------------------------------------------

property_types = sorted(
    df["title"].dropna().astype(str).str.strip().unique()
)

states = sorted(
    df["state"].dropna().astype(str).str.strip().unique()
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AutoValue AI | Real Estate Valuation",
    page_icon="🏠",
    layout="wide"
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.subheader("Real Estate Valuation")

st.write(
    "Estimate the market value of a property using "
    "our machine learning-powered valuation model."
)

st.divider()

# --------------------------------------------------
# PROPERTY INFORMATION FORM
# --------------------------------------------------

st.header("Enter Property Information")

selected_state = st.selectbox("State", states)

towns = sorted(
    df.loc[
        df["state"].str.strip() == selected_state,
        "town"
    ].dropna().astype(str).str.strip().unique()
)

selected_town = st.selectbox("Town / Area", towns)

location = f"{selected_town}, {selected_state}".lower()

with st.form("real_estate_valuation_form"):

    col1, col2 = st.columns(2)

    with col1:
        bedrooms = st.number_input("Number of Bedrooms", 0, 50, 3)
        bathrooms = st.number_input("Number of Bathrooms", 0, 50, 3)
        toilets = st.number_input("Number of Toilets", 0, 50, 3)

    with col2:
        parking_space = st.number_input("Number of Parking Spaces", 0, 100, 2)

        title = st.selectbox(
            "Property Type",
            property_types
        ).strip().lower()

    submitted = st.form_submit_button(
        "Estimate Property Value",
        use_container_width=True
    )

# --------------------------------------------------
# PROPERTY PRICE PREDICTION
# --------------------------------------------------

if submitted:
    try:
        prediction = predict_property_price(
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            toilets=toilets,
            parking_space=parking_space,
            title=title,
            location=location
        )

        st.divider()
        st.header("Estimated Property Value")

        st.metric(
            "Predicted Property Price",
            f"₦{prediction:,.0f}"
        )

        st.info(
            "This is a machine-learning estimate, not a guaranteed "
            "market selling price. Actual property values may vary."
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")