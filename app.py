
import streamlit as st
import runpy
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AutoValue AI | Intelligent Valuation Engine",
    page_icon="⚖️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("AutoValue AI")

st.subheader("Know What Your Assets Are Worth.")

st.write(
    "Get intelligent market value estimates for vehicles "
    "and real estate, powered by machine learning."
)

st.divider()

# --------------------------------------------------
# VALUATION SERVICES
# --------------------------------------------------

selected_service = st.selectbox(
    "Valuation Services",
    [
        "Vehicle Valuation",
        "Real Estate Valuation"
    ]
)

# --------------------------------------------------
# LOAD SELECTED PAGE
# --------------------------------------------------

if selected_service == "Vehicle Valuation":

    runpy.run_path(
        str(BASE_DIR / "pages" / "vehicle.py")
    )

elif selected_service == "Real Estate Valuation":

    runpy.run_path(
        str(BASE_DIR / "pages" / "real_estate.py")
    )
