
import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostRegressor

# Locate project directory
BASE_DIR = Path(__file__).resolve().parent

# Locate saved model
MODEL_PATH = BASE_DIR / "models" / "catboost_real_estate.cbm"

# Load trained CatBoost model
model = CatBoostRegressor()
model.load_model(str(MODEL_PATH))


def predict_property_price(
    bedrooms,
    bathrooms,
    toilets,
    parking_space,
    title,
    location
):
    # Calculate total rooms
    total_rooms = bedrooms + bathrooms + toilets

    # Prepare categorical inputs
    title = str(title).strip()
    location = str(location).strip()

    # Prepare property information
    property_data = pd.DataFrame([{
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "toilets": toilets,
        "parking_space": parking_space,
        "title": title,
        "total_rooms": total_rooms,
        "location": location
    }])

    # Ensure numerical columns are numeric
    numerical_columns = [
        "bedrooms",
        "bathrooms",
        "toilets",
        "parking_space",
        "total_rooms"
    ]

    property_data[numerical_columns] = (
        property_data[numerical_columns].astype(float)
    )

    # Predict log-transformed price
    predicted_log_price = model.predict(property_data)

    # Convert back to Naira
    predicted_price = np.expm1(predicted_log_price[0])

    return float(predicted_price)
