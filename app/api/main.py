"""
FastAPI application for Restaurant Rating Prediction.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.data.load_data import load_dataset
from src.features.feature_engineering import FeatureEngineer
from src.preprocessing.preprocessor import preprocess_data
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# Add the src directory to the path so we can import our custom modules
sys_path = Path(__file__).resolve().parent.parent.parent / "src"
if str(sys_path) not in os.sys.path:
    os.sys.path.append(str(sys_path))

from data.load_data import load_dataset
from features.feature_engineering import FeatureEngineer
from preprocessing.preprocessor import preprocess_data

# Initialize the FastAPI app
app = FastAPI(
    title="Restaurant Rating Prediction API",
    description="API for predicting restaurant ratings based on various features.",
    version="1.0.0"
)

# Load the model and preprocessing artifacts on startup
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"

# Global variables for the model, preprocessor, and feature engineer
model = None
preprocessor = None
feature_engineer = None


@app.on_event("startup")
def load_artifacts():
    """Load the saved model, preprocessor, and feature engineer on startup."""
    global model, preprocessor, feature_engineer

    # Load the model
    model_path = MODEL_DIR / "restaurant_rating_model.joblib"
    if not model_path.exists():
        raise RuntimeError(f"Model file not found at {model_path}. Please train the model first.")
    model = joblib.load(model_path)

    # Load the preprocessor
    preprocessor_path = MODEL_DIR / "preprocessor.joblib"
    if not preprocessor_path.exists():
        raise RuntimeError(f"Preprocessor file not found at {preprocessor_path}.")
    preprocessor = joblib.load(preprocessor_path)

    # Load the feature engineer
    feature_engineer_path = MODEL_DIR / "feature_engineer.joblib"
    if not feature_engineer_path.exists():
        raise RuntimeError(f"Feature engineer file not found at {feature_engineer_path}.")
    feature_engineer = joblib.load(feature_engineer_path)


# Define the input data model using Pydantic
class RestaurantFeatures(BaseModel):
    Restaurant_Name: str
    Country_Code: int
    City: str
    Address: str
    Locality: str
    Locality_Verbose: str
    Longitude: float
    Latitude: float
    Cuisines: str
    Average_Cost_for_two: float
    Currency: str
    Has_Table_booking: str
    Has_Online_delivery: str
    Is_delivering_now: str
    Switch_to_order_menu: str
    Price_range: int
    Votes: int

    class Config:
        # Allow underscore to be converted to space or vice versa if needed
        # We'll keep the field names as they are and map them in the preprocessing step.
        # Actually, we expect the exact column names as in the original dataset.
        # Let's use an alias to convert the underscores to spaces if needed?
        # But the frontend might send the data with underscores. We'll keep as is and then rename.
        allow_population_by_field_name = True


# Define a helper function to convert the Pydantic model to a DataFrame with the original column names
def convert_to_dataframe(features: RestaurantFeatures) -> pd.DataFrame:
    """Convert the Pydantic model to a DataFrame with the original column names."""
    # Create a dictionary with the original column names (with spaces)
    data = {
        'Restaurant Name': features.Restaurant_Name,
        'Country Code': features.Country_Code,
        'City': features.City,
        'Address': features.Address,
        'Locality': features.Locality,
        'Locality Verbose': features.Locality_Verbose,
        'Longitude': features.Longitude,
        'Latitude': features.Latitude,
        'Cuisines': features.Cuisines,
        'Average Cost for two': features.Average_Cost_for_two,
        'Currency': features.Currency,
        'Has Table booking': features.Has_Table_booking,
        'Has Online delivery': features.Has_Online_delivery,
        'Is delivering now': features.Is_delivering_now,
        'Switch to order menu': features.Switch_to_order_menu,
        'Price range': features.Price_range,
        'Votes': features.Votes
    }
    return pd.DataFrame([data])


# Define the prediction endpoint
@app.post("/predict", response_model=Dict[str, Any])
def predict_rating(features: RestaurantFeatures):
    """
    Predict the aggregate rating for a restaurant.

    Parameters:
    -----------
    features : RestaurantFeatures
        The features of the restaurant.

    Returns:
    --------
    dict
        A dictionary containing the predicted rating and some additional information.
    """
    try:
        # Convert the input to a DataFrame
        input_df = convert_to_dataframe(features)

        # Step 1: Feature Engineering
        engineered_df = feature_engineer.transform(input_df)

        # Step 2: Preprocessing
        processed_array, _, _ = preprocess_data(
            engineered_df,
            feature_engineer=feature_engineer,
            preprocessor=preprocessor,
            fit=False
        )

        # Step 3: Prediction
        prediction = model.predict(processed_array)
        # Ensure the prediction is within a reasonable range (0 to 5)
        prediction = np.clip(prediction[0], 0, 5)

        # Prepare the response
        response = {
            "predicted_rating": float(prediction),
            "rating_out_of_5": f"{prediction:.2f} / 5.0",
            "rating_category": get_rating_category(prediction),
            "message": "Prediction successful"
        }

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


def get_rating_category(rating: float) -> str:
    """Convert a numerical rating to a category."""
    if rating >= 4.5:
        return "Excellent"
    elif rating >= 3.5:
        return "Very Good"
    elif rating >= 2.5:
        return "Good"
    elif rating >= 1.5:
        return "Fair"
    else:
        return "Poor"


# Define a health check endpoint
@app.get("/health")
def health_check():
    """Health check endpoint to verify the API is running."""
    return {"status": "healthy", "message": "Restaurant Rating Prediction API is running."}


# Define a root endpoint
@app.get("/")
def root():
    """Root endpoint with basic information."""
    return {
        "message": "Welcome to the Restaurant Rating Prediction API",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    # Run the application with Uvicorn
    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000, reload=True)