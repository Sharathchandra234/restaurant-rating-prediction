"""
Configuration module for the Restaurant Rating Prediction project.
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
SRC_DIR = BASE_DIR / "src"
APP_DIR = BASE_DIR / "app"

# Data files
DATASET_PATH = DATA_DIR / "Dataset.csv"

# Model and artifact paths
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "restaurant_rating_model.joblib"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.joblib"

# Hyperparameters for models
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Feature engineering parameters
CUISINE_POPULARITY_THRESHOLD = 10  # Minimum occurrences to consider a cuisine popular
CITY_RATING_THRESHOLD = 5  # Minimum number of restaurants in a city to compute city-wise average

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)