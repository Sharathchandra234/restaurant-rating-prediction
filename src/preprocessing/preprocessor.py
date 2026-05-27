"""
Preprocessing module for the Restaurant Rating Prediction project.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.features.feature_engineering import FeatureEngineer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_preprocessor(numerical_features, categorical_features):
    """
    Create a preprocessing pipeline for numerical and categorical features.

    Parameters:
    -----------
    numerical_features : list
        List of numerical column names.
    categorical_features : list
        List of categorical column names.

    Returns:
        ColumnTransformer: Preprocessing pipeline.
    """
    logger.info("Creating preprocessing pipeline...")

    # Numerical pipeline: impute missing values with median and scale
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline: impute missing values with most frequent and one-hot encode
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ])

    logger.info("Preprocessing pipeline created.")
    return preprocessor


def preprocess_data(X, feature_engineer=None, preprocessor=None, fit=False):
    """
    Preprocess the data using feature engineering and preprocessing pipeline.

    Parameters:
    -----------
    X : pd.DataFrame
        Input features.
    feature_engineer : FeatureEngineer, optional
        Fitted feature engineer. If None, a new one is created and fitted (if fit=True).
    preprocessor : ColumnTransformer, optional
        Fitted preprocessor. If None, a new one is created and fitted (if fit=True).
    fit : bool, optional
        Whether to fit the feature engineer and preprocessor. Default is False.

    Returns:
    --------
    tuple
        (X_processed, feature_engineer, preprocessor) where X_processed is a numpy array
        and feature_engineer and preprocessor are the fitted objects.
    """
    logger.info("Preprocessing data...")

    # Step 1: Feature Engineering
    if feature_engineer is None:
        feature_engineer = FeatureEngineer()
        if fit:
            logger.info("Fitting feature engineer...")
            X_engineered = feature_engineer.fit_transform(X)
        else:
            logger.info("Transforming with feature engineer...")
            X_engineered = feature_engineer.transform(X)
    else:
        if fit:
            logger.info("Fitting feature engineer...")
            X_engineered = feature_engineer.fit_transform(X)
        else:
            logger.info("Transforming with feature engineer...")
            X_engineered = feature_engineer.transform(X)

    # Step 2: Identify feature types after engineering
    # Exclude the target variable if it's present (though it shouldn't be in X)
    # Also exclude any columns that are not features (like ID columns if we decide to keep them)
    # For now, we'll consider all columns as features for preprocessing
    # In practice, you might want to exclude non-feature columns like 'Restaurant ID'

    # Let's exclude ID columns and other non-feature columns
    non_feature_cols = ['Restaurant ID', 'Restaurant Name', 'Address', 'Locality',
                       'Locality Verbose', 'Longitude', 'Latitude', 'Cuisines',
                       'Has Table booking', 'Has Online delivery', 'Is delivering now',
                       'Switch to order menu', 'Rating color', 'Rating text', 'Votes']
    # Actually, we've engineered features from many of these, so we should keep the engineered ones
    # and drop the original ones that were used to create them

    # For simplicity, we'll use all columns except the ones we know are identifiers or original text
    # that we've already engineered features from
    cols_to_drop = ['Restaurant ID', 'Address', 'Locality', 'Locality Verbose',
                   'Longitude', 'Latitude', 'Cuisines', 'Has Table booking',
                   'Has Online delivery', 'Is delivering now', 'Switch to order menu',
                   'Rating color', 'Rating text']

    # Only drop columns that exist in the dataframe
    cols_to_drop = [col for col in cols_to_drop if col in X_engineered.columns]

    if cols_to_drop:
        X_for_preprocessing = X_engineered.drop(columns=cols_to_drop)
        logger.info(f"Dropped {len(cols_to_drop)} non-feature columns: {cols_to_drop}")
    else:
        X_for_preprocessing = X_engineered

    logger.info(f"Features for preprocessing: {X_for_preprocessing.columns.tolist()}")

    # Identify numerical and categorical features
    numerical_features = X_for_preprocessing.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_for_preprocessing.select_dtypes(include=['object']).columns.tolist()

    logger.info(f"Numerical features: {numerical_features}")
    logger.info(f"Categorical features: {categorical_features}")

    # Step 3: Preprocessing
    if preprocessor is None:
        preprocessor = get_preprocessor(numerical_features, categorical_features)
        if fit:
            logger.info("Fitting preprocessor...")
            X_processed = preprocessor.fit_transform(X_for_preprocessing)
        else:
            logger.info("Transforming with preprocessor...")
            X_processed = preprocessor.transform(X_for_preprocessing)
    else:
        if fit:
            logger.info("Fitting preprocessor...")
            X_processed = preprocessor.fit_transform(X_for_preprocessing)
        else:
            logger.info("Transforming with preprocessor...")
            X_processed = preprocessor.transform(X_for_preprocessing)

    logger.info(f"Data preprocessed. Shape: {X_processed.shape}")
    return X_processed, feature_engineer, preprocessor


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    from src.data.load_data import load_dataset, validate_dataset

    df = load_dataset()
    if validate_dataset(df):
        # For demonstration, we'll preprocess a subset of features
        # In practice, you would select your features for modeling
        X = df.drop('Aggregate rating', axis=1)  # Example: all except target
        X_processed, feature_engineer, preprocessor = preprocess_data(X, fit=True)
        print(X_processed.shape)