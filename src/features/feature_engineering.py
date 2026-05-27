"""
Feature engineering module for the Restaurant Rating Prediction project.
"""

import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path
from src.config import (
    CUISINE_POPULARITY_THRESHOLD,
    CITY_RATING_THRESHOLD,
    MODEL_DIR
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    A class to engineer features for the Restaurant Rating Prediction project.
    It learns certain statistics from the training data (like cuisine popularity, city average ratings, etc.)
    and then uses them to engineer features for any data (train, test, or new).
    """

    def __init__(self):
        self.restaurant_name_counts_ = None
        self.cuisine_popularity_ = None
        self.city_avg_rating_ = None
        self.cost_bin_edges_ = None
        self.is_fitted_ = False

    def fit(self, X: pd.DataFrame, y=None):
        """
        Learn the statistics from the training data.

        Parameters:
        -----------
        X : pd.DataFrame
            The training data (features only, no target).
        y : pd.Series, optional
            The target variable. Not used, but kept for compatibility with sklearn.

        Returns:
        --------
        self : object
            Returns self.
        """
        logger.info("Fitting FeatureEngineer...")

        # Make a copy to avoid modifying the original data
        X_fit = X.copy()

        # 1. Restaurant name frequency
        self.restaurant_name_counts_ = X_fit['Restaurant Name'].value_counts()
        logger.info("Learned restaurant name frequencies.")

        # 2. Cuisine popularity score
        # Split cuisines by comma and space, then explode and count
        cuisines_split = X_fit['Cuisines'].str.split(', ')
        cuisines_exploded = cuisines_split.explode()
        self.cuisine_popularity_ = cuisines_exploded.value_counts()
        logger.info("Learned cuisine popularities.")

        # 3. City-wise average ratings
        self.city_avg_rating_ = X_fit.groupby('City')['Aggregate rating'].mean()
        logger.info("Learned city average ratings.")

        # 4. Cost bin edges for cost categorization
        # We compute the bin edges for 'Average Cost for two' using quantiles
        try:
            # We'll use qcut to get the bin edges for 3 quantiles (Low, Medium, High)
            _, self.cost_bin_edges_ = pd.qcut(
                X_fit['Average Cost for two'],
                q=3,
                retbins=True,
                labels=['Low', 'Medium', 'High']
            )
            logger.info("Learned cost bin edges for categorization.")
        except Exception as e:
            logger.warning(f"Could not compute cost bin edges: {e}. Will use a default binning.")
            # Set default bin edges if computation fails
            self.cost_bin_edges_ = [0, X_fit['Average Cost for two'].median(), np.inf]

        self.is_fitted_ = True
        logger.info("FeatureEngineer fitting completed.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for the given data using the learned statistics.

        Parameters:
        -----------
        X : pd.DataFrame
            The data to engineer features for.

        Returns:
        --------
        pd.DataFrame
            The data with engineered features added.
        """
        if not self.is_fitted_:
            raise RuntimeError("This FeatureEngineer instance is not fitted yet. Call 'fit' with appropriate data first.")

        logger.info("Transforming data with FeatureEngineer...")
        # Make a copy to avoid modifying the original data
        X_transformed = X.copy()

        # 1. Restaurant name frequency
        X_transformed['Restaurant Name Frequency'] = X_transformed['Restaurant Name'].map(self.restaurant_name_counts_)
        # Fill NaN with 0 (if a restaurant name is not seen during fit)
        X_transformed['Restaurant Name Frequency'] = X_transformed['Restaurant Name Frequency'].fillna(0)
        logger.info("Added 'Restaurant Name Frequency' feature.")

        # 2. Cuisine popularity score
        def cuisine_popularity_score(cuisines):
            if pd.isna(cuisines):
                return 0
            cuisine_list = cuisines.split(', ')
            # Get popularity for each cuisine, default to 0 if not found
            scores = [self.cuisine_popularity_.get(cuisine, 0) for cuisine in cuisine_list]
            return np.mean(scores) if scores else 0

        X_transformed['Cuisine Popularity Score'] = X_transformed['Cuisines'].apply(cuisine_popularity_score)
        logger.info("Added 'Cuisine Popularity Score' feature.")

        # 3. City-wise average ratings
        X_transformed['City Avg Rating'] = X_transformed['City'].map(self.city_avg_rating_)
        # Fill NaN with the overall average rating (or 0) if city not seen during fit
        overall_avg = self.city_avg_rating_.mean() if not self.city_avg_rating_.isnull().all() else 0
        X_transformed['City Avg Rating'] = X_transformed['City Avg Rating'].fillna(overall_avg)
        logger.info("Added 'City Avg Rating' feature.")

        # 4. Cost efficiency metrics (rating per unit cost)
        # Avoid division by zero
        X_transformed['Cost Efficiency'] = np.where(
            X_transformed['Average Cost for two'] > 0,
            X_transformed['Aggregate rating'] / X_transformed['Average Cost for two'],
            0
        )
        logger.info("Added 'Cost Efficiency' feature.")

        # 5. Price category features (we'll keep the original Price range and also create a binary feature for high price?)
        # We'll just keep the original Price range for now, but note that it will be one-hot encoded in preprocessing.
        # We can also create an ordinal feature if needed, but we'll leave it as is.
        logger.info("Keeping 'Price range' as is for preprocessing.")

        # 6. Online delivery impact (binary)
        X_transformed['Online Delivery'] = X_transformed['Has Online delivery'].map({'Yes': 1, 'No': 0})
        # Fill NaN with 0
        X_transformed['Online Delivery'] = X_transformed['Online Delivery'].fillna(0)
        logger.info("Added 'Online Delivery' feature.")

        # 7. Table booking impact (binary)
        X_transformed['Table Booking'] = X_transformed['Has Table booking'].map({'Yes': 1, 'No': 0})
        # Fill NaN with 0
        X_transformed['Table Booking'] = X_transformed['Table Booking'].fillna(0)
        logger.info("Added 'Table Booking' feature.")

        # 8. Weighted rating features (rating * votes, normalized)
        # Note: We cannot normalize by the max of the entire dataset here because we don't have the entire dataset in transform.
        # Instead, we will not normalize and let the preprocessing step handle scaling.
        # Or we can normalize by a fixed value? We'll skip normalization for now and note that the preprocessing step will scale.
        X_transformed['Weighted Rating'] = X_transformed['Aggregate rating'] * X_transformed['Votes']
        logger.info("Added 'Weighted Rating' feature.")

        # 9. Restaurant service quality indicators
        X_transformed['Service Score'] = X_transformed['Online Delivery'] + X_transformed['Table Booking']
        logger.info("Added 'Service Score' feature.")

        # 10. Length of restaurant name
        X_transformed['Restaurant Name Length'] = X_transformed['Restaurant Name'].apply(
            lambda x: len(str(x)) if not pd.isna(x) else 0
        )
        logger.info("Added 'Restaurant Name Length' feature.")

        # 11. Is multivariate cuisine (more than one cuisine)
        X_transformed['Is Multivariate Cuisine'] = X_transformed['Cuisines'].apply(
            lambda x: 1 if pd.notna(x) and ', ' in str(x) else 0
        )
        logger.info("Added 'Is Multivariate Cuisine' feature.")

        # 12. Log transformation of cost
        X_transformed['Log Cost'] = np.log1p(X_transformed['Average Cost for two'])
        logger.info("Added 'Log Cost' feature.")

        # 13. Cost categories
        if self.cost_bin_edges_ is not None and len(self.cost_bin_edges_) > 1:
            X_transformed['Cost Category'] = pd.cut(
                X_transformed['Average Cost for two'],
                bins=self.cost_bin_edges_,
                labels=['Low', 'Medium', 'High'],
                include_lowest=True
            )
        else:
            # If we don't have bin edges (because we didn't fit or we didn't compute them), we'll put everything in 'Medium'
            X_transformed['Cost Category'] = 'Medium'
            logger.warning("Cost bin edges not found. Setting all to 'Medium'.")

        logger.info("Feature engineering transformation completed.")
        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y=None):
        """
        Fit to data, then transform it.

        Parameters:
        -----------
        X : pd.DataFrame
            Training data.
        y : pd.Series, optional
            Target variable.

        Returns:
        --------
        pd.DataFrame
            Transformed data.
        """
        return self.fit(X, y).transform(X)

    def save_artifacts(self, dir_path: Path = MODEL_DIR):
        """
        Save the learned attributes to disk.

        Parameters:
        -----------
        dir_path : Path, optional
            Directory to save the artifacts. Defaults to MODEL_DIR from config.
        """
        logger.info(f"Saving FeatureEngineer artifacts to {dir_path}...")
        dir_path.mkdir(exist_ok=True)
        joblib.dump(self.restaurant_name_counts_, dir_path / "restaurant_name_counts.joblib")
        joblib.dump(self.cuisine_popularity_, dir_path / "cuisine_popularity.joblib")
        joblib.dump(self.city_avg_rating_, dir_path / "city_avg_rating.joblib")
        # Save the cost bin edges if they exist
        if self.cost_bin_edges_ is not None:
            joblib.dump(self.cost_bin_edges_, dir_path / "cost_bin_edges.joblib")
        logger.info("FeatureEngineer artifacts saved.")

    def load_artifacts(self, dir_path: Path = MODEL_DIR):
        """
        Load the learned attributes from disk.

        Parameters:
        -----------
        dir_path : Path, optional
            Directory to load the artifacts from. Defaults to MODEL_DIR from config.
        """
        logger.info(f"Loading FeatureEngineer artifacts from {dir_path}...")
        self.restaurant_name_counts_ = joblib.load(dir_path / "restaurant_name_counts.joblib")
        self.cuisine_popularity_ = joblib.load(dir_path / "cuisine_popularity.joblib")
        self.city_avg_rating_ = joblib.load(dir_path / "city_avg_rating.joblib")
        # Try to load cost bin edges
        cost_bin_edges_path = dir_path / "cost_bin_edges.joblib"
        if cost_bin_edges_path.exists():
            self.cost_bin_edges_ = joblib.load(cost_bin_edges_path)
        else:
            self.cost_bin_edges_ = None
        self.is_fitted_ = True
        logger.info("FeatureEngineer artifacts loaded.")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to engineer features on a dataframe.
    Note: This function computes the statistics from the dataframe passed in and uses them to engineer features.
    This is only appropriate for the training data. For production, use the FeatureEngineer class.

    Parameters:
    -----------
    df : pd.DataFrame
        The input dataset.

    Returns:
    --------
    pd.DataFrame
        The dataset with engineered features.
    """
    logger.warning("Using the convenience function engineer_features. This is only appropriate for training data.")
    engineer = FeatureEngineer()
    return engineer.fit_transform(df)


if __name__ == "__main__":
    # Example usage
    from src.data.load_data import load_dataset, validate_dataset
    df = load_dataset()
    if validate_dataset(df):
        engineer = FeatureEngineer()
        df_engineered = engineer.fit_transform(df)
        print(df_engineered.head())
        print(f"Original shape: {df.shape}, Engineered shape: {df_engineered.shape}")
        # Save the artifacts for later use
        engineer.save_artifacts()