"""
Model training module for the Restaurant Rating Prediction project.
"""

import logging
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import joblib
from src.config import MODEL_PATH, PREPROCESSOR_PATH, FEATURE_NAMES_PATH, RANDOM_STATE, TEST_SIZE

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_models(X_train, y_train, X_test, y_test):
    """
    Train multiple regression models and evaluate their performance.

    Parameters:
    -----------
    X_train : array-like
        Training features.
    y_train : array-like
        Training target.
    X_test : array-like
        Test features.
    y_test : array-like
        Test target.

    Returns:
    --------
    dict
        Dictionary containing trained models and their evaluation metrics.
    """
    logger.info("Training multiple regression models...")

    # Define models to train
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=RANDOM_STATE),
        'Random Forest': RandomForestRegressor(random_state=RANDOM_STATE, n_estimators=100),
        'Gradient Boosting': GradientBoostingRegressor(random_state=RANDOM_STATE, n_estimators=100),
        'XGBoost': XGBRegressor(random_state=RANDOM_STATE, n_estimators=100)
    }

    # Dictionary to store results
    results = {}

    # Train and evaluate each model
    for name, model in models.items():
        logger.info(f"Training {name}...")

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Calculate metrics
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)

        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
        cv_mae = -cv_scores.mean()

        # Store results
        results[name] = {
            'model': model,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'cv_mae': cv_mae,
            'y_pred_test': y_pred_test
        }

        logger.info(f"{name} - Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}, Test R2: {test_r2:.4f}")

    return results


def hyperparameter_tuning(model, param_grid, X_train, y_train, method='grid', n_iter=10):
    """
    Perform hyperparameter tuning on a given model.

    Parameters:
    -----------
    model : estimator
        The model to tune.
    param_grid : dict
        Dictionary with parameters names as keys and lists of parameter settings to try.
    X_train : array-like
        Training features.
    y_train : array-like
        Training target.
    method : str, optional
        Method to use for search ('grid' or 'random'). Default is 'grid'.
    n_iter : int, optional
        Number of parameter settings sampled for random search. Default is 10.

    Returns:
    --------
    estimator
        The best model found by the search.
    """
    logger.info(f"Performing {method} search for hyperparameter tuning...")

    if method == 'grid':
        search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    elif method == 'random':
        search = RandomizedSearchCV(model, param_grid, n_iter=n_iter, cv=5,
                                  scoring='neg_mean_absolute_error', n_jobs=-1, random_state=RANDOM_STATE)
    else:
        raise ValueError("Method must be either 'grid' or 'random'")

    search.fit(X_train, y_train)
    logger.info(f"Best parameters: {search.best_params_}")
    logger.info(f"Best CV MAE: {-search.best_score_:.4f}")

    return search.best_estimator_


def save_model(model, preprocessor, feature_names):
    """
    Save the trained model, preprocessor, and feature names to disk.

    Parameters:
    -----------
    model : estimator
        Trained model to save.
    preprocessor : ColumnTransformer
        Fitted preprocessor to save.
    feature_names : list
        List of feature names used in training.
    """
    logger.info("Saving model and preprocessing artifacts...")

    # Ensure model directory exists
    MODEL_PATH.parent.mkdir(exist_ok=True)

    # Save model
    joblib.dump(model, MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")

    # Save preprocessor
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    logger.info(f"Preprocessor saved to {PREPROCESSOR_PATH}")

    # Save feature names
    joblib.dump(feature_names, FEATURE_NAMES_PATH)
    logger.info(f"Feature names saved to {FEATURE_NAMES_PATH}")


def load_model():
    """
    Load the trained model, preprocessor, and feature names from disk.

    Returns:
    --------
    tuple
        (model, preprocessor, feature_names)
    """
    logger.info("Loading model and preprocessing artifacts...")

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)

    logger.info("Model and preprocessing artifacts loaded.")
    return model, preprocessor, feature_names


if __name__ == "__main__":
    # Example usage would go here
    pass