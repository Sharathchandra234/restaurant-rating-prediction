"""
Data loading module for the Restaurant Rating Prediction project.
"""

import pandas as pd
import logging
from pathlib import Path
from src.config import DATASET_PATH

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(file_path: Path = DATASET_PATH) -> pd.DataFrame:
    """
    Load the dataset from the specified file path.

    Parameters:
    -----------
    file_path : Path, optional
        Path to the CSV file. Defaults to DATASET_PATH from config.

    Returns:
    --------
    pd.DataFrame
        Loaded dataset.

    Raises:
    -------
    FileNotFoundError
        If the file does not exist at the specified path.
    pd.errors.EmptyDataError
        If the file is empty.
    """
    logger.info(f"Loading dataset from {file_path}")
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    try:
        df = pd.read_csv(file_path)
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    except pd.errors.EmptyDataError:
        logger.error("The dataset file is empty.")
        raise
    except Exception as e:
        logger.error(f"An error occurred while loading the dataset: {e}")
        raise


def validate_dataset(df: pd.DataFrame) -> bool:
    """
    Perform basic validation on the dataset.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataset to validate.

    Returns:
    --------
    bool
        True if validation passes, False otherwise.
    """
    logger.info("Validating dataset...")

    # Check if DataFrame is empty
    if df.empty:
        logger.error("The dataset is empty.")
        return False

    # Check for essential columns
    essential_columns = ['Aggregate rating']  # Target variable
    missing_columns = [col for col in essential_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing essential columns: {missing_columns}")
        return False

    # Check for missing values in essential columns
    for col in essential_columns:
        if df[col].isnull().any():
            logger.warning(f"Column '{col}' contains missing values.")

    logger.info("Dataset validation passed.")
    return True


if __name__ == "__main__":
    # Example usage
    df = load_dataset()
    validate_dataset(df)
    print(df.head())