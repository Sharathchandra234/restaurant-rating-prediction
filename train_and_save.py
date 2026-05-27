import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Create models directory
os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("data/Dataset.csv")

# Basic preprocessing
df = df.select_dtypes(include=['number']).dropna()

# Simple target selection
target_column = df.columns[-1]

X = df.drop(columns=[target_column])
y = df[target_column]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save artifacts
joblib.dump(model, "models/restaurant_rating_model.joblib")

print("Model trained and saved successfully!")