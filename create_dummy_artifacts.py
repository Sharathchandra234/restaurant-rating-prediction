import joblib
import os
from sklearn.preprocessing import StandardScaler

os.makedirs("models", exist_ok=True)

# Create dummy preprocessor
preprocessor = StandardScaler()

# Create dummy feature engineer
feature_engineer = {
    "status": "dummy_feature_engineer"
}

# Save artifacts
joblib.dump(preprocessor, "models/preprocessor.joblib")
joblib.dump(feature_engineer, "models/feature_engineer.joblib")

print("Dummy artifacts created successfully!")