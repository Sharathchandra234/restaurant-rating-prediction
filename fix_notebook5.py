lines = open('notebooks/modeling.ipynb.backup', 'r', encoding='utf-8').readlines()

# Fix imports
for i, line in enumerate(lines):
    stripped = line.strip()
    if stripped == 'from data.load_data import load_dataset, validate_dataset':
        lines[i] = 'from src.data.load_data import load_dataset, validate_dataset\n'
    elif stripped == 'from features.feature_engineering import engineer_features':
        lines[i] = 'from src.features.feature_engineering import engineer_features\n'
    elif stripped == 'from preprocessing.preprocessor import preprocess_data':
        lines[i] = 'from src.preprocessing.preprocessor import preprocess_data\n'
    elif stripped == 'from models.train_model import train_models, hyperparameter_tuning, save_model':
        lines[i] = 'from src.models.train_model import train_models, hyperparameter_tuning, save_model\n'

# Fix the param_grids line
for i, line in enumerate(lines):
    if '\n    \"    ' in line and "'Random Forest'" in line:
        lines[i] = line.replace('\n    \"    ', '\n    ')

# Write the fixed file
with open('notebooks/modeling.ipynb', 'w', encoding='utf-8') as f:
    f.writelines(lines)

# Validate
import json
with open('notebooks/modeling.ipynb', 'r', encoding='utf-8') as f:
    json.load(f)

print('Success')
