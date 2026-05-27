import json

# Read the backup file
with open('notebooks/modeling.ipynb.backup', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix imports
content = content.replace('from data.load_data import load_dataset, validate_dataset', 'from src.data.load_data import load_dataset, validate_dataset')
content = content.replace('from features.feature_engineering import engineer_features', 'from src.features.feature_engineering import engineer_features')
content = content.replace('from preprocessing.preprocessor import preprocess_data', 'from src.preprocessing.preprocessor import preprocess_data')
content = content.replace('from models.train_model import train_models, hyperparameter_tuning, save_model', 'from src.models.train_model import train_models, hyperparameter_tuning, save_model')

# Fix the param_grids line: replace the pattern that has an extra quote and four spaces after the newline
# We want to change: '"    },\n    "    ' -> '"    },\n    '
content = content.replace('\"    },\\n    \"    ', '\"    },\\n    ')

# Write the fixed file
with open('notebooks/modeling.ipynb', 'w', encoding='utf-8') as f:
    f.write(content)

# Validate the JSON
try:
    with open('notebooks/modeling.ipynb', 'r', encoding='utf-8') as f:
        json.load(f)
    print('Success: Notebook JSON is valid')
except json.JSONDecodeError as e:
    print(f'Error: {e}')
    # Show the problematic line
    lines = content.split('\n')
    print(f'Line {e.lineno}: {lines[e.lineno-1]}')
