import json

with open('notebooks/modeling.ipynb.backup', 'r', encoding='utf-8') as f:
    lines = f.readlines()

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

# Fix the problematic line in param_grids (line 232 in 1-indexed -> index 231)
if len(lines) > 231:
    line = lines[231]
    # We want to change: "    },\n    "    'Random Forest': {\n
    # to:            "    },\n    'Random Forest': {\n
    # So we remove the double quote and four spaces after the newline? Actually, after the newline we have four spaces, then a double quote, then four spaces.
    # We want to remove the double quote and the four spaces that come after it? Let's try to remove the substring: '"    "    ' (double quote, four spaces, double quote, four spaces)
    # and replace it with '''    ' (four spaces).
    # But note: the line might have changed due to the import fixes? We assume not.
    # Let's do a direct string replacement for the known pattern.
    if '"    },\n    "    \'Random Forest\': {' in line:
        lines[231] = line.replace('"    },\n    "    \'Random Forest\': {', '"    },\n    \'Random Forest\': {')
    else:
        # Try another pattern: maybe the escaping is different
        if '"    },\n    "    \'Random Forest\': {' in line:
            lines[231] = line.replace('"    },\n    "    \'Random Forest\': {', '"    },\n    \'Random Forest\': {')

with open('notebooks/modeling.ipynb', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('Fixed notebook')
