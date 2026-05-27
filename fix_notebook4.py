# Read the backup file
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
    # We want to change: "    },\n    "    'Random Forest': {\n
    # to:            "    },\n    'Random Forest': {\n
    # So we remove the double quote and four spaces after the newline? Actually, we remove the substring: '"    "    ' (double quote, four spaces, double quote, four spaces) and replace it with '    ' (four spaces) but note that we already have the four spaces from the newline? Let's do a simple replace.
    lines[231] = lines[231].replace('"    },\n    "    ', '"    },\n    ')

# Write the fixed file
with open('notebooks/modeling.ipynb', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('Fixed notebook')
