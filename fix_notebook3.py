import json

with open('notebooks/modeling.ipynb.backup', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Fix imports
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'from data.load_data import load_dataset, validate_dataset' in source:
            new_source = []
            for line in cell['source']:
                stripped = line.strip()
                if stripped == 'from data.load_data import load_dataset, validate_dataset':
                    new_source.append('from src.data.load_data import load_dataset, validate_dataset\n')
                elif stripped == 'from features.feature_engineering import engineer_features':
                    new_source.append('from src.features.feature_engineering import engineer_features\n')
                elif stripped == 'from preprocessing.preprocessor import preprocess_data':
                    new_source.append('from src.preprocessing.preprocessor import preprocess_data\n')
                elif stripped == 'from models.train_model import train_models, hyperparameter_tuning, save_model':
                    new_source.append('from src.models.train_model import train_models, hyperparameter_tuning, save_model\n')
                else:
                    new_source.append(line)
            cell['source'] = new_source
            break

# Fix the param_grids line
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if "'Random Forest': {" in source:
            new_source = []
            for line in cell['source']:
                # We want to change lines that contain the pattern: '"    },\n    "    '
                # to: '"    },\n    '
                if '"    },\n    "    ' in line:
                    # Replace the first occurrence of '"    },\n    "    ' with '"    },\n    '
                    new_line = line.replace('"    },\n    "    ', '"    },\n    ', 1)
                    new_source.append(new_line)
                else:
                    new_source.append(line)
            cell['source'] = new_source
            break

with open('notebooks/modeling.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Fixed notebook')
