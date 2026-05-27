import json

# Load the backup notebook
with open('notebooks/modeling.ipynb.backup', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Fix 1: Correct the imports in the code cell
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'from data.load_data import load_dataset, validate_dataset' in source:
            # We found the cell with the imports
            new_source = []
            for line in cell['source']:
                if line.strip() == 'from data.load_data import load_dataset, validate_dataset':
                    new_source.append('from src.data.load_data import load_dataset, validate_dataset\n')
                elif line.strip() == 'from features.feature_engineering import engineer_features':
                    new_source.append('from src.features.feature_engineering import engineer_features\n')
                elif line.strip() == 'from preprocessing.preprocessor import preprocess_data':
                    new_source.append('from src.preprocessing.preprocessor import preprocess_data\n')
                elif line.strip() == 'from models.train_model import train_models, hyperparameter_tuning, save_model':
                    new_source.append('from src.models.train_model import train_models, hyperparameter_tuning, save_model\n')
                else:
                    new_source.append(line)
            cell['source'] = new_source
            break  # Assuming there's only one such cell

# Fix 2: Correct the param_grids cell
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if "'Random Forest': {" in source:
            # We found the cell with the param_grids
            # We'll replace the entire source with a corrected version.
            # But let's try to fix just the problematic line.
            new_source = []
            for line in cell['source']:
                # We are looking for the line that has the pattern: '    },\n    "    '
                # and we want to change it to: '    },\n    '
                # However, note that the line in the source array is a string that ends with a newline (if it's not the last line) and is enclosed in quotes.
                # We'll do a string replacement on the line.
                if '\n    \"    ' in line and "'Random Forest'" in line:
                    # Replace the problematic part
                    line = line.replace('\n    \"    ', '\n    ')
                new_source.append(line)
            cell['source'] = new_source
            break  # Assuming there's only one such cell

# Write the fixed notebook
with open('notebooks/modeling.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Notebook fixed successfully.')
