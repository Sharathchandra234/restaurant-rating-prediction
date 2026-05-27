import json

# Read the backup file as text
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

# Now we need to fix the param_grids cell.
# We'll find the cell that contains the param_grids and fix its source.
# Instead of line-by-line, let's parse as JSON, fix the cell, then write back.
# But we already have the lines, so let's try to fix the problematic line by splitting it.
# We know that the problematic line is the one that contains: \"    },\n    \"    'Random Forest': {\n
# We'll change it to two lines: \"    },\n\" and \"    'Random Forest': {\n\"
# But we need to do this in the list of lines.

# First, let's join the lines and then split again? Instead, let's work with the list.
# We'll find the index of the line that contains the pattern.
for i, line in enumerate(lines):
    if '\n    \"    ' in line and \"'Random Forest'\" in line:
        # This is the problematic line.
        # We want to split it into two lines:
        # Part 1: everything up to and including the newline after the comma? Actually, we want:
        #   line1 = \"    },\n\"   (but note that the line already has a \n in the middle?)
        # Let's examine the line: it is: \"    },\n    \"    'Random Forest': {\n\"
        # We want to change it to:
        #   line1 = \"    },\n\"
        #   line2 = \"    'Random Forest': {\n\"
        # But note that the line currently ends with \n\", so we have to keep that.
        # Actually, the line as stored in the .ipynb file is a string that ends with \n\" (because each line in the source array ends with \n and then a comma or nothing, and then the quote).
        # Let's just replace the line with two lines.
        # We'll remove the current line and insert two lines in its place.
        # The current line is: lines[i]
        # We want to replace it with:
        #   \"    },\n\"
        #   \"    'Random Forest': {\n\"
        # But note that the original line already has a \n in the middle? Actually, the string contains the two characters backslash and n as part of the escape sequence.
        # Let's just do:
        new_line1 = \"    },\\n\"\n        new_line2 = \"    'Random Forest': {\\n\"\n        # Replace lines[i] with new_line1 and insert new_line2 after it.
        lines[i] = new_line1
        lines.insert(i+1, new_line2)
        break

# Write the fixed file
with open('notebooks/modeling.ipynb', 'w', encoding='utf-8') as f:
    f.writelines(lines)

# Validate by trying to load as JSON
try:
    with open('notebooks/modeling.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    print('Success: Notebook JSON is valid')
except json.JSONDecodeError as e:
    print(f'Error: {e}')
    # If there's still an error, let's try to rebuild the notebook from scratch using the backup but fixing the known issues.
    # We'll do a more robust fix: reconstruct the entire notebook by reading the backup and fixing the known problematic parts.
    # But for now, let's output the problematic lines.
    lines = open('notebooks/modeling.ipynb', 'r', encoding='utf-8').readlines()
    print(f'Line 232 (1-indexed): {repr(lines[231]) if len(lines) > 231 else \"Line does not exist\"}')
    exit(1)
