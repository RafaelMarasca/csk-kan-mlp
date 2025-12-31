"""
Remove ALL normalization code from the notebook.
User wants absolutely NO normalization - not even as an option.
"""

import json

# Load the notebook
with open('kan_experiments_refactored.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find and modify the load_data function cell
for cell in notebook['cells']:
    if cell['cell_type'] == 'code' and 'source' in cell:
        source = ''.join(cell['source'])
        
        # Remove normalize parameter from load_data function
        if 'def load_data' in source and 'normalize' in source:
            # Replace the function with one WITHOUT any normalization
            cell['source'] = [
                "def load_data(filepath, feature_cols, target_col='Symbol', \n",
                "              train_size=64, test_size=None):\n",
                "    \"\"\"\n",
                "    Load and split data - NO NORMALIZATION.\n",
                "    \n",
                "    Args:\n",
                "        filepath: Path to CSV file\n",
                "        feature_cols: List of feature column names\n",
                "        target_col: Target column name\n",
                "        train_size: Number of training samples\n",
                "        test_size: Number of test samples (None = use remaining)\n",
                "    \n",
                "    Returns:\n",
                "        X_train, X_test, y_train, y_test, num_classes\n",
                "    \"\"\"\n",
                "    df = pd.read_csv(filepath)\n",
                "    X = df[feature_cols].values\n",
                "    y = df[target_col].values\n",
                "    \n",
                "    if test_size is not None:\n",
                "        X_train, X_rest, y_train, y_rest = train_test_split(\n",
                "            X, y, train_size=train_size, random_state=SEED, stratify=y)\n",
                "        if len(X_rest) >= test_size:\n",
                "            X_test, _, y_test, _ = train_test_split(\n",
                "                X_rest, y_rest, train_size=test_size, random_state=SEED, stratify=y_rest)\n",
                "        else:\n",
                "            X_test, y_test = X_rest, y_rest\n",
                "    else:\n",
                "        X_train, X_test, y_train, y_test = train_test_split(\n",
                "            X, y, train_size=train_size, random_state=SEED, stratify=y)\n",
                "    \n",
                "    # Convert to tensors - NO NORMALIZATION\n",
                "    X_train = torch.FloatTensor(X_train).unsqueeze(-1).to(device)\n",
                "    y_train = torch.LongTensor(y_train).to(device)\n",
                "    X_test = torch.FloatTensor(X_test).unsqueeze(-1).to(device)\n",
                "    y_test = torch.LongTensor(y_test).to(device)\n",
                "    \n",
                "    return X_train, X_test, y_train, y_test, len(np.unique(y))"
            ]
        
        # Remove normalize from CONFIG
        if 'CONFIG = {' in source and 'normalize' in source:
            cell['source'] = [
                "# Output directory\n",
                "OUTPUT_DIR = 'results_refactored'\n",
                "os.makedirs(OUTPUT_DIR, exist_ok=True)\n",
                "print(f'Results directory: {OUTPUT_DIR}')\n",
                "\n",
                "# Training configuration\n",
                "CONFIG = {\n",
                "    'search_epochs': 50,\n",
                "    'final_epochs': 200,\n",
                "    'n_repeats': 3,\n",
                "}\n",
                "\n",
                "print('\\nConfiguration:')\n",
                "for k, v in CONFIG.items():\n",
                "    print(f'  {k}: {v}')"
            ]
        
        # Remove normalize parameter from load_data calls
        if 'load_data(' in source and 'normalize=' in source:
            # Remove the normalize parameter from function calls
            new_source = []
            for line in cell['source']:
                if 'normalize=' not in line:
                    new_source.append(line)
                elif 'load_data(' in line or 'filename,' in line or 'cols,' in line:
                    # Keep the line but remove normalize parameter
                    new_source.append(line.replace(', \n            normalize=CONFIG[\'normalize\']', '').replace(',\n            normalize=CONFIG[\'normalize\']', ''))
            cell['source'] = new_source

# Update markdown cells
for cell in notebook['cells']:
    if cell['cell_type'] == 'markdown' and 'source' in cell:
        source = ''.join(cell['source'])
        if '## 2.1 Data Loading' in source:
            cell['source'] = ["## 2.1 Data Loading - NO NORMALIZATION"]

# Save the modified notebook
with open('kan_experiments_refactored.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("Notebook updated - ALL normalization code removed!")
print("- Removed 'normalize' parameter from load_data function")
print("- Removed 'normalize' from CONFIG dictionary")
print("- Removed all normalization-related code")
print("- Updated all load_data calls to remove normalize parameter")
