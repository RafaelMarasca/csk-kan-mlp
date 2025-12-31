"""
Update cell 24 (feature selection) to pass random_state to load_data
to ensure 10 independent runs with different data splits.
"""

import json

def update_feature_selection_random_state():
    """Update feature selection to use random_state in load_data."""
    
    # Read the notebook
    with open('kan_experiments_refactored.ipynb', 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # First, update load_data function to accept random_state
    print("Step 1: Updating load_data function to accept random_state parameter...")
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code' and any('def load_data' in line for line in cell.get('source', [])):
            print(f"  Updating load_data in cell {i}")
            cell['source'] = [
                "def load_data(filepath, feature_cols, target_col='Symbol', \n",
                "              train_size=64, test_size=None, random_state=None):\n",
                "    \"\"\"\n",
                "    Load and split data - NO NORMALIZATION.\n",
                "    \n",
                "    Args:\n",
                "        filepath: Path to CSV file\n",
                "        feature_cols: List of feature column names\n",
                "        target_col: Target column name\n",
                "        train_size: Number of training samples\n",
                "        test_size: Number of test samples (None = use remaining)\n",
                "        random_state: Random state for reproducibility\n",
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
                "            X, y, train_size=train_size, random_state=random_state, stratify=y)\n",
                "        if len(X_rest) >= test_size:\n",
                "            X_test, _, y_test, _ = train_test_split(\n",
                "                X_rest, y_rest, train_size=test_size, random_state=random_state, stratify=y_rest)\n",
                "        else:\n",
                "            X_test, y_test = X_rest, y_rest\n",
                "    else:\n",
                "        X_train, X_test, y_train, y_test = train_test_split(\n",
                "            X, y, train_size=train_size, random_state=random_state, stratify=y)\n",
                "    \n",
                "    # Convert to tensors - NO NORMALIZATION\n",
                "    X_train = torch.FloatTensor(X_train).unsqueeze(-1).to(device)\n",
                "    y_train = torch.LongTensor(y_train).to(device)\n",
                "    X_test = torch.FloatTensor(X_test).unsqueeze(-1).to(device)\n",
                "    y_test = torch.LongTensor(y_test).to(device)\n",
                "    \n",
                "    return X_train, X_test, y_train, y_test, len(np.unique(y))"
            ]
            break
    
    # Now update feature selection (cell 24/25) to use random_state
    print("\nStep 2: Updating feature selection to pass random_state...")
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code' and any('Storage for results' in line for line in cell.get('source', [])):
            print(f"  Updating feature selection in cell {i}")
            cell['source'] = [
                "# Storage for results\n",
                "results = {\n",
                "    'Dataset': [], 'Feature_Set': [], 'Hidden_Layers': [],\n",
                "    'Mean_SER': [], 'Std_SER': [], 'Best_SER': []\n",
                "}\n",
                "\n",
                "best_configs = {}  # Store best configuration for each dataset\n",
                "\n",
                "print('Starting Feature Selection and Hyperparameter Search...')\n",
                "print('='*80)\n",
                "\n",
                "# Use local seed for reproducibility\n",
                "base_seed = 42\n",
                "\n",
                "for filename in ['data_4csk.csv', 'data_8csk.csv']:\n",
                "    print(f'\\nDataset: {filename}')\n",
                "    best_mean_ser = 1.0\n",
                "    best_config = None\n",
                "    \n",
                "    for feat_name, cols in FEATURE_SETS.items():\n",
                "        for params in PARAM_COMBINATIONS:\n",
                "            sers = []\n",
                "            \n",
                "            # Multiple runs for statistics - EACH WITH DIFFERENT DATA SPLIT\n",
                "            for run_idx in range(CONFIG['n_repeats']):\n",
                "                # Use different random_state for each run\n",
                "                random_state = base_seed + run_idx\n",
                "                \n",
                "                # Load data with different random state\n",
                "                X_train, X_test, y_train, y_test, num_classes = load_data(\n",
                "                    filename, cols, train_size=64, test_size=64,\n",
                "                    random_state=random_state\n",
                "                )\n",
                "                \n",
                "                # Define layers after we know num_classes\n",
                "                if run_idx == 0:\n",
                "                    layers = [len(cols)] + params['hidden_layers'] + [num_classes]\n",
                "                \n",
                "                model = ReLUKAN(layers, grid=params['grid'], k=params['k']).to(device)\n",
                "                best_ser, _, _ = train_model(\n",
                "                    model, X_train, y_train, X_test, y_test,\n",
                "                    epochs=CONFIG['search_epochs'],\n",
                "                    lr=params['lr'],\n",
                "                    optimizer_type='adam'\n",
                "                )\n",
                "                sers.append(best_ser)\n",
                "            \n",
                "            mean_ser = np.mean(sers)\n",
                "            std_ser = np.std(sers)\n",
                "            min_ser = np.min(sers)\n",
                "            \n",
                "            results['Dataset'].append(filename.replace('.csv', '').replace('data_', '').upper())\n",
                "            results['Feature_Set'].append(feat_name)\n",
                "            results['Hidden_Layers'].append(str(params['hidden_layers']))\n",
                "            results['Mean_SER'].append(mean_ser)\n",
                "            results['Std_SER'].append(std_ser)\n",
                "            results['Best_SER'].append(min_ser)\n",
                "            \n",
                "            if mean_ser < best_mean_ser:\n",
                "                best_mean_ser = mean_ser\n",
                "                best_config = {\n",
                "                    'features': feat_name,\n",
                "                    'columns': cols,\n",
                "                    'params': params,\n",
                "                    'mean_ser': mean_ser,\n",
                "                    'std_ser': std_ser\n",
                "                }\n",
                "    \n",
                "    best_configs[filename] = best_config\n",
                "    print(f'  Best: {best_config[\"features\"]} | SER: {best_config[\"mean_ser\"]:.4f} +/- {best_config[\"std_ser\"]:.4f}')\n",
                "\n",
                "# Save results\n",
                "df_results = pd.DataFrame(results)\n",
                "df_results.to_csv(os.path.join(OUTPUT_DIR, 'feature_search_results.csv'), index=False)\n",
                "print(f'\\nResults saved to {OUTPUT_DIR}/feature_search_results.csv')"
            ]
            break
    
    # Save the updated notebook
    with open('kan_experiments_refactored.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("Feature selection updated!")
    print("="*80)
    print("\nChanges made:")
    print("1. Updated load_data function to accept random_state parameter")
    print("2. Feature selection now passes random_state=base_seed+run_idx")
    print("3. Each of the 10 runs uses a different random_state (42, 43, 44, ...)")
    print("4. Simplified code - no manual seed setting needed")
    print("5. Keeps current PARAM_GRID parameters unchanged")

if __name__ == "__main__":
    update_feature_selection_random_state()
