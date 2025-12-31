"""
Script to fix the feature selection experiment in kan_experiments_refactored.ipynb
Move data loading inside the repeat loop so each run uses different random splits.
"""

import json

def fix_feature_selection():
    """Fix cell 24 to load data inside the repeat loop."""
    
    # Read the notebook
    with open('kan_experiments_refactored.ipynb', 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Find cell 24 (feature selection experiment)
    # Based on the structure, it should be around index 24
    cell_index = None
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code' and any('Storage for results' in line for line in cell.get('source', [])):
            cell_index = i
            print(f"Found feature selection cell at index {i}")
            break
    
    if cell_index is None:
        print("ERROR: Could not find feature selection cell")
        return
    
    # The corrected cell content with data loading inside the repeat loop
    corrected_content = [
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
        "for filename in ['data_4csk.csv', 'data_8csk.csv']:\n",
        "    print(f'\\nDataset: {filename}')\n",
        "    best_mean_ser = 1.0\n",
        "    best_config = None\n",
        "    \n",
        "    for feat_name, cols in FEATURE_SETS.items():\n",
        "        for params in PARAM_COMBINATIONS:\n",
        "            sers = []\n",
        "            layers = [len(cols)] + params['hidden_layers'] + [num_classes]\n",
        "            \n",
        "            # Multiple runs for statistics - EACH WITH DIFFERENT DATA SPLIT\n",
        "            for run_idx in range(CONFIG['n_repeats']):\n",
        "                # Load data with different random seed for each run\n",
        "                # Note: We get num_classes from first load, it's the same for all splits\n",
        "                if run_idx == 0:\n",
        "                    X_train, X_test, y_train, y_test, num_classes = load_data(\n",
        "                        filename, cols, train_size=64, test_size=64,\n",
        "                    )\n",
        "                    layers = [len(cols)] + params['hidden_layers'] + [num_classes]\n",
        "                else:\n",
        "                    # Use different random state for different splits\n",
        "                    # Temporarily change the seed\n",
        "                    temp_seed = SEED + run_idx\n",
        "                    np.random.seed(temp_seed)\n",
        "                    torch.manual_seed(temp_seed)\n",
        "                    X_train, X_test, y_train, y_test, _ = load_data(\n",
        "                        filename, cols, train_size=64, test_size=64,\n",
        "                    )\n",
        "                    # Restore original seed\n",
        "                    np.random.seed(SEED)\n",
        "                    torch.manual_seed(SEED)\n",
        "                \n",
        "                model = ReLUKAN(layers, grid=params['grid'], k=params['k']).to(device)\n",
        "                best_ser, _, _ = train_model(\n",
        "                    model, X_train, y_train, X_test, y_test,\n",
        "                    epochs=CONFIG['search_epochs'],\n",
        "                    lr=params['lr'],\n",
        "                    weight_decay=params['weight_decay']\n",
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
    
    # Update the cell
    notebook['cells'][cell_index]['source'] = corrected_content
    
    # Save the updated notebook
    with open('kan_experiments_refactored.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("Feature selection cell updated successfully!")
    print("="*80)
    print("\nChanges made:")
    print("- Data loading moved inside the repeat loop")
    print("- Each repeat uses a different random seed (SEED + run_idx)")
    print("- This ensures each run has different train/test splits")
    print("- More robust statistics for model performance evaluation")

if __name__ == "__main__":
    fix_feature_selection()
