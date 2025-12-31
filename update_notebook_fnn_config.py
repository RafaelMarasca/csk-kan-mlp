"""
Script to update the FNN configuration in kan_experiments_refactored.ipynb
Updates:
1. FNN configurations: 4CSK uses [10], 8CSK uses [100]
2. Final epochs: 1000 for both datasets
3. Fixes FNN model instantiation to use correct architecture
"""

import json

def update_notebook():
    """Update the notebook with corrected FNN configurations and epochs."""
    
    # Read the notebook
    with open('kan_experiments_refactored.ipynb', 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Find and update the configuration cell (around line 86-115)
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code' and any('FNN_CONFIGS' in line for line in cell.get('source', [])):
            print(f"Found FNN_CONFIGS cell at index {i}")
            # Update the cell with correct configuration
            cell['source'] = [
                "# Output directory\n",
                "OUTPUT_DIR = 'results_refactored'\n",
                "os.makedirs(OUTPUT_DIR, exist_ok=True)\n",
                "print(f'Results directory: {OUTPUT_DIR}')\n",
                "\n",
                "# Training configuration\n",
                "CONFIG = {\n",
                "    'search_epochs': 50,\n",
                "    'final_epochs': 1000,  # Updated to 1000 epochs\n",
                "    'n_repeats': 3,\n",
                "}\n",
                "\n",
                "# FNN configurations per dataset\n",
                "FNN_CONFIGS = {\n",
                "    'data_4csk.csv': [10],      # FNN-1: 1 hidden layer, 10 neurons\n",
                "    'data_8csk.csv': [100],     # FNN-3: 1 hidden layer, 100 neurons\n",
                "}\n",
                "\n",
                "print('\\nConfiguration:')\n",
                "for k, v in CONFIG.items():\n",
                "    print(f'  {k}: {v}')\n",
                "print('\\nFNN Configurations:')\n",
                "for dataset, layers in FNN_CONFIGS.items():\n",
                "    print(f'  {dataset}: {layers}')"
            ]
            print("Updated CONFIG and FNN_CONFIGS")
    
    # Find and update the sample efficiency cell (around line 594-661)
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code' and any('Sample efficiency experiment' in line for line in cell.get('source', [])):
            print(f"Found sample efficiency cell at index {i}")
            # Update the cell to fix FNN instantiation
            cell['source'] = [
                "# Sample efficiency experiment\n",
                "train_sizes = [8, 16, 32, 64, 128, 256, 384, 512, 640, 768, 896, 1024]\n",
                "test_size = 1024\n",
                "\n",
                "efficiency_results = {\n",
                "    'Dataset': [], 'Train_Size': [], 'Model': [], 'Mean_SER': [], 'Std_SER': []\n",
                "}\n",
                "\n",
                "print('Sample Efficiency Experiment')\n",
                "print('='*80)\n",
                "\n",
                "for filename in ['data_4csk.csv', 'data_8csk.csv']:\n",
                "    print(f'\\nDataset: {filename}')\n",
                "    \n",
                "    # Use best configuration from previous experiment\n",
                "    best_cfg = best_configs[filename]\n",
                "    cols = best_cfg['columns']\n",
                "    params = best_cfg['params']\n",
                "    \n",
                "    for train_size in train_sizes:\n",
                "        print(f'  Train size: {train_size}')\n",
                "        \n",
                "        X_train, X_test, y_train, y_test, num_classes = load_data(\n",
                "            filename, cols, train_size=train_size, test_size=test_size,\n",
                "        )\n",
                "        \n",
                "        # KAN\n",
                "        kan_sers = []\n",
                "        for _ in range(CONFIG['n_repeats']):\n",
                "            layers = [len(cols)] + params['hidden_layers'] + [num_classes]\n",
                "            model = ReLUKAN(layers, grid=params['grid'], k=params['k']).to(device)\n",
                "            best_ser, _, _ = train_model(\n",
                "                model, X_train, y_train, X_test, y_test,\n",
                "                epochs=CONFIG['final_epochs'],\n",
                "                lr=params['lr'], weight_decay=params['weight_decay']\n",
                "            )\n",
                "            kan_sers.append(best_ser)\n",
                "        \n",
                "        # FNN - CORRECTED to use dataset-specific configuration\n",
                "        fnn_hidden_layers = FNN_CONFIGS[filename]\n",
                "        fnn_sers = []\n",
                "        for _ in range(CONFIG['n_repeats']):\n",
                "            model = FNN(len(cols), fnn_hidden_layers, num_classes).to(device)\n",
                "            best_ser, _, _ = train_model(\n",
                "                model, X_train, y_train, X_test, y_test,\n",
                "                epochs=CONFIG['final_epochs'],\n",
                "                lr=params['lr'], weight_decay=params['weight_decay']\n",
                "            )\n",
                "            fnn_sers.append(best_ser)\n",
                "        \n",
                "        ds_name = filename.replace('.csv', '').replace('data_', '').upper()\n",
                "        \n",
                "        efficiency_results['Dataset'].extend([ds_name, ds_name])\n",
                "        efficiency_results['Train_Size'].extend([train_size, train_size])\n",
                "        efficiency_results['Model'].extend(['KAN', 'FNN'])\n",
                "        efficiency_results['Mean_SER'].extend([np.mean(kan_sers), np.mean(fnn_sers)])\n",
                "        efficiency_results['Std_SER'].extend([np.std(kan_sers), np.std(fnn_sers)])\n",
                "\n",
                "df_efficiency = pd.DataFrame(efficiency_results)\n",
                "df_efficiency.to_csv(os.path.join(OUTPUT_DIR, 'sample_efficiency_results.csv'), index=False)\n",
                "print(f'\\nResults saved to {OUTPUT_DIR}/sample_efficiency_results.csv')"
            ]
            print("Updated sample efficiency experiment with correct FNN configuration")
    
    # Save the updated notebook
    with open('kan_experiments_refactored.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("Notebook updated successfully!")
    print("="*80)
    print("\nChanges made:")
    print("1. Updated final_epochs to 1000")
    print("2. FNN configurations:")
    print("   - data_4csk.csv: [10] (1 hidden layer, 10 neurons)")
    print("   - data_8csk.csv: [100] (1 hidden layer, 100 neurons)")
    print("3. Fixed FNN model instantiation to use correct architecture")

if __name__ == "__main__":
    update_notebook()
