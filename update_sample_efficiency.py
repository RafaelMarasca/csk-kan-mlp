"""
Update sample efficiency experiment to use 10 independent runs with different data splits.
1. Update CONFIG to set n_repeats = 10
2. Update sample efficiency cell (cell 32) to load data inside the loop like feature selection
"""

import json

def update_sample_efficiency():
    """Update sample efficiency to use 10 runs with different data splits."""
    
    # Read the notebook
    with open('kan_experiments_refactored.ipynb', 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Step 1: Update CONFIG cell to set n_repeats = 10
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code' and any('FNN_CONFIGS' in line for line in cell.get('source', [])):
            print(f"Updating CONFIG in cell {i}")
            cell['source'] = [
                "# Output directory\n",
                "OUTPUT_DIR = 'results_refactored'\n",
                "os.makedirs(OUTPUT_DIR, exist_ok=True)\n",
                "print(f'Results directory: {OUTPUT_DIR}')\n",
                "\n",
                "# Training configuration\n",
                "CONFIG = {\n",
                "    'search_epochs': 50,\n",
                "    'final_epochs': 1000,\n",
                "    'n_repeats': 10,  # Updated to 10 independent runs\n",
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
            break
    
    # Step 2: Update sample efficiency cell to load data inside the loop
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code' and any('Sample efficiency experiment' in line for line in cell.get('source', [])):
            print(f"Updating sample efficiency cell {i}")
            cell['source'] = [
                "# Sample efficiency experiment\n",
                "train_sizes = [8, 16, 32, 64, 128, 256, 384, 512, 640, 768, 896, 1024]\n",
                "test_size = 1024\n",
                "\n",
                "efficiency_results = {\n",
                "    'Dataset': [], 'Train_Size': [], 'Model': [], 'Mean_SER': [], 'Std_SER': []\n",
                "}\n",
                "\n",
                "# Use local seed for reproducibility\n",
                "base_seed = 42\n",
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
                "        # KAN - 10 independent runs with different data splits\n",
                "        kan_sers = []\n",
                "        for run_idx in range(CONFIG['n_repeats']):\n",
                "            # Load data with different random seed for each run\n",
                "            if run_idx == 0:\n",
                "                X_train, X_test, y_train, y_test, num_classes = load_data(\n",
                "                    filename, cols, train_size=train_size, test_size=test_size,\n",
                "                )\n",
                "                layers = [len(cols)] + params['hidden_layers'] + [num_classes]\n",
                "            else:\n",
                "                temp_seed = base_seed + run_idx\n",
                "                np.random.seed(temp_seed)\n",
                "                torch.manual_seed(temp_seed)\n",
                "                X_train, X_test, y_train, y_test, _ = load_data(\n",
                "                    filename, cols, train_size=train_size, test_size=test_size,\n",
                "                )\n",
                "                np.random.seed(base_seed)\n",
                "                torch.manual_seed(base_seed)\n",
                "            \n",
                "            model = ReLUKAN(layers, grid=params['grid'], k=params['k']).to(device)\n",
                "            best_ser, _, _ = train_model(\n",
                "                model, X_train, y_train, X_test, y_test,\n",
                "                epochs=CONFIG['final_epochs'],\n",
                "                lr=params['lr']\n",
                "            )\n",
                "            kan_sers.append(best_ser)\n",
                "        \n",
                "        # FNN - 10 independent runs with different data splits\n",
                "        fnn_hidden_layers = FNN_CONFIGS[filename]\n",
                "        fnn_sers = []\n",
                "        for run_idx in range(CONFIG['n_repeats']):\n",
                "            # Load data with different random seed for each run\n",
                "            if run_idx == 0:\n",
                "                X_train, X_test, y_train, y_test, num_classes = load_data(\n",
                "                    filename, cols, train_size=train_size, test_size=test_size,\n",
                "                )\n",
                "            else:\n",
                "                temp_seed = base_seed + run_idx\n",
                "                np.random.seed(temp_seed)\n",
                "                torch.manual_seed(temp_seed)\n",
                "                X_train, X_test, y_train, y_test, _ = load_data(\n",
                "                    filename, cols, train_size=train_size, test_size=test_size,\n",
                "                )\n",
                "                np.random.seed(base_seed)\n",
                "                torch.manual_seed(base_seed)\n",
                "            \n",
                "            model = FNN(len(cols), fnn_hidden_layers, num_classes).to(device)\n",
                "            best_ser, _, _ = train_model(\n",
                "                model, X_train, y_train, X_test, y_test,\n",
                "                epochs=CONFIG['final_epochs'],\n",
                "                lr=params['lr']\n",
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
            break
    
    # Save the updated notebook
    with open('kan_experiments_refactored.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("Sample efficiency updated successfully!")
    print("="*80)
    print("\nChanges made:")
    print("- CONFIG['n_repeats'] = 10 (updated from 3)")
    print("- Sample efficiency now loads data inside the loop for each run")
    print("- Each of the 10 runs uses a different random seed (base_seed + run_idx)")
    print("- Both KAN and FNN get 10 independent runs with different data splits")

if __name__ == "__main__":
    update_sample_efficiency()
