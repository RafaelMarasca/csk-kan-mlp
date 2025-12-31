"""
Script to fix cell 32 in kan_experiments_refactored.ipynb
This cell should contain the sample efficiency experiment code, not the CONFIG code.
"""

import json

def fix_cell32():
    """Restore cell 32 with the correct sample efficiency experiment code."""
    
    # Read the notebook
    with open('kan_experiments_refactored.ipynb', 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # The correct content for cell 32 (sample efficiency experiment)
    correct_cell32_content = [
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
        "        # FNN - Use dataset-specific configuration\n",
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
    
    # Update cell 32
    notebook['cells'][32]['source'] = correct_cell32_content
    
    # Save the updated notebook
    with open('kan_experiments_refactored.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print("="*80)
    print("Cell 32 fixed successfully!")
    print("="*80)
    print("\nCell 32 now contains the sample efficiency experiment with:")
    print("- Correct FNN configuration using FNN_CONFIGS[filename]")
    print("- Training with CONFIG['final_epochs'] (1000 epochs)")

if __name__ == "__main__":
    fix_cell32()
