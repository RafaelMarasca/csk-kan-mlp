"""
Fix noise analysis issues:
1. Restore proper features for FNN and KAN (not x,y)
2. Fix noise level sign (should be positive)
3. Fix FNN optimizer (should be LBFGS)
"""

import json

def fix_noise_analysis():
    """Fix the noise analysis implementation."""
    
    # Read the notebook
    with open('kan_experiments_refactored.ipynb', 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Find and fix noise analysis cell
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code' and any('Noise analysis experiment' in line for line in cell.get('source', [])):
            print(f"Fixing noise analysis in cell {i}")
            
            # Recreate the correct noise analysis code
            cell['source'] = [
                "# Noise analysis experiment\n",
                "import glob\n",
                "\n",
                "# Find all noise level files\n",
                "noise_files_4csk = sorted([f for f in glob.glob('data_4csk_m*dB.csv') if 'm1000dB' not in f])\n",
                "noise_files_8csk = sorted([f for f in glob.glob('data_8csk_m*dB.csv') if 'm1000dB' not in f])\n",
                "\n",
                "noise_results = {\n",
                "    'Dataset': [], 'Noise_Level_dB': [], 'Model': [], 'Mean_SER': [], 'Std_SER': []\n",
                "}\n",
                "\n",
                "# Use local seed\n",
                "base_seed = 42\n",
                "\n",
                "print('Noise Analysis Experiment')\n",
                "print('='*80)\n",
                "\n",
                "for dataset_type, noise_files, ref_file, base_file in [\n",
                "    ('4CSK', noise_files_4csk, 'data_4csk_m1000dB.csv', 'data_4csk.csv'),\n",
                "    ('8CSK', noise_files_8csk, 'data_8csk_m1000dB.csv', 'data_8csk.csv')\n",
                "]:\n",
                "    print(f'\\nDataset: {dataset_type}')\n",
                "    \n",
                "    # Load reference symbols for MLE (from noiseless data)\n",
                "    ref_df = pd.read_csv(ref_file)\n",
                "    reference_symbols = torch.FloatTensor(\n",
                "        ref_df.groupby('Symbol')[['x', 'y']].mean().values\n",
                "    ).to(device)\n",
                "    \n",
                "    # Get configurations - USE PROPER FEATURES\n",
                "    if dataset_type == '4CSK':\n",
                "        fnn_cols = ['Vr', 'Vg', 'Vb']  # FNN-1: voltage features\n",
                "        kan_cols = best_configs[base_file]['columns']  # Best KAN features\n",
                "        params = best_configs[base_file]['params']\n",
                "        fnn_hidden = FNN_CONFIGS[base_file]\n",
                "    else:  # 8CSK\n",
                "        fnn_cols = ['X_ne', 'Y_ne', 'Z_ne']  # FNN-3: X_ne, Y_ne, Z_ne\n",
                "        kan_cols = best_configs[base_file]['columns']  # Best KAN features\n",
                "        params = best_configs[base_file]['params']\n",
                "        fnn_hidden = FNN_CONFIGS[base_file]\n",
                "    \n",
                "    for noise_file in noise_files:\n",
                "        # Extract noise level from filename (e.g., 'm20dB' -> 20) - POSITIVE\n",
                "        noise_level = int(noise_file.split('_m')[1].replace('dB.csv', ''))\n",
                "        print(f'  Noise level: {noise_level}dB')\n",
                "        \n",
                "        # Test each model with 10 independent runs\n",
                "        for model_type in ['FNN', 'KAN', 'MLE']:\n",
                "            sers = []\n",
                "            \n",
                "            for run_idx in range(CONFIG['n_repeats']):\n",
                "                # Set seed for this run\n",
                "                if run_idx > 0:\n",
                "                    temp_seed = base_seed + run_idx\n",
                "                    np.random.seed(temp_seed)\n",
                "                    torch.manual_seed(temp_seed)\n",
                "                \n",
                "                if model_type == 'MLE':\n",
                "                    # MLE only needs x,y coordinates\n",
                "                    df = pd.read_csv(noise_file)\n",
                "                    # Use all data for testing MLE\n",
                "                    X_test_xy = torch.FloatTensor(df[['x', 'y']].values).to(device)\n",
                "                    y_test = torch.LongTensor(df['Symbol'].values).to(device)\n",
                "                    \n",
                "                    preds = mle_predict(X_test_xy, reference_symbols)\n",
                "                    ser = (preds != y_test).float().mean().item()\n",
                "                    \n",
                "                elif model_type == 'FNN':\n",
                "                    # Load data with FNN features (NOT x,y)\n",
                "                    X_train, X_test, y_train, y_test, num_classes = load_data(\n",
                "                        noise_file, fnn_cols, train_size=64, test_size=1024\n",
                "                    )\n",
                "                    \n",
                "                    model = FNN(len(fnn_cols), fnn_hidden, num_classes).to(device)\n",
                "                    best_ser, _, _ = train_model(\n",
                "                        model, X_train, y_train, X_test, y_test,\n",
                "                        epochs=CONFIG['final_epochs'],\n",
                "                        lr=params['lr'],\n",
                "                        optimizer_type='lbfgs'  # LBFGS for FNN\n",
                "                    )\n",
                "                    ser = best_ser\n",
                "                    \n",
                "                else:  # KAN\n",
                "                    # Load data with KAN features (NOT x,y)\n",
                "                    if run_idx == 0:\n",
                "                        X_train, X_test, y_train, y_test, num_classes = load_data(\n",
                "                            noise_file, kan_cols, train_size=64, test_size=1024\n",
                "                        )\n",
                "                        layers = [len(kan_cols)] + params['hidden_layers'] + [num_classes]\n",
                "                    else:\n",
                "                        X_train, X_test, y_train, y_test, _ = load_data(\n",
                "                            noise_file, kan_cols, train_size=64, test_size=1024\n",
                "                        )\n",
                "                    \n",
                "                    model = ReLUKAN(layers, grid=params['grid'], k=params['k']).to(device)\n",
                "                    best_ser, _, _ = train_model(\n",
                "                        model, X_train, y_train, X_test, y_test,\n",
                "                        epochs=CONFIG['final_epochs'],\n",
                "                        lr=params['lr'],\n",
                "                        optimizer_type='adam'  # Adam for KAN\n",
                "                    )\n",
                "                    ser = best_ser\n",
                "                \n",
                "                # Restore seed\n",
                "                if run_idx > 0:\n",
                "                    np.random.seed(base_seed)\n",
                "                    torch.manual_seed(base_seed)\n",
                "                \n",
                "                sers.append(ser)\n",
                "            \n",
                "            noise_results['Dataset'].append(dataset_type)\n",
                "            noise_results['Noise_Level_dB'].append(noise_level)\n",
                "            noise_results['Model'].append(model_type)\n",
                "            noise_results['Mean_SER'].append(np.mean(sers))\n",
                "            noise_results['Std_SER'].append(np.std(sers))\n",
                "\n",
                "df_noise = pd.DataFrame(noise_results)\n",
                "df_noise.to_csv(os.path.join(OUTPUT_DIR, 'noise_analysis_results.csv'), index=False)\n",
                "print(f'\\nResults saved to {OUTPUT_DIR}/noise_analysis_results.csv')"
            ]
            break
    
    # Save the updated notebook
    with open('kan_experiments_refactored.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("Noise analysis fixed!")
    print("="*80)
    print("\nFixes applied:")
    print("1. FNN now uses proper features:")
    print("   - 4CSK: Vr, Vg, Vb (voltage)")
    print("   - 8CSK: X_ne, Y_ne, Z_ne")
    print("2. KAN now uses best features from feature selection")
    print("3. Noise level sign corrected (positive values)")
    print("4. FNN optimizer corrected to LBFGS")
    print("\nNow FNN, KAN, and MLE use different features as intended!")

if __name__ == "__main__":
    fix_noise_analysis()
