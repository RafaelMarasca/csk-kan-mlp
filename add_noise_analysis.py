"""
Comprehensive update to add noise analysis experiment:
1. Update sample efficiency to use specific features for FNNs
2. Add MLE predictor function
3. Add noise analysis experiment section
"""

import json

def add_noise_analysis():
    """Add complete noise analysis experiment to the notebook."""
    
    # Read the notebook
    with open('kan_experiments_refactored.ipynb', 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    print("Step 1: Updating sample efficiency to use specific features...")
    # Find and update sample efficiency cell (cell 32)
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code' and any('Sample efficiency experiment' in line for line in cell.get('source', [])):
            print(f"  Updating cell {i}: sample efficiency")
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
                "    # Use best KAN configuration from previous experiment\n",
                "    best_cfg = best_configs[filename]\n",
                "    kan_cols = best_cfg['columns']\n",
                "    params = best_cfg['params']\n",
                "    \n",
                "    # Set FNN-specific features\n",
                "    if filename == 'data_4csk.csv':\n",
                "        fnn_cols = ['Vr', 'Vg', 'Vb']  # FNN-1: voltage features\n",
                "        fnn_hidden_layers = FNN_CONFIGS[filename]\n",
                "    else:  # data_8csk.csv\n",
                "        fnn_cols = ['X_ne', 'Y_ne', 'Z_ne']  # FNN-3: X_ne, Y_ne, Z_ne\n",
                "        fnn_hidden_layers = FNN_CONFIGS[filename]\n",
                "    \n",
                "    for train_size in train_sizes:\n",
                "        print(f'  Train size: {train_size}')\n",
                "        \n",
                "        # KAN - 10 independent runs with different data splits\n",
                "        kan_sers = []\n",
                "        for run_idx in range(CONFIG['n_repeats']):\n",
                "            # Load data with KAN features\n",
                "            if run_idx == 0:\n",
                "                X_train, X_test, y_train, y_test, num_classes = load_data(\n",
                "                    filename, kan_cols, train_size=train_size, test_size=test_size,\n",
                "                )\n",
                "                layers = [len(kan_cols)] + params['hidden_layers'] + [num_classes]\n",
                "            else:\n",
                "                temp_seed = base_seed + run_idx\n",
                "                np.random.seed(temp_seed)\n",
                "                torch.manual_seed(temp_seed)\n",
                "                X_train, X_test, y_train, y_test, _ = load_data(\n",
                "                    filename, kan_cols, train_size=train_size, test_size=test_size,\n",
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
                "        fnn_sers = []\n",
                "        for run_idx in range(CONFIG['n_repeats']):\n",
                "            # Load data with FNN features\n",
                "            if run_idx == 0:\n",
                "                X_train, X_test, y_train, y_test, num_classes = load_data(\n",
                "                    filename, fnn_cols, train_size=train_size, test_size=test_size,\n",
                "                )\n",
                "            else:\n",
                "                temp_seed = base_seed + run_idx\n",
                "                np.random.seed(temp_seed)\n",
                "                torch.manual_seed(temp_seed)\n",
                "                X_train, X_test, y_train, y_test, _ = load_data(\n",
                "                    filename, fnn_cols, train_size=train_size, test_size=test_size,\n",
                "                )\n",
                "                np.random.seed(base_seed)\n",
                "                torch.manual_seed(base_seed)\n",
                "            \n",
                "            model = FNN(len(fnn_cols), fnn_hidden_layers, num_classes).to(device)\n",
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
    
    print("\nStep 2: Adding MLE predictor function...")
    # Find the plotting functions cell and add MLE predictor after it
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code' and any('def plot_confusion_matrix' in line for line in cell.get('source', [])):
            print(f"  Inserting MLE predictor after cell {i}")
            # Insert new cell after plotting functions
            new_cell = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "def mle_predict(X_test_xy, reference_symbols):\n",
                    "    \"\"\"\n",
                    "    MLE predictor using Euclidean distance.\n",
                    "    \n",
                    "    Args:\n",
                    "        X_test_xy: Test data (N, 2) with x, y coordinates\n",
                    "        reference_symbols: Reference symbol positions (num_classes, 2)\n",
                    "    \n",
                    "    Returns:\n",
                    "        predictions: Predicted class labels\n",
                    "    \"\"\"\n",
                    "    # Compute Euclidean distance to each reference symbol\n",
                    "    distances = torch.cdist(X_test_xy, reference_symbols)  # (N, num_classes)\n",
                    "    predictions = torch.argmin(distances, dim=1)\n",
                    "    return predictions\n",
                    "\n",
                    "print('MLE predictor function defined')"
                ]
            }
            notebook['cells'].insert(i + 1, new_cell)
            break
    
    print("\nStep 3: Adding noise analysis experiment section...")
    # Add noise analysis after sample efficiency plots
    # First, find where to insert (after sample efficiency plots)
    insert_idx = None
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'markdown' and any('Section 5' in line for line in cell.get('source', [])):
            insert_idx = i
            break
    
    if insert_idx:
        # Add markdown header
        notebook['cells'].insert(insert_idx, {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4.3 Noise Analysis: FNN vs KAN vs MLE\n",
                "\n",
                "Compare performance across different noise levels using:\n",
                "- **FNN-1** (4CSK with voltage features)\n",
                "- **FNN-3** (8CSK with X_ne, Y_ne, Z_ne)\n",
                "- **Best KAN** configurations\n",
                "- **MLE Predictor** (Euclidean distance)"
            ]
        })
        
        # Add noise analysis code cell
        notebook['cells'].insert(insert_idx + 1, {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": create_noise_analysis_code()
        })
        
        # Add noise analysis plots
        notebook['cells'].insert(insert_idx + 2, {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### 4.3.1 Noise Analysis Plots"]
        })
        
        notebook['cells'].insert(insert_idx + 3, {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": create_noise_plots_code()
        })
    
    # Save the updated notebook
    with open('kan_experiments_refactored.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("Noise analysis experiment added successfully!")
    print("="*80)
    print("\nChanges made:")
    print("1. Updated sample efficiency:")
    print("   - FNN-1 (4CSK) uses voltage features (Vr, Vg, Vb)")
    print("   - FNN-3 (8CSK) uses (X_ne, Y_ne, Z_ne)")
    print("   - KAN uses best features from feature selection")
    print("2. Added MLE predictor function")
    print("3. Added noise analysis experiment (all noise levels 20-60dB)")
    print("4. Added noise analysis visualization")

def create_noise_analysis_code():
    """Generate the noise analysis experiment code."""
    return [
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
        "    # Get configurations\n",
        "    if dataset_type == '4CSK':\n",
        "        fnn_cols = ['Vr', 'Vg', 'Vb']\n",
        "        kan_cols = best_configs[base_file]['columns']\n",
        "        params = best_configs[base_file]['params']\n",
        "        fnn_hidden = FNN_CONFIGS[base_file]\n",
        "    else:\n",
        "        fnn_cols = ['X_ne', 'Y_ne', 'Z_ne']\n",
        "        kan_cols = best_configs[base_file]['columns']\n",
        "        params = best_configs[base_file]['params']\n",
        "        fnn_hidden = FNN_CONFIGS[base_file]\n",
        "    \n",
        "    for noise_file in noise_files:\n",
        "        # Extract noise level from filename (e.g., 'm20dB' -> 20)\n",
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
        "                    # Load data with FNN features\n",
        "                    X_train, X_test, y_train, y_test, num_classes = load_data(\n",
        "                        noise_file, fnn_cols, train_size=64, test_size=1024\n",
        "                    )\n",
        "                    \n",
        "                    model = FNN(len(fnn_cols), fnn_hidden, num_classes).to(device)\n",
        "                    best_ser, _, _ = train_model(\n",
        "                        model, X_train, y_train, X_test, y_test,\n",
        "                        epochs=CONFIG['final_epochs'],\n",
        "                        lr=params['lr']\n",
        "                    )\n",
        "                    ser = best_ser\n",
        "                    \n",
        "                else:  # KAN\n",
        "                    # Load data with KAN features\n",
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
        "                        lr=params['lr']\n",
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

def create_noise_plots_code():
    """Generate the noise analysis plotting code."""
    return [
        "# Plot noise analysis results\n",
        "for dataset in ['4CSK', '8CSK']:\n",
        "    fig, ax = plt.subplots(figsize=(10, 6))\n",
        "    \n",
        "    for model in ['FNN', 'KAN', 'MLE']:\n",
        "        data = df_noise[(df_noise['Dataset'] == dataset) & (df_noise['Model'] == model)]\n",
        "        data = data.sort_values('Noise_Level_dB')\n",
        "        \n",
        "        ax.errorbar(data['Noise_Level_dB'], data['Mean_SER'], \n",
        "                    yerr=data['Std_SER'], label=model, \n",
        "                    marker='o', capsize=5, linewidth=2)\n",
        "    \n",
        "    ax.set_xlabel('Noise Level (dB)')\n",
        "    ax.set_ylabel('Mean SER')\n",
        "    ax.set_title(f'Noise Analysis - {dataset}')\n",
        "    ax.set_yscale('log')\n",
        "    ax.legend()\n",
        "    ax.grid(True, which='both', alpha=0.3)\n",
        "    plt.tight_layout()\n",
        "    \n",
        "    filename = f'noise_analysis_{dataset}'\n",
        "    plt.savefig(os.path.join(OUTPUT_DIR, filename + '.png'), dpi=300, bbox_inches='tight')\n",
        "    plt.savefig(os.path.join(OUTPUT_DIR, filename + '.pdf'), dpi=300, bbox_inches='tight')\n",
        "    plt.show()"
    ]

if __name__ == "__main__":
    add_noise_analysis()
