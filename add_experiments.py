"""
Add experiment sections to the refactored notebook.
This extends the notebook with all experiments and visualizations.
"""

import json
import os

def add_experiment_sections():
    """Add experiment sections to the existing notebook."""
    
    # Load existing notebook
    notebook_path = "kan_experiments_refactored.ipynb"
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    cells = notebook["cells"]
    
    # ========================================================================
    # SECTION 4: EXPERIMENTS
    # ========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\\n",
            "# Section 4: Experiments\\n",
            "\\n",
            "Comprehensive analysis of KAN performance across different configurations."
        ]
    })
    
    # Experiment 1: Feature Selection
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4.1 Feature Selection and Hyperparameter Search\\n",
            "\\n",
            "Systematic search across all feature sets and hyperparameter combinations."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Storage for results\\n",
            "results = {\\n",
            "    'Dataset': [], 'Feature_Set': [], 'Hidden_Layers': [],\\n",
            "    'Mean_SER': [], 'Std_SER': [], 'Best_SER': []\\n",
            "}\\n",
            "\\n",
            "best_configs = {}  # Store best configuration for each dataset\\n",
            "\\n",
            "print('Starting Feature Selection and Hyperparameter Search...')\\n",
            "print('='*80)\\n",
            "\\n",
            "for filename in ['data_4csk.csv', 'data_8csk.csv']:\\n",
            "    print(f'\\\\nDataset: {filename}')\\n",
            "    best_mean_ser = 1.0\\n",
            "    best_config = None\\n",
            "    \\n",
            "    for feat_name, cols in FEATURE_SETS.items():\\n",
            "        X_train, X_test, y_train, y_test, num_classes = load_data(\\n",
            "            filename, cols, train_size=64, test_size=64, \\n",
            "            normalize=CONFIG['normalize']  # NO NORMALIZATION\\n",
            "        )\\n",
            "        \\n",
            "        for params in PARAM_COMBINATIONS:\\n",
            "            sers = []\\n",
            "            layers = [len(cols)] + params['hidden_layers'] + [num_classes]\\n",
            "            \\n",
            "            # Multiple runs for statistics\\n",
            "            for _ in range(CONFIG['n_repeats']):\\n",
            "                model = ReLUKAN(layers, grid=params['grid'], k=params['k']).to(device)\\n",
            "                best_ser, _, _ = train_model(\\n",
            "                    model, X_train, y_train, X_test, y_test,\\n",
            "                    epochs=CONFIG['search_epochs'],\\n",
            "                    lr=params['lr'],\\n",
            "                    weight_decay=params['weight_decay']\\n",
            "                )\\n",
            "                sers.append(best_ser)\\n",
            "            \\n",
            "            mean_ser = np.mean(sers)\\n",
            "            std_ser = np.std(sers)\\n",
            "            min_ser = np.min(sers)\\n",
            "            \\n",
            "            results['Dataset'].append(filename.replace('.csv', '').replace('data_', '').upper())\\n",
            "            results['Feature_Set'].append(feat_name)\\n",
            "            results['Hidden_Layers'].append(str(params['hidden_layers']))\\n",
            "            results['Mean_SER'].append(mean_ser)\\n",
            "            results['Std_SER'].append(std_ser)\\n",
            "            results['Best_SER'].append(min_ser)\\n",
            "            \\n",
            "            if mean_ser < best_mean_ser:\\n",
            "                best_mean_ser = mean_ser\\n",
            "                best_config = {\\n",
            "                    'features': feat_name,\\n",
            "                    'columns': cols,\\n",
            "                    'params': params,\\n",
            "                    'mean_ser': mean_ser,\\n",
            "                    'std_ser': std_ser\\n",
            "                }\\n",
            "    \\n",
            "    best_configs[filename] = best_config\\n",
            "    print(f'  Best: {best_config[\\\"features\\\"]} | SER: {best_config[\\\"mean_ser\\\"]:.4f} +/- {best_config[\\\"std_ser\\\"]:.4f}')\\n",
            "\\n",
            "# Save results\\n",
            "df_results = pd.DataFrame(results)\\n",
            "df_results.to_csv(os.path.join(OUTPUT_DIR, 'feature_search_results.csv'), index=False)\\n",
            "print(f'\\\\nResults saved to {OUTPUT_DIR}/feature_search_results.csv')"
        ]
    })
    
    # Results table
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["### 4.1.1 Results Table - Best Configuration per Feature Set"]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Get best configuration for each feature set\\n",
            "best_per_feature = df_results.loc[\\n",
            "    df_results.groupby(['Dataset', 'Feature_Set'])['Mean_SER'].idxmin()\\n",
            "].copy()\\n",
            "\\n",
            "# Display table\\n",
            "display_df = best_per_feature[['Dataset', 'Feature_Set', 'Hidden_Layers', 'Mean_SER', 'Std_SER']].copy()\\n",
            "display_df['Mean_SER'] = display_df['Mean_SER'].apply(lambda x: f'{x:.4f}')\\n",
            "display_df['Std_SER'] = display_df['Std_SER'].apply(lambda x: f'{x:.4f}')\\n",
            "\\n",
            "plot_results_table(display_df, 'Best Configuration per Feature Set')"
        ]
    })
    
    # Visualization
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["### 4.1.2 Visualization - Feature Set Comparison"]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Bar plot comparing feature sets\\n",
            "datasets = best_per_feature['Dataset'].unique()\\n",
            "features = best_per_feature['Feature_Set'].unique()\\n",
            "\\n",
            "fig, ax = plt.subplots(figsize=(12, 6))\\n",
            "x = np.arange(len(features))\\n",
            "width = 0.35\\n",
            "\\n",
            "for i, ds in enumerate(datasets):\\n",
            "    subset = best_per_feature[best_per_feature['Dataset'] == ds]\\n",
            "    subset = subset.set_index('Feature_Set').reindex(features).reset_index()\\n",
            "    \\n",
            "    ax.bar(x + i*width, subset['Mean_SER'], width, \\n",
            "           label=ds, yerr=subset['Std_SER'], capsize=5)\\n",
            "\\n",
            "ax.set_ylabel('Mean SER')\\n",
            "ax.set_xlabel('Feature Set')\\n",
            "ax.set_xticks(x + width/2 if len(datasets) > 1 else x)\\n",
            "ax.set_xticklabels(features, rotation=45, ha='right')\\n",
            "ax.legend()\\n",
            "ax.set_yscale('log')\\n",
            "ax.grid(True, which='both', alpha=0.3)\\n",
            "plt.tight_layout()\\n",
            "plt.savefig(os.path.join(OUTPUT_DIR, 'feature_comparison.png'), dpi=300, bbox_inches='tight')\\n",
            "plt.savefig(os.path.join(OUTPUT_DIR, 'feature_comparison.pdf'), dpi=300, bbox_inches='tight')\\n",
            "plt.show()"
        ]
    })
    
    # Top configurations
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 4.1.3 Top 5 Configurations per Dataset"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "for dataset in df_results['Dataset'].unique():\\n",
            "    print(f'\\\\nTop 5 Configurations for {dataset}:')\\n",
            "    print('='*80)\\n",
            "    \\n",
            "    top5 = df_results[df_results['Dataset'] == dataset].nsmallest(5, 'Mean_SER')\\n",
            "    display_top5 = top5[['Feature_Set', 'Hidden_Layers', 'Mean_SER', 'Std_SER', 'Best_SER']].copy()\\n",
            "    display_top5['Mean_SER'] = display_top5['Mean_SER'].apply(lambda x: f'{x:.4f}')\\n",
            "    display_top5['Std_SER'] = display_top5['Std_SER'].apply(lambda x: f'{x:.4f}')\\n",
            "    display_top5['Best_SER'] = display_top5['Best_SER'].apply(lambda x: f'{x:.4f}')\\n",
            "    \\n",
            "    print(display_top5.to_string(index=False))\\n",
            "    print('='*80)"
        ]
    })
    
    # Sample Efficiency
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4.2 Sample Efficiency: KAN vs FNN\\n",
            "\\n",
            "Compare KAN and FNN performance across different training set sizes."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Sample efficiency experiment\\n",
            "train_sizes = [8, 16, 32, 64, 128, 256, 384, 512, 640, 768, 896, 1024]\\n",
            "test_size = 1024\\n",
            "\\n",
            "efficiency_results = {\\n",
            "    'Dataset': [], 'Train_Size': [], 'Model': [], 'Mean_SER': [], 'Std_SER': []\\n",
            "}\\n",
            "\\n",
            "print('Sample Efficiency Experiment')\\n",
            "print('='*80)\\n",
            "\\n",
            "for filename in ['data_4csk.csv', 'data_8csk.csv']:\\n",
            "    print(f'\\\\nDataset: {filename}')\\n",
            "    \\n",
            "    # Use best configuration from previous experiment\\n",
            "    best_cfg = best_configs[filename]\\n",
            "    cols = best_cfg['columns']\\n",
            "    params = best_cfg['params']\\n",
            "    \\n",
            "    for train_size in train_sizes:\\n",
            "        print(f'  Train size: {train_size}')\\n",
            "        \\n",
            "        X_train, X_test, y_train, y_test, num_classes = load_data(\\n",
            "            filename, cols, train_size=train_size, test_size=test_size,\\n",
            "            normalize=CONFIG['normalize']\\n",
            "        )\\n",
            "        \\n",
            "        # KAN\\n",
            "        kan_sers = []\\n",
            "        for _ in range(CONFIG['n_repeats']):\\n",
            "            layers = [len(cols)] + params['hidden_layers'] + [num_classes]\\n",
            "            model = ReLUKAN(layers, grid=params['grid'], k=params['k']).to(device)\\n",
            "            best_ser, _, _ = train_model(\\n",
            "                model, X_train, y_train, X_test, y_test,\\n",
            "                epochs=CONFIG['final_epochs'],\\n",
            "                lr=params['lr'], weight_decay=params['weight_decay']\\n",
            "            )\\n",
            "            kan_sers.append(best_ser)\\n",
            "        \\n",
            "        # FNN\\n",
            "        fnn_sers = []\\n",
            "        for _ in range(CONFIG['n_repeats']):\\n",
            "            model = FNN(len(cols), params['hidden_layers'], num_classes).to(device)\\n",
            "            best_ser, _, _ = train_model(\\n",
            "                model, X_train, y_train, X_test, y_test,\\n",
            "                epochs=CONFIG['final_epochs'],\\n",
            "                lr=params['lr'], weight_decay=params['weight_decay']\\n",
            "            )\\n",
            "            fnn_sers.append(best_ser)\\n",
            "        \\n",
            "        ds_name = filename.replace('.csv', '').replace('data_', '').upper()\\n",
            "        \\n",
            "        efficiency_results['Dataset'].extend([ds_name, ds_name])\\n",
            "        efficiency_results['Train_Size'].extend([train_size, train_size])\\n",
            "        efficiency_results['Model'].extend(['KAN', 'FNN'])\\n",
            "        efficiency_results['Mean_SER'].extend([np.mean(kan_sers), np.mean(fnn_sers)])\\n",
            "        efficiency_results['Std_SER'].extend([np.std(kan_sers), np.std(fnn_sers)])\\n",
            "\\n",
            "df_efficiency = pd.DataFrame(efficiency_results)\\n",
            "df_efficiency.to_csv(os.path.join(OUTPUT_DIR, 'sample_efficiency_results.csv'), index=False)\\n",
            "print(f'\\\\nResults saved to {OUTPUT_DIR}/sample_efficiency_results.csv')"
        ]
    })
    
    # Efficiency plots
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["### 4.2.1 Sample Efficiency Plots"]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Plot sample efficiency curves\\n",
            "for dataset in df_efficiency['Dataset'].unique():\\n",
            "    fig, ax = plt.subplots(figsize=(10, 6))\\n",
            "    \\n",
            "    for model in ['KAN', 'FNN']:\\n",
            "        data = df_efficiency[(df_efficiency['Dataset'] == dataset) & \\n",
            "                             (df_efficiency['Model'] == model)]\\n",
            "        ax.errorbar(data['Train_Size'], data['Mean_SER'], \\n",
            "                    yerr=data['Std_SER'], label=model, \\n",
            "                    marker='o', capsize=5, linewidth=2)\\n",
            "    \\n",
            "    ax.set_xlabel('Training Set Size')\\n",
            "    ax.set_ylabel('Mean SER')\\n",
            "    ax.set_title(f'Sample Efficiency - {dataset}')\\n",
            "    ax.set_yscale('log')\\n",
            "    ax.legend()\\n",
            "    ax.grid(True, which='both', alpha=0.3)\\n",
            "    plt.tight_layout()\\n",
            "    \\n",
            "    filename = f'sample_efficiency_{dataset}'\\n",
            "    plt.savefig(os.path.join(OUTPUT_DIR, filename + '.png'), dpi=300, bbox_inches='tight')\\n",
            "    plt.savefig(os.path.join(OUTPUT_DIR, filename + '.pdf'), dpi=300, bbox_inches='tight')\\n",
            "    plt.show()"
        ]
    })
    
    # ========================================================================
    # SECTION 5: SUMMARY
    # ========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\\n",
            "# Section 5: Summary and Conclusions"
        ]
    })
    
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5.1 Best Configurations Summary"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print('\\\\nBEST CONFIGURATIONS SUMMARY')\\n",
            "print('='*80)\\n",
            "\\n",
            "for filename, config in best_configs.items():\\n",
            "    dataset = filename.replace('.csv', '').replace('data_', '').upper()\\n",
            "    print(f'\\\\n{dataset}:')\\n",
            "    print(f'  Feature Set: {config[\\\"features\\\"]}')\\n",
            "    print(f'  Columns: {config[\\\"columns\\\"]}')\\n",
            "    print(f'  Hidden Layers: {config[\\\"params\\\"][\\\"hidden_layers\\\"]}')\\n",
            "    print(f'  Mean SER: {config[\\\"mean_ser\\\"]:.4f} +/- {config[\\\"std_ser\\\"]:.4f}')\\n",
            "    print(f'  Learning Rate: {config[\\\"params\\\"][\\\"lr\\\"]}')\\n",
            "    print(f'  Weight Decay: {config[\\\"params\\\"][\\\"weight_decay\\\"]}')\\n",
            "\\n",
            "print('\\\\n' + '='*80)\\n",
            "print('NOTE: All experiments conducted WITHOUT data normalization')\\n",
            "print('='*80)"
        ]
    })
    
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5.2 Key Findings\\n",
            "\\n",
            "1. **Feature Sets**: Analysis shows which feature representations work best\\n",
            "2. **Sample Efficiency**: KAN vs FNN comparison across training set sizes\\n",
            "3. **No Normalization**: All results obtained using raw, unnormalized features\\n",
            "4. **Top Configurations**: Best performing architectures identified for each dataset\\n",
            "\\n",
            "All results and visualizations have been saved to the `results_refactored/` directory."
        ]
    })
    
    # Save updated notebook
    notebook["cells"] = cells
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print(f"\\nNotebook updated: {notebook_path}")
    print(f"Total cells: {len(cells)}")

if __name__ == "__main__":
    add_experiment_sections()
