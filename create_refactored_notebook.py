"""
Script to generate the refactored KAN experiments notebook.
This creates a clean, well-organized notebook with improved flow and no data normalization.
"""

import json

def create_refactored_notebook():
    """Create the complete refactored notebook structure."""
    
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Add all cells
    cells = []
    
    # ========================================================================
    # TITLE AND INTRODUCTION
    # ========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# KAN (Kolmogorov-Arnold Networks) - CSK Modulation Analysis\\n",
            "\\n",
            "**Refactored Version** - Improved flow, clearer code, comprehensive visualizations\\n",
            "\\n",
            "This notebook performs a comprehensive analysis of KAN networks for 4-CSK and 8-CSK modulation classification.\\n",
            "\\n",
            "## Key Features:\\n",
            "- ✓ **No Data Normalization**: Raw features are used for training\\n",
            "- ✓ **Comprehensive Hyperparameter Search**: Systematic exploration of network architectures\\n",
            "- ✓ **Feature Set Comparison**: Analysis of different feature representations\\n",
            "- ✓ **Sample Efficiency Analysis**: KAN vs FNN performance comparison\\n",
            "- ✓ **Detailed Visualizations**: Results tables, confusion matrices, and performance plots\\n",
            "\\n",
            "## Datasets:\\n",
            "- `data_4csk.csv` - 4-CSK modulation data\\n",
            "- `data_8csk.csv` - 8-CSK modulation data\\n",
            "\\n",
            "## Notebook Structure:\\n",
            "1. **Setup and Configuration** - Imports, parameters, feature definitions\\n",
            "2. **Utility Functions** - Data loading, training, plotting\\n",
            "3. **Model Definitions** - FNN baseline model\\n",
            "4. **Experiments** - Feature selection, sample efficiency, top configurations\\n",
            "5. **Results Analysis** - Summary tables and visualizations"
        ]
    })
    
    # ========================================================================
    # SECTION 1: SETUP AND CONFIGURATION
    # ========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\\n",
            "# Section 1: Setup and Configuration\\n",
            "\\n",
            "This section contains all imports, global settings, and parameter definitions."
        ]
    })
    
    # 1.1 Imports
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 1.1 Imports and Device Setup"]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Standard library imports\\n",
            "import os\\n",
            "import copy\\n",
            "import itertools\\n",
            "from typing import List, Tuple, Dict, Optional\\n",
            "\\n",
            "# Data manipulation and analysis\\n",
            "import pandas as pd\\n",
            "import numpy as np\\n",
            "\\n",
            "# PyTorch\\n",
            "import torch\\n",
            "import torch.nn as nn\\n",
            "import torch.optim as optim\\n",
            "\\n",
            "# Scikit-learn\\n",
            "from sklearn.model_selection import train_test_split\\n",
            "from sklearn.metrics import confusion_matrix, classification_report\\n",
            "\\n",
            "# Visualization\\n",
            "import matplotlib.pyplot as plt\\n",
            "import seaborn as sns\\n",
            "\\n",
            "# Progress bar\\n",
            "from tqdm import tqdm\\n",
            "\\n",
            "# KAN implementation\\n",
            "try:\\n",
            "    from torch_relu_kan import ReLUKAN\\n",
            "    print(\\\"✓ ReLUKAN imported successfully\\\")\\n",
            "except ImportError:\\n",
            "    print(\\\"✗ Error: torch_relu_kan.py not found. Please ensure it's in the same directory.\\\")\\n",
            "    raise\\n",
            "\\n",
            "# Device configuration\\n",
            "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\\n",
            "print(f\\\"✓ Using device: {device}\\\")\\n",
            "\\n",
            "# Set random seeds for reproducibility\\n",
            "SEED = 42\\n",
            "torch.manual_seed(SEED)\\n",
            "np.random.seed(SEED)\\n",
            "print(f\\\"✓ Random seed set to: {SEED}\\\")"
        ]
    })
    
    # 1.2 Global Configuration
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 1.2 Global Configuration"]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ============================================================================\\n",
            "# OUTPUT CONFIGURATION\\n",
            "# ============================================================================\\n",
            "OUTPUT_DIR = \\\"results_refactored\\\"\\n",
            "os.makedirs(OUTPUT_DIR, exist_ok=True)\\n",
            "print(f\\\"✓ Results will be saved to: {OUTPUT_DIR}/\\\")\\n",
            "\\n",
            "# ============================================================================\\n",
            "# TRAINING CONFIGURATION\\n",
            "# ============================================================================\\n",
            "TRAINING_CONFIG = {\\n",
            "    'search_epochs': 50,      # Epochs for hyperparameter search\\n",
            "    'final_epochs': 200,      # Epochs for final training\\n",
            "    'n_repeats': 3,           # Number of training repeats for statistics\\n",
            "    'default_lr': 1e-3,       # Default learning rate\\n",
            "    'default_wd': 1e-3,       # Default weight decay\\n",
            "}\\n",
            "\\n",
            "# ============================================================================\\n",
            "# DATA CONFIGURATION\\n",
            "# ============================================================================\\n",
            "DATA_CONFIG = {\\n",
            "    'datasets': ['data_4csk.csv', 'data_8csk.csv'],\\n",
            "    'default_train_size': 64,\\n",
            "    'default_test_size': 64,\\n",
            "    'normalize': False,  # *** NO NORMALIZATION ***\\n",
            "}\\n",
            "\\n",
            "print(\\\"\\\\n\\\" + \\\"=\\\"*80)\\n",
            "print(\\\"CONFIGURATION SUMMARY\\\")\\n",
            "print(\\\"=\\\"*80)\\n",
            "print(f\\\"Search Epochs: {TRAINING_CONFIG['search_epochs']}\\\")\\n",
            "print(f\\\"Final Epochs: {TRAINING_CONFIG['final_epochs']}\\\")\\n",
            "print(f\\\"Training Repeats: {TRAINING_CONFIG['n_repeats']}\\\")\\n",
            "print(f\\\"Data Normalization: {DATA_CONFIG['normalize']}\\\")\\n",
            "print(\\\"=\\\"*80 + \\\"\\\\n\\\")"
        ]
    })
    
    # Continue with remaining cells...
    # (I'll add the rest in the next part to keep this manageable)
    
    notebook["cells"] = cells
    
    # Save the notebook
    output_path = "kan_experiments_refactored.ipynb"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=4, ensure_ascii=False)
    
    print(f"Refactored notebook created: {output_path}")
    return output_path

if __name__ == "__main__":
    create_refactored_notebook()
