"""
Add softmax activation to FNN output layer.
"""

import json

def add_softmax_to_fnn():
    """Update FNN class to include softmax activation on output."""
    
    # Read the notebook
    with open('kan_experiments_refactored.ipynb', 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Find and update FNN class definition
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code' and any('class FNN(nn.Module)' in line for line in cell.get('source', [])):
            print(f"Updating FNN class in cell {i}")
            cell['source'] = [
                "class FNN(nn.Module):\n",
                "    \"\"\"Feedforward Neural Network baseline with softmax output.\"\"\"\n",
                "    def __init__(self, input_dim, hidden_dims, output_dim):\n",
                "        super(FNN, self).__init__()\n",
                "        layers = []\n",
                "        prev_dim = input_dim\n",
                "        for h_dim in hidden_dims:\n",
                "            layers.append(nn.Linear(prev_dim, h_dim))\n",
                "            layers.append(nn.ReLU())\n",
                "            prev_dim = h_dim\n",
                "        layers.append(nn.Linear(prev_dim, output_dim))\n",
                "        layers.append(nn.Softmax(dim=1))  # Add softmax activation\n",
                "        self.network = nn.Sequential(*layers)\n",
                "    \n",
                "    def forward(self, x):\n",
                "        if x.dim() == 3:\n",
                "            x = x.squeeze(-1)\n",
                "        return self.network(x).unsqueeze(-1)\n",
                "\n",
                "print('FNN model defined with softmax output')"
            ]
            break
    
    # Save the updated notebook
    with open('kan_experiments_refactored.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("FNN updated with softmax activation!")
    print("="*80)
    print("\nChange made:")
    print("- Added nn.Softmax(dim=1) to FNN output layer")
    print("- Output now produces probability distributions over classes")

if __name__ == "__main__":
    add_softmax_to_fnn()
