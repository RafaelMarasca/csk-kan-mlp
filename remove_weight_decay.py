"""
Remove all weight_decay references from the notebook.
- Cell 16: Remove weight_decay parameter from train_model function
- Cell 24: Remove weight_decay from train_model call in feature selection
- Cell 37: Remove weight_decay from best config summary
"""

import json

def remove_weight_decay():
    """Remove all weight_decay references."""
    
    # Read the notebook
    with open('kan_experiments_refactored.ipynb', 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    changes_made = []
    
    # Cell 16: Update train_model function
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code' and any('def train_model' in line for line in cell.get('source', [])):
            print(f"Updating cell {i}: train_model function")
            # Remove weight_decay from signature and don't use it in optimizer
            cell['source'] = [
                "def train_model(model, X_train, y_train, X_test, y_test, \n",
                "                epochs=200, lr=1e-4, verbose=False):\n",
                "    \"\"\"Train model and return best SER.\"\"\"\n",
                "    optimizer = optim.Adam(model.parameters(), lr=lr)\n",
                "    criterion = nn.CrossEntropyLoss()\n",
                "    \n",
                "    history = {'train_loss': [], 'test_ser': [], 'test_acc': []}\n",
                "    best_ser = 1.0\n",
                "    best_model_state = None\n",
                "    \n",
                "    iterator = tqdm(range(epochs), desc='Training') if verbose else range(epochs)\n",
                "    \n",
                "    for epoch in iterator:\n",
                "        model.train()\n",
                "        optimizer.zero_grad()\n",
                "        output = model(X_train).squeeze(-1)\n",
                "        loss = criterion(output, y_train)\n",
                "        loss.backward()\n",
                "        optimizer.step()\n",
                "        \n",
                "        history['train_loss'].append(loss.item())\n",
                "        \n",
                "        model.eval()\n",
                "        with torch.no_grad():\n",
                "            test_out = model(X_test).squeeze(-1)\n",
                "            test_loss = criterion(test_out, y_test)\n",
                "            _, preds = torch.max(test_out, 1)\n",
                "            accuracy = (preds == y_test).float().mean().item()\n",
                "            ser = 1.0 - accuracy\n",
                "            \n",
                "            history['test_ser'].append(ser)\n",
                "            history['test_acc'].append(accuracy)\n",
                "            \n",
                "            if ser < best_ser:\n",
                "                best_ser = ser\n",
                "                best_model_state = copy.deepcopy(model.state_dict())\n",
                "        \n",
                "        if verbose and (epoch % 50 == 0):\n",
                "            iterator.set_postfix(loss=loss.item(), ser=ser)\n",
                "    \n",
                "    return best_ser, history, best_model_state"
            ]
            changes_made.append(f"Cell {i}: Removed weight_decay from train_model")
            break
    
    # Cell 24: Update feature selection to remove weight_decay from train_model call
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code' and any('Storage for results' in line for line in cell.get('source', [])):
            print(f"Updating cell {i}: feature selection")
            # Find and remove weight_decay line
            new_source = []
            for line in cell['source']:
                if 'weight_decay=' not in line:
                    new_source.append(line)
            cell['source'] = new_source
            changes_made.append(f"Cell {i}: Removed weight_decay from train_model call")
            break
    
    # Cell 37: Update best config summary to remove weight_decay line
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code' and any('BEST CONFIGURATIONS SUMMARY' in line for line in cell.get('source', [])):
            print(f"Updating cell {i}: best config summary")
            # Remove the weight_decay print line
            new_source = []
            for line in cell['source']:
                if 'Weight Decay' not in line:
                    new_source.append(line)
            cell['source'] = new_source
            changes_made.append(f"Cell {i}: Removed weight_decay from summary")
            break
    
    # Save the updated notebook
    with open('kan_experiments_refactored.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("Weight decay removed successfully!")
    print("="*80)
    print("\nChanges made:")
    for change in changes_made:
        print(f"  - {change}")
    print("\nThe training now uses simple Adam optimizer without weight decay.")

if __name__ == "__main__":
    remove_weight_decay()
