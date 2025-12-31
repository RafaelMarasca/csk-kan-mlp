"""
Update train_model function to use LBFGS for FNN and Adam for KAN.
LBFGS requires a closure function for line search.
"""

import json

def update_optimizers():
    """Update train_model to use different optimizers."""
    
    # Read the notebook
    with open('kan_experiments_refactored.ipynb', 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Find and update train_model function
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code' and any('def train_model' in line for line in cell.get('source', [])):
            print(f"Updating train_model function in cell {i}")
            cell['source'] = [
                "def train_model(model, X_train, y_train, X_test, y_test, \n",
                "                epochs=200, lr=1e-4, verbose=False, optimizer_type='adam'):\n",
                "    \"\"\"\n",
                "    Train model and return best SER.\n",
                "    \n",
                "    Args:\n",
                "        optimizer_type: 'adam' for KAN, 'lbfgs' for FNN\n",
                "    \"\"\"\n",
                "    criterion = nn.CrossEntropyLoss()\n",
                "    \n",
                "    if optimizer_type.lower() == 'lbfgs':\n",
                "        optimizer = optim.LBFGS(model.parameters(), lr=lr, max_iter=20, \n",
                "                                history_size=10, line_search_fn='strong_wolfe')\n",
                "    else:  # adam\n",
                "        optimizer = optim.Adam(model.parameters(), lr=lr)\n",
                "    \n",
                "    history = {'train_loss': [], 'test_ser': [], 'test_acc': []}\n",
                "    best_ser = 1.0\n",
                "    best_model_state = None\n",
                "    \n",
                "    iterator = tqdm(range(epochs), desc='Training') if verbose else range(epochs)\n",
                "    \n",
                "    for epoch in iterator:\n",
                "        model.train()\n",
                "        \n",
                "        if optimizer_type.lower() == 'lbfgs':\n",
                "            # LBFGS requires closure function\n",
                "            def closure():\n",
                "                optimizer.zero_grad()\n",
                "                output = model(X_train).squeeze(-1)\n",
                "                loss = criterion(output, y_train)\n",
                "                loss.backward()\n",
                "                return loss\n",
                "            \n",
                "            optimizer.step(closure)\n",
                "            # Get loss for history\n",
                "            with torch.no_grad():\n",
                "                output = model(X_train).squeeze(-1)\n",
                "                loss = criterion(output, y_train)\n",
                "        else:\n",
                "            # Adam optimizer\n",
                "            optimizer.zero_grad()\n",
                "            output = model(X_train).squeeze(-1)\n",
                "            loss = criterion(output, y_train)\n",
                "            loss.backward()\n",
                "            optimizer.step()\n",
                "        \n",
                "        history['train_loss'].append(loss.item())\n",
                "        \n",
                "        # Evaluation\n",
                "        model.eval()\n",
                "        with torch.no_grad():\n",
                "            test_out = model(X_test).squeeze(-1)\n",
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
            break
    
    print("\nUpdating all train_model calls...")
    
    # Update feature selection to use Adam for KAN
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code' and any('Storage for results' in line for line in cell.get('source', [])):
            print(f"  Updating feature selection (cell {i}) - KAN uses Adam")
            # The feature selection only trains KAN, so add optimizer_type='adam'
            source = ''.join(cell['source'])
            source = source.replace(
                "lr=params['lr']\n",
                "lr=params['lr'],\n                    optimizer_type='adam'\n"
            )
            cell['source'] = source.split('\n')
            cell['source'] = [line + '\n' for line in cell['source'][:-1]] + [cell['source'][-1]]
            break
    
    # Update sample efficiency to use Adam for KAN, LBFGS for FNN
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code' and any('Sample efficiency experiment' in line for line in cell.get('source', [])):
            print(f"  Updating sample efficiency (cell {i}) - KAN uses Adam, FNN uses LBFGS")
            source = ''.join(cell['source'])
            # Add optimizer_type for KAN
            source = source.replace(
                "epochs=CONFIG['final_epochs'],\n                lr=params['lr']\n            )\n            kan_sers.append(best_ser)",
                "epochs=CONFIG['final_epochs'],\n                lr=params['lr'],\n                optimizer_type='adam'\n            )\n            kan_sers.append(best_ser)"
            )
            # Add optimizer_type for FNN
            source = source.replace(
                "epochs=CONFIG['final_epochs'],\n                lr=params['lr']\n            )\n            fnn_sers.append(best_ser)",
                "epochs=CONFIG['final_epochs'],\n                lr=params['lr'],\n                optimizer_type='lbfgs'\n            )\n            fnn_sers.append(best_ser)"
            )
            cell['source'] = source.split('\n')
            cell['source'] = [line + '\n' for line in cell['source'][:-1]] + [cell['source'][-1]]
            break
    
    # Update noise analysis to use Adam for KAN, LBFGS for FNN
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code' and any('Noise analysis experiment' in line for line in cell.get('source', [])):
            print(f"  Updating noise analysis (cell {i}) - KAN uses Adam, FNN uses LBFGS")
            source = ''.join(cell['source'])
            # Add optimizer_type for FNN
            source = source.replace(
                "epochs=CONFIG['final_epochs'],\n                        lr=params['lr']\n                    )\n                    ser = best_ser\n                    \n                else:  # KAN",
                "epochs=CONFIG['final_epochs'],\n                        lr=params['lr'],\n                        optimizer_type='lbfgs'\n                    )\n                    ser = best_ser\n                    \n                else:  # KAN"
            )
            # Add optimizer_type for KAN
            source = source.replace(
                "epochs=CONFIG['final_epochs'],\n                        lr=params['lr']\n                    )\n                    ser = best_ser\n                \n                # Restore seed",
                "epochs=CONFIG['final_epochs'],\n                        lr=params['lr'],\n                        optimizer_type='adam'\n                    )\n                    ser = best_ser\n                \n                # Restore seed"
            )
            cell['source'] = source.split('\n')
            cell['source'] = [line + '\n' for line in cell['source'][:-1]] + [cell['source'][-1]]
            break
    
    # Save the updated notebook
    with open('kan_experiments_refactored.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("Optimizers updated successfully!")
    print("="*80)
    print("\nChanges made:")
    print("1. Updated train_model function:")
    print("   - Added optimizer_type parameter")
    print("   - LBFGS for FNN (with closure function)")
    print("   - Adam for KAN")
    print("2. Updated all train_model calls:")
    print("   - Feature selection: Adam (KAN only)")
    print("   - Sample efficiency: Adam for KAN, LBFGS for FNN")
    print("   - Noise analysis: Adam for KAN, LBFGS for FNN")

if __name__ == "__main__":
    update_optimizers()
