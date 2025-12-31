import json

with open('kan_experiments_refactored.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f'Total cells: {len(nb["cells"])}')

# Check for MLE predictor
mle_found = False
for i, cell in enumerate(nb['cells']):
    if 'mle_predict' in ''.join(cell.get('source', [])):
        mle_found = True
        print(f'MLE predictor found in cell {i}')
        break

# Check for noise analysis
noise_found = False
for i, cell in enumerate(nb['cells']):
    if 'Noise Analysis' in ''.join(cell.get('source', [])) or 'noise analysis' in ''.join(cell.get('source', [])).lower():
        noise_found = True
        print(f'Noise analysis found in cell {i}')
        break

print(f'\nVerification:')
print(f'  MLE predictor: {"PASS" if mle_found else "FAIL"}')
print(f'  Noise analysis: {"PASS" if noise_found else "FAIL"}')
