import json

# Find the noise analysis cell
with open('kan_experiments_refactored.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("Searching for noise analysis code...")
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        if 'Noise analysis experiment' in source or 'noise_results' in source:
            print(f"\nFound noise analysis in cell {i}")
            print("="*80)
            # Show the MLE part
            lines = cell['source']
            in_mle = False
            for j, line in enumerate(lines):
                if 'MLE' in line or in_mle:
                    print(f"{j}: {line}", end='')
                    in_mle = True
                    if 'else:' in line and 'KAN' in line:
                        break
            print("\n" + "="*80)
            break
