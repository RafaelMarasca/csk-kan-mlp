import json

with open('kan_experiments_refactored.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("Finding all weight_decay occurrences:")
print("="*80)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        if 'weight_decay' in source:
            print(f"\nCell {i}:")
            print("-"*80)
            for j, line in enumerate(cell['source']):
                if 'weight_decay' in line:
                    print(f"  Line {j}: {line}", end='')
