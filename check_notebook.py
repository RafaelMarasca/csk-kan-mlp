import json

with open('kan_experiments_refactored.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f'Total cells: {len(nb["cells"])}')
print('\nFirst 40 cells:')
for i, c in enumerate(nb['cells'][:40]):
    source_preview = c['source'][0][:60] if c['source'] else 'empty'
    print(f'{i}: {c["cell_type"]:10} - {source_preview}...')
