import json

with open('kan_experiments_refactored.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print('Cell 32 content:')
print('='*80)
for line in nb['cells'][32]['source']:
    print(line, end='')
print('\n' + '='*80)
