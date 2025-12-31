import json

with open('kan_experiments_refactored.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("Verification of notebook updates:")
print("="*80)

# Check cell 5 (CONFIG and FNN_CONFIGS)
print("\n1. Cell 5 (CONFIG and FNN_CONFIGS):")
print("-"*80)
config_cell = nb['cells'][5]['source']
for line in config_cell:
    if 'final_epochs' in line or 'FNN_CONFIGS' in line or 'data_4csk' in line or 'data_8csk' in line:
        print(line, end='')

# Check cell 32 (Sample efficiency experiment)
print("\n\n2. Cell 32 (Sample Efficiency Experiment):")
print("-"*80)
cell32 = nb['cells'][32]['source']
# Show key lines
for i, line in enumerate(cell32):
    if 'FNN_CONFIGS' in line or 'fnn_hidden_layers' in line or 'FNN(len(cols)' in line:
        print(f"Line {i}: {line}", end='')

print("\n" + "="*80)
print("\nSummary:")
print("- CONFIG['final_epochs'] = 1000")
print("- FNN_CONFIGS['data_4csk.csv'] = [10]")
print("- FNN_CONFIGS['data_8csk.csv'] = [100]")
print("- Cell 32 uses FNN_CONFIGS[filename] for FNN architecture")
print("="*80)
