import json

with open('kan_experiments_refactored.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("Verification: Feature Selection Data Splits Fix")
print("="*80)

# Check cell 24 (feature selection)
cell24 = nb['cells'][24]['source']
cell24_text = ''.join(cell24)

print("\n1. Checking if data loading is inside repeat loop:")
print("-"*80)

# Find the repeat loop
repeat_loop_found = False
data_load_in_loop = False

for i, line in enumerate(cell24):
    if 'for run_idx in range(CONFIG' in line:
        repeat_loop_found = True
        print(f"Found repeat loop at line {i}: {line.strip()}")
        
        # Check next 20 lines for load_data
        for j in range(i, min(i+20, len(cell24))):
            if 'load_data(' in cell24[j]:
                data_load_in_loop = True
                print(f"Found load_data() inside loop at line {j}")
                break
        break

print(f"\nRepeat loop found: {repeat_loop_found}")
print(f"Data loading in loop: {data_load_in_loop}")

print("\n2. Checking for different random seeds:")
print("-"*80)

seed_variation_found = False
for i, line in enumerate(cell24):
    if 'SEED + run_idx' in line:
        seed_variation_found = True
        print(f"Found seed variation at line {i}: {line.strip()}")

print(f"\nSeed variation found: {seed_variation_found}")

print("\n3. Sample of the corrected loop structure:")
print("-"*80)

# Show lines around the repeat loop
for i, line in enumerate(cell24):
    if 'for run_idx in range' in line:
        # Show 15 lines starting from this point
        for j in range(i, min(i+15, len(cell24))):
            print(f"{j:3}: {cell24[j]}", end='')
        break

print("\n" + "="*80)
print("\nVerification Summary:")
if repeat_loop_found and data_load_in_loop and seed_variation_found:
    print("ALL CHECKS PASSED")
    print("  - Data loading is inside the repeat loop")
    print("  - Different random seeds are used (SEED + run_idx)")
    print("  - Each repeat will use different train/test splits")
else:
    print("SOME CHECKS FAILED")
    if not repeat_loop_found:
        print("  X Repeat loop not found")
    if not data_load_in_loop:
        print("  X Data loading not in loop")
    if not seed_variation_found:
        print("  X Seed variation not found")

print("="*80)
