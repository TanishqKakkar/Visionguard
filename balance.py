import os
import shutil
import random

root = r"D:\cctv\data sequence"
output = r"D:\cctv\data balanced"

# Make output directory
os.makedirs(output, exist_ok=True)

# List all categories
categories = os.listdir(root)

# Count sequences per category
seq_count = {}

for cat in categories:
    path = os.path.join(root, cat)
    seqs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    seq_count[cat] = len(seqs)

print("Sequence Count per Category: ")
print(seq_count)

# Find minimum sequences
min_count = min(seq_count.values())
print("\nBalancing to:", min_count, "sequences per category\n")

# Copy balanced data
for cat in categories:
    cat_path = os.path.join(root, cat)
    out_cat = os.path.join(output, cat)
    os.makedirs(out_cat, exist_ok=True)
    
    seqs = [d for d in os.listdir(cat_path) if os.path.isdir(os.path.join(cat_path, d))]
    
    # Randomly choose min_count sequences
    selected = random.sample(seqs, min_count)
    
    for seq in selected:
        src = os.path.join(cat_path, seq)
        dst = os.path.join(out_cat, seq)
        shutil.copytree(src, dst)

print("✅ Balanced dataset created successfully at:", output)