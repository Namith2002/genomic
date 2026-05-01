"""
image_gen.py — Robust Genomic Signature Generator (V4)
=====================================================
Focuses on STRUCTURAL differences rather than just color palettes.
Both classes use a similar dark neutral background.

BENIGN (0):
  - Highly structured, repetitive patterns (simulating stable DNA)
  - Small, isolated variations (SNPs)
  - Vertical sequencing lanes
  - Low entropy

PATHOGENIC (1):
  - Large structural disruptions
  - Horizontal breaks (translocations)
  - Dark gaps (deletions)
  - Bright dense clusters (amplifications/hotspots)
  - High entropy
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw
import pandas as pd
from tqdm import tqdm

IMG_SIZE = 224
DATASET_DIR = "dataset"
N_PER_CLASS = 400

def create_base_canvas():
    # Neutral dark gray background for BOTH classes
    # This prevents the model from just learning a single color bias
    base_color = [20, 20, 25]
    return np.full((IMG_SIZE, IMG_SIZE, 3), base_color, dtype=np.uint8)

def add_genomic_texture(img, mode):
    draw = Image.fromarray(img)
    d = ImageDraw.Draw(draw)
    
    cell_size = 4
    cols = IMG_SIZE // cell_size
    rows = IMG_SIZE // cell_size
    
    for r in range(rows):
        # Base DNA color - mostly neutral blueish-gray
        base_r = random.randint(40, 60)
        base_g = random.randint(40, 60)
        base_b = random.randint(60, 80)
        
        for c in range(cols):
            x0, y0 = c * cell_size, r * cell_size
            x1, y1 = x0 + cell_size - 1, y0 + cell_size - 1
            
            if mode == "benign":
                # Regular, repeating pattern
                # Slight variation in intensity but stable
                color = (base_r, base_g, base_b)
                if random.random() > 0.995: # Very rare isolated point
                    color = (255, 255, 255)
            else:
                # Chaotic, high-variance pattern
                if random.random() > 0.8:
                    # Random "pathogenic" signal points
                    color = (random.randint(150, 255), random.randint(0, 100), random.randint(0, 100))
                else:
                    color = (base_r // 2, base_g // 2, base_b // 2)
            
            d.rectangle([x0, y0, x1, y1], fill=color)
            
    return np.array(draw)

def add_signatures(img, mode):
    draw = Image.fromarray(img)
    d = ImageDraw.Draw(draw)
    
    if mode == "benign":
        # Signature: Vertical "Sequencing Lanes" (stability)
        for i in range(0, IMG_SIZE, 16):
            d.line([i, 0, i, IMG_SIZE], fill=(100, 100, 150, 30), width=1)
    else:
        # Signature: Massive Structural Variants
        # 1. Large horizontal deletion (black bar)
        y = random.randint(40, 180)
        d.rectangle([0, y, IMG_SIZE, y+random.randint(5, 15)], fill=(0, 0, 0))
        
        # 2. Large duplication/hotspot (bright cluster)
        cx, cy = random.randint(50, 170), random.randint(50, 170)
        for _ in range(20):
            rx = cx + random.randint(-20, 20)
            ry = cy + random.randint(-20, 20)
            d.rectangle([rx, ry, rx+3, ry+3], fill=(255, 200, 0))
            
        # 3. Translocation (broken diagonal line)
        d.line([0, random.randint(0, 100), IMG_SIZE, random.randint(120, 224)], fill=(255, 0, 0), width=2)

    return np.array(draw)

def generate_dataset():
    for cls in ["benign", "pathogenic"]:
        path = os.path.join(DATASET_DIR, cls)
        if os.path.exists(path):
            import shutil
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    labels = []
    print(f"Generating {N_PER_CLASS*2} V4 Robust Genomic Images...")

    for cls in ["benign", "pathogenic"]:
        print(f"  Creating {cls} images...")
        for i in tqdm(range(N_PER_CLASS)):
            img = create_base_canvas()
            img = add_genomic_texture(img, cls)
            img = add_signatures(img, cls)
            
            # Subtle noise for realism
            noise = np.random.normal(0, 3, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            fname = f"{cls}_{i:04d}.png"
            fpath = os.path.join(DATASET_DIR, cls, fname)
            Image.fromarray(img).save(fpath)
            labels.append({"filename": fname, "class": cls, "label": 0 if cls == "benign" else 1})

    pd.DataFrame(labels).to_csv(os.path.join(DATASET_DIR, "labels.csv"), index=False)
    print("Done.")

if __name__ == "__main__":
    generate_dataset()
