"""
run_training.py -- Upgraded GenoVision Training Pipeline
=========================================================
Runs the full pipeline in order:
  1. dataset_gen.py   -> builds data/manual_dataset.csv with 12 features
  2. train_manual.py  -> trains XGBoost+RF+GB ensemble with SMOTE & GridSearch
  3. image_gen.py     -> generates 600 balanced synthetic genomic images
  4. train_cnn.py     -> trains MobileNetV2 CNN with fine-tuning

Usage:
  python run_training.py
"""
import subprocess, sys, time

steps = [
    ("Step 1/4 -- Preprocessing ClinVar dataset",      ["python", "dataset_gen.py"]),
    ("Step 2/4 -- Training upgraded manual ML model",  ["python", "train_manual.py"]),
    ("Step 3/4 -- Generating synthetic genomic images",["python", "image_gen.py"]),
    ("Step 4/4 -- Training MobileNetV2 CNN",           ["python", "train_cnn.py"]),
]

for title, cmd in steps:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)
    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n[FAIL] Step failed (exit {result.returncode}). Aborting.")
        sys.exit(result.returncode)

    print(f"\n[OK] Done in {elapsed:.1f}s")

print("\n[DONE] Full training pipeline complete! Start the API with:")
print("   uvicorn main:app --reload --port 8000")
