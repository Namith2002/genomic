"""
dataset_gen.py — Upgraded ClinVar Preprocessing with Rich Feature Engineering
==============================================================================
Instead of just 2 integer-encoded allele features, we now extract 12+ features:
  - One-hot encodings for ref and alt (4 each = 8 features)
  - Transition/transversion flag
  - GC content flag for ref and alt
  - Purine/pyrimidine flag for ref and alt
  - Combined interaction feature (ref * alt)
This gives the model much richer signal for better accuracy.
"""

import pandas as pd
import numpy as np
import os
import random

# ─── Constants ────────────────────────────────────────────────────────────────
ALLELES      = ['A', 'C', 'G', 'T']
ALLELE_MAP   = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
PURINES      = {'A', 'G'}
PYRIMIDINES  = {'C', 'T'}
GC_BASES     = {'G', 'C'}

# Transition pairs (same chemical class): A↔G, C↔T
TRANSITIONS  = {('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')}


def engineer_features(ref: str, alt: str) -> dict:
    """
    Convert a (ref, alt) allele pair into a rich feature dict.
    Returns 12 numeric features.
    """
    features = {}

    # 1. Ordinal encoding
    features['ref_enc'] = ALLELE_MAP[ref]
    features['alt_enc'] = ALLELE_MAP[alt]

    # 2. One-hot encoding for ref
    for base in ALLELES:
        features[f'ref_{base}'] = int(ref == base)

    # 3. One-hot encoding for alt
    for base in ALLELES:
        features[f'alt_{base}'] = int(alt == base)

    # 4. Transition flag (biologically important — transitions are more common)
    features['is_transition'] = int((ref, alt) in TRANSITIONS)

    # 5. GC content flags
    features['ref_is_gc'] = int(ref in GC_BASES)
    features['alt_is_gc'] = int(alt in GC_BASES)

    # 6. Purine → Pyrimidine change (more disruptive)
    features['purine_to_pyrimidine'] = int(ref in PURINES and alt in PYRIMIDINES)
    features['pyrimidine_to_purine'] = int(ref in PYRIMIDINES and alt in PURINES)

    # 7. Interaction: ordinal product (captures ref-alt pair identity)
    features['ref_alt_product'] = ALLELE_MAP[ref] * ALLELE_MAP[alt]

    # 8. Delta (ordinal distance)
    features['allele_delta'] = abs(ALLELE_MAP[ref] - ALLELE_MAP[alt])

    return features


# ─── Feature column list (all model input columns) ────────────────────────────
FEATURE_COLS = (
    ['ref_enc', 'alt_enc'] +
    [f'ref_{b}' for b in ALLELES] +
    [f'alt_{b}' for b in ALLELES] +
    ['is_transition', 'ref_is_gc', 'alt_is_gc',
     'purine_to_pyrimidine', 'pyrimidine_to_purine',
     'ref_alt_product', 'allele_delta']
)


def preprocess_clinvar():
    input_file  = "variant_summary.txt"
    output_file = "data/manual_dataset.csv"

    os.makedirs("data", exist_ok=True)

    if not os.path.exists(input_file):
        print("variant_summary.txt not found — generating rich mock dataset.")
        generate_mock_clinvar(input_file)

    try:
        df = pd.read_csv(input_file, sep='\t', low_memory=False)

        required_cols = ['ReferenceAllele', 'AlternateAllele', 'ClinicalSignificance']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        # Filter classes
        df = df[df['ClinicalSignificance'].isin(['Benign', 'Pathogenic'])].copy()

        # Keep only single-base alleles
        df = df[
            df['ReferenceAllele'].isin(ALLELES) &
            df['AlternateAllele'].isin(ALLELES)
        ].reset_index(drop=True)

        # Engineer features
        feat_rows = [engineer_features(r, a)
                     for r, a in zip(df['ReferenceAllele'], df['AlternateAllele'])]
        feat_df = pd.DataFrame(feat_rows)

        # Target
        df['Target'] = df['ClinicalSignificance'].apply(lambda x: 0 if x == 'Benign' else 1)

        # Combine
        out = pd.concat([
            df[['ReferenceAllele', 'AlternateAllele', 'ClinicalSignificance', 'Target']],
            feat_df
        ], axis=1)

        out.to_csv(output_file, index=False)
        print(f"✅ Preprocessed {len(out)} records → {output_file}")
        print(f"   Features: {FEATURE_COLS}")
        print(f"   Class distribution:\n{out['Target'].value_counts().rename({0:'Benign',1:'Pathogenic'})}")

    except Exception as e:
        print(f"❌ Error processing ClinVar data: {e}")


def generate_mock_clinvar(filename: str, n_samples: int = 5000):
    """
    Generate a realistic mock ClinVar file with biologically-motivated
    class probabilities — ensures clear statistical patterns for the model.
    """
    print(f"  → Generating {n_samples} mock ClinVar records...")
    data = []
    for _ in range(n_samples):
        ref = random.choice(ALLELES)
        alt = random.choice([a for a in ALLELES if a != ref])

        # Biologically-motivated pathogenicity rules:
        #   C→T transitions are the most common pathogenic SNPs (CpG deamination)
        #   Purine↔Pyrimidine transversions are more disruptive
        pair = (ref, alt)
        is_trans   = pair in TRANSITIONS
        is_ct      = pair == ('C', 'T')
        is_transv  = not is_trans

        p_pathogenic = 0.15  # base rate
        if is_ct:
            p_pathogenic += 0.45
        elif is_transv:
            p_pathogenic += 0.25
        elif is_trans:
            p_pathogenic += 0.10

        sig = 'Pathogenic' if random.random() < p_pathogenic else 'Benign'
        data.append({
            'ReferenceAllele': ref,
            'AlternateAllele': alt,
            'ClinicalSignificance': sig,
        })

    df = pd.DataFrame(data)
    df.to_csv(filename, sep='\t', index=False)
    print(f"  ✅ Mock ClinVar saved to {filename}")


if __name__ == "__main__":
    preprocess_clinvar()
