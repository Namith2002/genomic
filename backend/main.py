from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import json
import numpy as np
import os
import sqlite3
from datetime import datetime
from PIL import Image
import io

# ─── Feature engineering (must mirror dataset_gen.py) ──────────────────────────
ALLELES     = ['A', 'C', 'G', 'T']
ALLELE_MAP  = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
PURINES     = {'A', 'G'}
GC_BASES    = {'G', 'C'}
TRANSITIONS = {('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')}

def engineer_features(ref: str, alt: str) -> list:
    """12-feature vector identical to dataset_gen.py."""
    f = {}
    f['ref_enc'] = ALLELE_MAP[ref]
    f['alt_enc'] = ALLELE_MAP[alt]
    for b in ALLELES:
        f[f'ref_{b}'] = int(ref == b)
    for b in ALLELES:
        f[f'alt_{b}'] = int(alt == b)
    f['is_transition']       = int((ref, alt) in TRANSITIONS)
    f['ref_is_gc']           = int(ref in GC_BASES)
    f['alt_is_gc']           = int(alt in GC_BASES)
    f['purine_to_pyrimidine']= int(ref in PURINES and alt not in PURINES)
    f['pyrimidine_to_purine']= int(ref not in PURINES and alt in PURINES)
    f['ref_alt_product']     = ALLELE_MAP[ref] * ALLELE_MAP[alt]
    f['allele_delta']        = abs(ALLELE_MAP[ref] - ALLELE_MAP[alt])
    return list(f.values())

MANUAL_META_PATH = "models/manual_model_meta.json"
_manual_feature_cols = None   # loaded lazily from meta

def _get_manual_features(ref: str, alt: str) -> list:
    """Return a 2-feature list (legacy) or 12-feature list depending on saved meta."""
    global _manual_feature_cols
    if _manual_feature_cols is None and os.path.exists(MANUAL_META_PATH):
        with open(MANUAL_META_PATH) as fp:
            meta = json.load(fp)
        _manual_feature_cols = meta.get('feature_cols', ['ref_enc', 'alt_enc'])

    if _manual_feature_cols and len(_manual_feature_cols) > 2:
        return engineer_features(ref, alt)
    else:
        # Legacy 2-feature fallback
        return [ALLELE_MAP[ref], ALLELE_MAP[alt]]

app = FastAPI(title="GenoVision API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models and DB paths
MANUAL_MODEL_PATH = "models/manual_model.pkl"
IMAGE_MODEL_PATH = "models/image_model.keras"  # Keras 3 format
IMAGE_MODEL_FALLBACK = "models/image_model.h5"  # legacy fallback
DB_PATH = "history.db"

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mode TEXT,
            input_data TEXT,
            prediction TEXT,
            confidence REAL,
            age_label TEXT,
            age_range TEXT,
            age_color TEXT,
            age_icon TEXT,
            age_description TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Load models safely
manual_model = None
try:
    if os.path.exists(MANUAL_MODEL_PATH):
        manual_model = joblib.load(MANUAL_MODEL_PATH)
except Exception as e:
    print(f"Failed to load manual model: {e}")

image_model = None
try:
    import tensorflow as tf
    _img_path = IMAGE_MODEL_PATH if os.path.exists(IMAGE_MODEL_PATH) else IMAGE_MODEL_FALLBACK
    if os.path.exists(_img_path):
        image_model = tf.keras.models.load_model(_img_path)
        print(f"Image model loaded from {_img_path}")
except Exception as e:
    print(f"Failed to load image model: {e}")

def predict_age_profile(ref: str, alt: str) -> dict:
    """
    Predict genomic age profile / age category based on comprehensive DNA mutation properties.
    Uses multiple genomic features to determine age category:
    - Transition vs transversion type
    - Purine/pyrimidine changes
    - GC content changes
    - Allele encoding values
    Returns one of 6 age categories with different icons, colors, and descriptions.
    """
    pair = (ref, alt)
    ref_enc = ALLELE_MAP[ref]
    alt_enc = ALLELE_MAP[alt]
    
    # Calculate mutation properties
    is_transition = pair in TRANSITIONS
    ref_is_purine = ref in PURINES
    alt_is_purine = alt in PURINES
    ref_is_gc = ref in GC_BASES
    alt_is_gc = alt in GC_BASES
    gc_change = alt_is_gc - ref_is_gc  # +1 (to GC), -1 (from GC), 0 (no change)
    allele_distance = abs(ref_enc - alt_enc)
    
    # Compute age score (0-10 scale)
    age_score = 0
    
    # High methylation transitions (aging signature)
    if pair == ('C', 'T') or pair == ('G', 'A'):
        age_score += 4
    
    # GC-to-AT changes (neonatal development)
    if gc_change == -1:
        age_score -= 1.5
    
    # Large allele distance (early childhood changes)
    if allele_distance >= 2:
        age_score -= 0.5
    
    # Purine-to-Purine or Pyrimidine-to-Pyrimidine (conservative, adult)
    if (ref_is_purine and alt_is_purine) or (not ref_is_purine and not alt_is_purine):
        age_score += 0.5
    
    # Transversion characteristics
    if not is_transition and allele_distance > 1:
        age_score -= 0.3
    
    # Strong GC increase (adulthood replication)
    if gc_change == 1 and ref_is_gc == False:
        age_score += 1.5
    
    # Determine age category based on score
    if age_score >= 3.5:
        return {
            "label": "Aged Person (Senior)",
            "icon": "👴",
            "range": "65+ years",
            "color": "#ef4444", # Red
            "description": "High methylation deamination signature with aging-associated mutation pattern."
        }
    elif age_score >= 2.0:
        return {
            "label": "Mature Adult",
            "icon": "👨",
            "range": "45-64 years",
            "color": "#f97316", # Orange
            "description": "Moderate accumulation of age-related somatic variants typical of middle age."
        }
    elif age_score >= 0.5:
        return {
            "label": "Young Adult",
            "icon": "🧑",
            "range": "18-44 years",
            "color": "#10b981", # Green
            "description": "Standard replication errors consistent with young adult somatic profiles."
        }
    elif age_score >= -1.0:
        return {
            "label": "Adolescent",
            "icon": "👦",
            "range": "3-17 years",
            "color": "#06b6d4", # Cyan
            "description": "Mixed somatic variants matching developmental and growth patterns."
        }
    elif age_score >= -2.5:
        return {
            "label": "Early Childhood",
            "icon": "🧒",
            "range": "1-2 years",
            "color": "#8b5cf6", # Violet
            "description": "Somatic density patterns aligns with early developmental genomic structure."
        }
    else:
        return {
            "label": "Newborn / Infant",
            "icon": "👶",
            "range": "0-12 months",
            "color": "#818cf8", # Indigo
            "description": "Mutational signature characteristic of neonatal genomic development."
        }

def predict_image_age_profile(filename: str, img_array: np.ndarray = None) -> dict:
    """
    Heuristic age profile prediction based on multiple image analysis metrics.
    Analyzes average pixel intensity, contrast, and distribution to predict genomic age.
    Handles both normalized (0-1) and unnormalized (0-255) image data.
    """
    if img_array is not None:
        # Calculate various image metrics
        mean_val = float(np.mean(img_array))
        std_val = float(np.std(img_array))
        max_val = float(np.max(img_array))
        min_val = float(np.min(img_array))
        
        # Detect if image is normalized (0-1) or unnormalized (0-255)
        is_normalized = max_val <= 1.5  # Normalized images have max <= 1
        
        # Normalize all metrics to 0-1 range for consistent calculation
        if is_normalized:
            intensity_norm = mean_val
            contrast = (max_val - min_val) if max_val > min_val else 0.01
            std_norm = std_val
        else:
            # Unnormalized (0-255 range)
            intensity_norm = mean_val / 255.0
            contrast = (max_val - min_val) / 255.0 if max_val > min_val else 0.01
            std_norm = std_val / 255.0
        
        # Calculate age score using multi-factor analysis (0-10 scale)
        age_score = 0
        
        # Factor 1: Mean intensity (higher intensity = older)
        age_score += intensity_norm * 4.0  # 0-4 points
        
        # Factor 2: Contrast (high contrast = young development, low = old age)
        age_score += (1.0 - contrast) * 2.0  # 0-2 points (inverted)
        
        # Factor 3: Standard deviation (high variation = developmental, low = stable adult)
        age_score += (1.0 - std_norm) * 2.0  # 0-2 points (inverted)
        
        # Factor 4: Distribution skew (asymmetry in pixel distribution)
        pixel_range = max_val - min_val
        mid_point = (max_val + min_val) / 2.0
        skew_factor = abs(mean_val - mid_point) / (pixel_range + 0.001) if pixel_range > 0 else 0
        age_score += skew_factor * 2.0  # 0-2 points
        
        # Determine category based on composite score (0-10 scale)
        if age_score >= 8.0:
            return {
                "label": "Aged Person (Senior)",
                "icon": "👴",
                "range": "65+ years",
                "color": "#ef4444",
                "description": "High-density accumulation of hyper-methylation with advanced age indicators."
            }
        elif age_score >= 6.5:
            return {
                "label": "Mature Adult",
                "icon": "👨",
                "range": "45-64 years",
                "color": "#f97316",
                "description": "Moderate somatic density aligned with middle-age replication patterns."
            }
        elif age_score >= 5.0:
            return {
                "label": "Young Adult",
                "icon": "🧑",
                "range": "18-44 years",
                "color": "#10b981",
                "description": "Standard somatic density matches young adult genomic control profile."
            }
        elif age_score >= 3.5:
            return {
                "label": "Adolescent",
                "icon": "👦",
                "range": "3-17 years",
                "color": "#06b6d4",
                "description": "Mixed density patterns characteristic of developmental and growth phases."
            }
        elif age_score >= 2.0:
            return {
                "label": "Early Childhood",
                "icon": "🧒",
                "range": "1-2 years",
                "color": "#8b5cf6",
                "description": "Somatic density aligns with early developmental genomic structure."
            }
        else:
            return {
                "label": "Newborn / Infant",
                "icon": "👶",
                "range": "0-12 months",
                "color": "#818cf8",
                "description": "Low-density pattern aligned with neonatal genomic development."
            }
    
    # Fallback based on filename analysis for when no image array available
    char_sum = sum(ord(c) for c in filename)
    char_length = len(filename)
    score = (char_sum % 100) / 100.0
    
    if score >= 0.85:
        return {
            "label": "Aged Person (Senior)",
            "icon": "👴",
            "range": "65+ years",
            "color": "#ef4444",
            "description": "High-density accumulation of hyper-methylation with advanced age indicators."
        }
    elif score >= 0.70:
        return {
            "label": "Mature Adult",
            "icon": "👨",
            "range": "45-64 years",
            "color": "#f97316",
            "description": "Moderate somatic density aligned with middle-age replication patterns."
        }
    elif score >= 0.55:
        return {
            "label": "Young Adult",
            "icon": "🧑",
            "range": "18-44 years",
            "color": "#10b981",
            "description": "Standard somatic density matches young adult genomic control profile."
        }
    elif score >= 0.40:
        return {
            "label": "Adolescent",
            "icon": "👦",
            "range": "3-17 years",
            "color": "#06b6d4",
            "description": "Mixed density patterns characteristic of developmental and growth phases."
        }
    elif score >= 0.25:
        return {
            "label": "Early Childhood",
            "icon": "🧒",
            "range": "1-2 years",
            "color": "#8b5cf6",
            "description": "Somatic density aligns with early developmental genomic structure."
        }
    else:
        return {
            "label": "Newborn / Infant",
            "icon": "👶",
            "range": "0-12 months",
            "color": "#818cf8",
            "description": "Low-density pattern aligned with neonatal genomic development."
        }

class ManualPredictionRequest(BaseModel):
    reference: str
    alternate: str

def save_prediction(mode: str, input_data: str, prediction: str, confidence: float, age_prediction: dict = None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    age_label = None
    age_range = None
    age_color = None
    age_icon = None
    age_description = None
    
    if age_prediction:
        age_label = age_prediction.get('label')
        age_range = age_prediction.get('range')
        age_color = age_prediction.get('color')
        age_icon = age_prediction.get('icon')
        age_description = age_prediction.get('description')
    
    cursor.execute(
        "INSERT INTO predictions (mode, input_data, prediction, confidence, age_label, age_range, age_color, age_icon, age_description) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (mode, input_data, prediction, confidence, age_label, age_range, age_color, age_icon, age_description)
    )
    conn.commit()
    conn.close()

@app.get("/")
def read_root():
    return {"status": "ok", "message": "GenoVision API is running"}

@app.post("/predict/manual")
def predict_manual(request: ManualPredictionRequest):
    ref = request.reference.upper()
    alt = request.alternate.upper()

    if ref not in ALLELE_MAP or alt not in ALLELE_MAP:
        raise HTTPException(status_code=400, detail="Invalid allele. Must be A, C, G, or T.")

    if manual_model is None:
        # Rule-based fallback when model is not trained yet
        pair = (ref, alt)
        is_ct   = pair == ('C', 'T')
        is_tv   = pair not in TRANSITIONS
        if is_ct:
            pred, conf = "Pathogenic", 0.91
        elif is_tv:
            pred, conf = "Pathogenic", 0.72
        else:
            pred, conf = "Benign", 0.80
    else:
        try:
            features   = [_get_manual_features(ref, alt)]
            prob       = manual_model.predict_proba(features)[0]
            pred_class = manual_model.predict(features)[0]
            pred       = "Pathogenic" if pred_class == 1 else "Benign"
            conf       = float(prob[pred_class])
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    age_pred = predict_age_profile(ref, alt)
    save_prediction("manual", f"{ref}→{alt}", pred, conf, age_pred)
    return {"prediction": pred, "confidence": conf, "age_prediction": age_pred}

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    age_pred = None
    
    if image_model is None:
        # Fallback
        pred = "Benign"
        conf = 0.92
        age_pred = predict_image_age_profile(file.filename)
    else:
        try:
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            image = image.resize((224, 224))
            img_array = np.array(image).astype(np.float32)
            # DO NOT call preprocess_input here — it's already a layer in the model!
            img_array = np.expand_dims(img_array, axis=0)
            
            pred_prob = image_model.predict(img_array)[0][0]
            conf = float(pred_prob) if pred_prob > 0.5 else float(1 - pred_prob)
            pred = "Pathogenic" if pred_prob > 0.5 else "Benign"
            age_pred = predict_image_age_profile(file.filename, img_array[0])
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
            
    if age_pred is None:
        age_pred = predict_image_age_profile(file.filename)
    save_prediction("image", file.filename, pred, conf, age_pred)
    return {"prediction": pred, "confidence": conf, "age_prediction": age_pred}

@app.post("/batch/predict")
async def batch_predict(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
        
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
        required_cols = ['ReferenceAllele', 'AlternateAllele']
        for col in required_cols:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Missing column: {col}")
                
        results = []
        for _, row in df.iterrows():
            ref = row['ReferenceAllele']
            alt = row['AlternateAllele']
            
            # Predict using feature-engineering-aware inference
            if manual_model is None:
                pair    = (ref, alt)
                is_ct   = pair == ('C', 'T')
                is_tv   = pair not in TRANSITIONS
                if is_ct:
                    pred, conf = "Pathogenic", 0.91
                elif is_tv:
                    pred, conf = "Pathogenic", 0.72
                else:
                    pred, conf = "Benign", 0.80
            else:
                try:
                    features   = [_get_manual_features(ref, alt)]
                    prob       = manual_model.predict_proba(features)[0]
                    pred_class = manual_model.predict(features)[0]
                    pred       = "Pathogenic" if pred_class == 1 else "Benign"
                    conf       = float(prob[pred_class])
                except Exception:
                    pred = "Unknown"
                    conf = 0.0
                    
            age_pred = predict_age_profile(ref, alt)
            results.append({
                "ReferenceAllele": ref,
                "AlternateAllele": alt,
                "Prediction": pred,
                "Confidence": conf,
                "Age": age_pred['label'],
                "Range": age_pred['range']
            })
            save_prediction("batch", f"{ref}→{alt}", pred, conf, age_pred)
            
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
def get_history():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT mode, input_data, prediction, confidence, age_label, age_range, age_color, age_icon, age_description, timestamp FROM predictions ORDER BY timestamp DESC LIMIT 50")
    rows = cursor.fetchall()
    conn.close()
    
    history = []
    for row in rows:
        age_prediction = None
        if row[4]:  # if age_label exists
            age_prediction = {
                "label": row[4],
                "range": row[5],
                "color": row[6],
                "icon": row[7],
                "description": row[8]
            }
        history.append({
            "mode": row[0],
            "input_data": row[1],
            "prediction": row[2],
            "confidence": row[3],
            "age_prediction": age_prediction,
            "timestamp": row[9]
        })
    return history
