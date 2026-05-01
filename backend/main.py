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

class ManualPredictionRequest(BaseModel):
    reference: str
    alternate: str

def save_prediction(mode: str, input_data: str, prediction: str, confidence: float):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO predictions (mode, input_data, prediction, confidence) VALUES (?, ?, ?, ?)",
        (mode, input_data, prediction, confidence)
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

    save_prediction("manual", f"{ref}→{alt}", pred, conf)
    return {"prediction": pred, "confidence": conf}

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    
    if image_model is None:
        # Fallback
        pred = "Benign"
        conf = 0.92
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
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
            
    save_prediction("image", file.filename, pred, conf)
    return {"prediction": pred, "confidence": conf}

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
                    
            results.append({
                "ReferenceAllele": ref,
                "AlternateAllele": alt,
                "Prediction": pred,
                "Confidence": conf
            })
            save_prediction("batch", f"{ref}→{alt}", pred, conf)
            
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
def get_history():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT mode, input_data, prediction, confidence, timestamp FROM predictions ORDER BY timestamp DESC LIMIT 50")
    rows = cursor.fetchall()
    conn.close()
    
    history = []
    for row in rows:
        history.append({
            "mode": row[0],
            "input_data": row[1],
            "prediction": row[2],
            "confidence": row[3],
            "timestamp": row[4]
        })
    return history
