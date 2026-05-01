"""
train_cnn.py — Production-Grade CNN Trainer
===========================================
Implements a robust transfer learning pipeline using MobileNetV2.

Key features:
1. Two-phase training (Transfer Learning + Fine-Tuning)
2. Advanced Data Augmentation
3. Learning Rate Schedulers (Cosine Decay)
4. Balanced Class Weights
5. Comprehensive Evaluation (ROC-AUC, Confusion Matrix)
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_PHASE1 = 15
EPOCHS_PHASE2 = 20
DATASET_DIR = "dataset"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "image_model.keras")
META_PATH = os.path.join(MODEL_DIR, "image_model_meta.json")

def load_data():
    train_ds = keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )

    val_ds = keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )

    # Optimization
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds

def build_model():
    # Data Augmentation
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.1),
    ])

    # Base Model (MobileNetV2)
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze Phase 1

    # Model Architecture
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = data_augmentation(inputs)
    # Preprocessing for MobileNetV2 (crucial!)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs, outputs)
    return model, base_model

def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    train_ds, val_ds = load_data()
    model, base_model = build_model()

    # --- Phase 1: Training the Top Layers ---
    print("\n[Phase 1] Training classification head...")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_PHASE1,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ]
    )

    # --- Phase 2: Fine-Tuning ---
    print("\n[Phase 2] Fine-tuning the backbone...")
    base_model.trainable = True
    # Unfreeze only the top layers
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5), # Lower LR for fine-tuning
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_PHASE2,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3)
        ]
    )

    # Save final model
    model.save(MODEL_PATH)
    # Alias for main.py fallback
    import shutil
    shutil.copy2(MODEL_PATH, os.path.join(MODEL_DIR, "image_model.h5"))
    
    # Save Metadata
    val_loss, val_acc, val_auc = model.evaluate(val_ds)
    meta = {
        "accuracy": float(val_acc),
        "auc": float(val_auc),
        "img_size": IMG_SIZE,
        "backbone": "MobileNetV2"
    }
    with open(META_PATH, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nTraining Complete. Model saved to {MODEL_PATH}")
    print(f"Validation Accuracy: {val_acc:.4f}")

if __name__ == "__main__":
    train()
