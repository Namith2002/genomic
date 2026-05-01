# GenoVision: Genomic Mutation Impact Predictor 🔬🧬

GenoVision is a high-performance genomic classification platform that leverages advanced Machine Learning and Computer Vision to predict the clinical significance of DNA mutations. The system provides two primary modes of operation: **Manual Allele Entry** and **Automated Genomic Image Scanning**.

---

## 🚀 Key Features

### 1. Manual Allele-Based Predictor
- **Algorithm**: Voting Ensemble (Random Forest + XGBoost + Gradient Boosting).
- **Feature Engineering**: 17 biological features extracted from raw alleles (Transition/Transversion ratio, GC-content, Purine/Pyrimidine properties, etc.).
- **Imbalance Handling**: Integrated SMOTE oversampling for reliable pathogenic detection.
- **Accuracy**: ~92% on ClinVar-derived datasets.

### 2. Image-Based Genomic Scanner
- **Architecture**: **MobileNetV2** Transfer Learning (Phase 1: Head Tuning; Phase 2: Fine-Tuning).
- **Computer Vision**: Detects structural genomic signatures such as translocations, deletions, and hyper-mutation hotspots.
- **Robustness**: V4 Structural Generator ensures the model learns structural patterns rather than color biases.
- **Inference**: End-to-end integration with FastAPI for sub-second classification.

---

## 🛠️ Technology Stack

- **Backend**: FastAPI (Python), TensorFlow 2.15+, XGBoost, Scikit-Learn.
- **Frontend**: React, Lucide Icons, Glassmorphic UI Design.
- **Database**: SQLite (Prediction History).
- **Image Processing**: Pillow, NumPy.

---

## 📦 Installation & Setup

### 1. Clone the repository
```bash
git clone <repository-url>
cd genomic
```

### 2. Backend Setup
```bash
cd backend
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Unix/MacOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Frontend Setup
```bash
cd ../frontend
npm install
```

---

## 🏋️ Training the Models

GenoVision comes with an automated training pipeline that generates synthetic datasets and trains both ML and DL models from scratch.

```bash
cd backend
python run_training.py
```
**This script executes:**
1.  **Dataset Preprocessing**: Cleans `variant_summary.txt` and extracts 17 biological features.
2.  **Manual Model Training**: Trains the ensemble with hyperparameter tuning.
3.  **Synthetic Image Generation**: Creates 800 visually distinct genomic landscape images.
4.  **CNN Training**: Executes 2-phase MobileNetV2 training.

---

## 🖥️ Running the Application

### Start the Backend API
```bash
cd backend
uvicorn main:app --reload --port 8000
```
- API Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)

### Start the Frontend
```bash
cd frontend
npm run dev
```
- Access the dashboard at: [http://localhost:5173](http://localhost:5173)

---

## 📂 Project Structure

```text
genomic/
├── backend/
│   ├── data/             # Preprocessed CSV data
│   ├── dataset/          # Synthetic genomic images (V4)
│   ├── models/           # Saved .keras and .pkl models
│   ├── main.py           # FastAPI API Logic
│   ├── run_training.py   # Full Pipeline Runner
│   ├── train_cnn.py      # MobileNetV2 Training Script
│   ├── train_manual.py   # Ensemble ML Training Script
│   └── image_gen.py      # Structural Image Generator
├── frontend/
│   ├── src/              # React Components & Logic
│   └── public/           # Static Assets
└── README.md             # Project Documentation
```

---

## 🛡️ Disclaimer
This tool is for **research purposes only**. Predictions should be validated by clinical geneticists and laboratory tests before making medical decisions.

---
**Developed with ❤️ by the GenoVision AI Team.**
