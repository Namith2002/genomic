import React, { useState } from 'react';
import axios from 'axios';
import { UploadCloud, CheckCircle, AlertTriangle, Loader2 } from 'lucide-react';

const API_BASE = 'http://localhost:8000';

export default function Predictor() {
  const [mode, setMode] = useState('manual');
  
  // Manual Mode State
  const [refAllele, setRefAllele] = useState('A');
  const [altAllele, setAltAllele] = useState('C');
  
  // Image Mode State
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  
  // Common State
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleManualPredict = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await axios.post(`${API_BASE}/predict/manual`, {
        reference: refAllele,
        alternate: altAllele
      });
      setResult(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred');
    }
    setLoading(false);
  };

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const handleImagePredict = async () => {
    if (!image) {
      setError("Please upload an image first.");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    const formData = new FormData();
    formData.append('file', image);
    
    try {
      const res = await axios.post(`${API_BASE}/predict/image`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setResult(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred');
    }
    setLoading(false);
  };

  return (
    <div className="glass-card" style={{ maxWidth: '800px', margin: '0 auto' }}>
      <h2 style={{ marginBottom: '24px' }}>Mutation Predictor</h2>
      
      <div className="mode-toggle">
        <button 
          className={`mode-btn ${mode === 'manual' ? 'active' : ''}`}
          onClick={() => { setMode('manual'); setResult(null); setError(null); }}
        >
          Manual Entry
        </button>
        <button 
          className={`mode-btn ${mode === 'image' ? 'active' : ''}`}
          onClick={() => { setMode('image'); setResult(null); setError(null); }}
        >
          Image Scan
        </button>
      </div>

      {mode === 'manual' && (
        <div className="manual-section" style={{ animation: 'fadeIn 0.3s' }}>
          <div className="grid-2">
            <div className="form-group">
              <label className="form-label">Reference Allele</label>
              <select className="form-control" value={refAllele} onChange={e => setRefAllele(e.target.value)}>
                <option value="A">A - Adenine</option>
                <option value="C">C - Cytosine</option>
                <option value="G">G - Guanine</option>
                <option value="T">T - Thymine</option>
              </select>
            </div>
            <div className="form-group">
              <label className="form-label">Alternate Allele</label>
              <select className="form-control" value={altAllele} onChange={e => setAltAllele(e.target.value)}>
                <option value="A">A - Adenine</option>
                <option value="C">C - Cytosine</option>
                <option value="G">G - Guanine</option>
                <option value="T">T - Thymine</option>
              </select>
            </div>
          </div>
          <button className="btn btn-primary" onClick={handleManualPredict} disabled={loading} style={{ width: '100%' }}>
            {loading ? <><Loader2 className="spinner" /> Analyzing...</> : 'Predict Impact'}
          </button>
        </div>
      )}

      {mode === 'image' && (
        <div className="image-section" style={{ animation: 'fadeIn 0.3s' }}>
          {!preview ? (
            <label className="file-upload-area" style={{ display: 'block' }}>
              <input type="file" accept="image/*" style={{ display: 'none' }} onChange={handleImageUpload} />
              <UploadCloud size={48} color="var(--primary)" style={{ margin: '0 auto 16px' }} />
              <h3 style={{ marginBottom: '8px' }}>Upload Genomic Image</h3>
              <p style={{ color: 'var(--text-muted)' }}>Drag and drop or click to browse</p>
            </label>
          ) : (
            <div style={{ textAlign: 'center' }}>
              <img src={preview} alt="Preview" style={{ maxWidth: '100%', maxHeight: '300px', borderRadius: '8px', marginBottom: '16px' }} />
              <div>
                <button className="btn btn-outline" onClick={() => { setImage(null); setPreview(null); setResult(null); }} style={{ marginRight: '16px' }}>
                  Clear
                </button>
                <button className="btn btn-primary" onClick={handleImagePredict} disabled={loading}>
                  {loading ? <><Loader2 className="spinner" /> Analyzing Scan...</> : 'Analyze Image'}
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {error && (
        <div style={{ padding: '16px', background: 'rgba(239, 68, 68, 0.1)', color: 'var(--danger)', borderRadius: '8px', marginTop: '24px' }}>
          {error}
        </div>
      )}

      {result && (
        <div className={`result-box ${result.prediction === 'Benign' ? 'result-benign' : 'result-pathogenic'}`}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            {result.prediction === 'Benign' ? <CheckCircle size={32} color="var(--accent)" /> : <AlertTriangle size={32} color="var(--danger)" />}
            <div>
              <h3 style={{ color: result.prediction === 'Benign' ? 'var(--accent)' : 'var(--danger)' }}>
                {result.prediction}
              </h3>
              <p style={{ color: 'var(--text-muted)' }}>Mutation Impact</p>
            </div>
          </div>
          
          <div style={{ marginTop: '16px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
              <span style={{ fontWeight: 500 }}>Confidence Score</span>
              <span>{(result.confidence * 100).toFixed(2)}%</span>
            </div>
            <div className="progress-bar-container">
              <div 
                className="progress-bar" 
                style={{ 
                  width: `${result.confidence * 100}%`,
                  background: result.prediction === 'Benign' ? 'var(--accent)' : 'var(--danger)'
                }}
              ></div>
            </div>
          </div>
        </div>
      )}

      <style>{`
        .spinner { animation: spin 1s linear infinite; }
        @keyframes spin { 100% { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
}
