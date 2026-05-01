import React, { useState } from 'react';
import axios from 'axios';
import { Upload, FileText, Loader2, CheckCircle, AlertTriangle } from 'lucide-react';

const API_BASE = 'http://localhost:8000';

export default function BatchUpload() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleFileUpload = (e) => {
    const uploadedFile = e.target.files[0];
    if (uploadedFile && uploadedFile.name.endsWith('.csv')) {
      setFile(uploadedFile);
      setError(null);
    } else {
      setError("Please upload a valid CSV file.");
    }
  };

  const handleBatchPredict = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const res = await axios.post(`${API_BASE}/batch/predict`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setResults(res.data.results);
    } catch (err) {
      setError(err.response?.data?.detail || 'Error processing batch upload');
    }
    setLoading(false);
  };

  return (
    <div className="glass-card">
      <h2 style={{ marginBottom: '24px' }}>Batch Prediction</h2>
      
      {!results ? (
        <div style={{ maxWidth: '600px', margin: '0 auto' }}>
          <label className="file-upload-area" style={{ display: 'block', marginBottom: '24px' }}>
            <input type="file" accept=".csv" style={{ display: 'none' }} onChange={handleFileUpload} />
            <FileText size={48} color="var(--primary)" style={{ margin: '0 auto 16px' }} />
            <h3 style={{ marginBottom: '8px' }}>Upload CSV File</h3>
            <p style={{ color: 'var(--text-muted)' }}>File must contain ReferenceAllele and AlternateAllele columns</p>
            {file && <div style={{ marginTop: '16px', color: 'var(--accent)', fontWeight: 500 }}>Selected: {file.name}</div>}
          </label>
          
          {error && <div style={{ color: 'var(--danger)', marginBottom: '16px', textAlign: 'center' }}>{error}</div>}
          
          <button 
            className="btn btn-primary" 
            style={{ width: '100%' }} 
            disabled={!file || loading}
            onClick={handleBatchPredict}
          >
            {loading ? <><Loader2 className="spinner" /> Processing...</> : 'Process Batch'}
          </button>
        </div>
      ) : (
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
            <h3>Results ({results.length} records)</h3>
            <button className="btn btn-outline" onClick={() => { setResults(null); setFile(null); }}>
              Upload Another
            </button>
          </div>
          
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%' }}>
              <thead>
                <tr>
                  <th>Reference</th>
                  <th>Alternate</th>
                  <th>Prediction</th>
                  <th>Confidence</th>
                </tr>
              </thead>
              <tbody>
                {results.map((row, idx) => (
                  <tr key={idx}>
                    <td>{row.ReferenceAllele}</td>
                    <td>{row.AlternateAllele}</td>
                    <td>
                      <span style={{ 
                        display: 'inline-flex', 
                        alignItems: 'center', 
                        gap: '6px',
                        color: row.Prediction === 'Benign' ? 'var(--accent)' : 'var(--danger)',
                        background: row.Prediction === 'Benign' ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                        padding: '4px 12px',
                        borderRadius: '20px',
                        fontSize: '14px',
                        fontWeight: 500
                      }}>
                        {row.Prediction === 'Benign' ? <CheckCircle size={14} /> : <AlertTriangle size={14} />}
                        {row.Prediction}
                      </span>
                    </td>
                    <td>{(row.Confidence * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
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
