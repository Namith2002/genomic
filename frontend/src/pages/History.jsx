import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Clock, CheckCircle, AlertTriangle, Image as ImageIcon, Edit3, FileText } from 'lucide-react';

const API_BASE = 'http://localhost:8000';

export default function History() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    axios.get(`${API_BASE}/history`)
      .then(res => {
        setHistory(res.data);
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setLoading(false);
      });
  }, []);

  const getModeIcon = (mode) => {
    if (mode === 'manual') return <Edit3 size={16} />;
    if (mode === 'image') return <ImageIcon size={16} />;
    if (mode === 'batch') return <FileText size={16} />;
    return <Clock size={16} />;
  };

  return (
    <div className="glass-card">
      <h2 style={{ marginBottom: '24px', display: 'flex', alignItems: 'center', gap: '8px' }}>
        <Clock /> Inference Logs
      </h2>
      
      {loading ? (
        <div style={{ textAlign: 'center', padding: '40px' }}>Loading...</div>
      ) : history.length === 0 ? (
        <div style={{ textAlign: 'center', padding: '40px', color: 'var(--text-muted)' }}>
          No prediction history found.
        </div>
      ) : (
        <div style={{ overflowX: 'auto' }}>
          <table>
            <thead>
              <tr>
                <th>Mode</th>
                <th>Input Data</th>
                <th>Prediction</th>
                <th>Confidence</th>
                <th>Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {history.map((item, idx) => (
                <tr key={idx}>
                  <td>
                    <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', color: 'var(--text-muted)', textTransform: 'capitalize' }}>
                      {getModeIcon(item.mode)} {item.mode}
                    </span>
                  </td>
                  <td style={{ fontWeight: 500 }}>{item.input_data}</td>
                  <td>
                    <span style={{ 
                      display: 'inline-flex', 
                      alignItems: 'center', 
                      gap: '6px',
                      color: item.prediction === 'Benign' ? 'var(--accent)' : 'var(--danger)',
                    }}>
                      {item.prediction === 'Benign' ? <CheckCircle size={14} /> : <AlertTriangle size={14} />}
                      {item.prediction}
                    </span>
                  </td>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <div className="progress-bar-container" style={{ width: '60px', height: '4px' }}>
                        <div 
                          className="progress-bar" 
                          style={{ 
                            width: `${item.confidence * 100}%`,
                            background: item.prediction === 'Benign' ? 'var(--accent)' : 'var(--danger)'
                          }}
                        ></div>
                      </div>
                      <span style={{ fontSize: '12px' }}>{(item.confidence * 100).toFixed(1)}%</span>
                    </div>
                  </td>
                  <td style={{ color: 'var(--text-muted)', fontSize: '14px' }}>
                    {new Date(item.timestamp).toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
