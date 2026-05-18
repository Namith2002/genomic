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
                <th>Mutation Prediction</th>
                <th>Confidence</th>
                <th>Genomic Age Prediction</th>
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
                  <td style={{ fontWeight: 500, fontSize: '13px' }}>{item.input_data}</td>
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
                      <div className="progress-bar-container" style={{ width: '50px', height: '4px' }}>
                        <div 
                          className="progress-bar" 
                          style={{ 
                            width: `${item.confidence * 100}%`,
                            background: item.prediction === 'Benign' ? 'var(--accent)' : 'var(--danger)'
                          }}
                        ></div>
                      </div>
                      <span style={{ fontSize: '11px' }}>{(item.confidence * 100).toFixed(0)}%</span>
                    </div>
                  </td>
                  <td>
                    {item.age_prediction ? (
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span style={{ fontSize: '18px' }}>{item.age_prediction.icon}</span>
                        <div>
                          <div style={{ fontSize: '12px', fontWeight: 600, color: item.age_prediction.color }}>
                            {item.age_prediction.label}
                          </div>
                          <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                            {item.age_prediction.range}
                          </div>
                        </div>
                      </div>
                    ) : (
                      <span style={{ color: 'var(--text-muted)', fontSize: '12px' }}>N/A</span>
                    )}
                  </td>
                  <td style={{ color: 'var(--text-muted)', fontSize: '12px' }}>
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
