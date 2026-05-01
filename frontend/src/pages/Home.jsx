import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Activity, ShieldCheck, Database, Zap } from 'lucide-react';

export default function Home() {
  const navigate = useNavigate();

  return (
    <div className="home-container">
      <div className="glass-card" style={{ textAlign: 'center', padding: '60px 20px', marginBottom: '40px' }}>
        <h1 style={{ fontSize: '48px', marginBottom: '20px', background: 'linear-gradient(to right, var(--primary), var(--accent))', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
          Welcome to GenoVision
        </h1>
        <p style={{ fontSize: '18px', color: 'var(--text-muted)', maxWidth: '600px', margin: '0 auto 40px' }}>
          An advanced AI-powered platform for predicting genomic mutation impacts.
          Determine whether a variant is Benign or Pathogenic with high confidence.
        </p>
        <button className="btn btn-primary" style={{ padding: '16px 32px', fontSize: '18px' }} onClick={() => navigate('/predictor')}>
          Start Prediction
        </button>
      </div>

      <div className="grid-3">
        <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center' }}>
          <ShieldCheck size={48} color="var(--accent)" style={{ marginBottom: '20px' }} />
          <h3>High Accuracy</h3>
          <p style={{ color: 'var(--text-muted)', marginTop: '12px' }}>
            Powered by advanced Machine Learning models trained on ClinVar datasets.
          </p>
        </div>
        <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center' }}>
          <Database size={48} color="var(--primary)" style={{ marginBottom: '20px' }} />
          <h3>Multiple Modes</h3>
          <p style={{ color: 'var(--text-muted)', marginTop: '12px' }}>
            Support for both manual allele-based prediction and image-based mutation detection.
          </p>
        </div>
        <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center' }}>
          <Zap size={48} color="#eab308" style={{ marginBottom: '20px' }} />
          <h3>Fast Processing</h3>
          <p style={{ color: 'var(--text-muted)', marginTop: '12px' }}>
            Batch processing capabilities for analyzing large sets of genomic variants rapidly.
          </p>
        </div>
      </div>
    </div>
  );
}
