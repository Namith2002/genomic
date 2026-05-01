import React from 'react';

export default function About() {
  return (
    <div className="glass-card" style={{ maxWidth: '800px', margin: '0 auto' }}>
      <h2 style={{ marginBottom: '24px' }}>About GenoVision</h2>
      
      <div style={{ lineHeight: '1.6', color: 'var(--text-muted)' }}>
        <p style={{ marginBottom: '16px' }}>
          GenoVision is a cutting-edge full-stack platform designed to predict the impact of genomic mutations. By leveraging advanced Machine Learning and Computer Vision technologies, it helps determine whether a genetic variant is likely to be <strong>Benign</strong> or <strong>Pathogenic</strong>.
        </p>
        
        <h3 style={{ color: 'var(--text-main)', marginTop: '24px', marginBottom: '12px' }}>Features</h3>
        <ul style={{ paddingLeft: '24px', marginBottom: '16px' }}>
          <li style={{ marginBottom: '8px' }}><strong>Manual Prediction:</strong> Input reference and alternate alleles directly to get instant predictions powered by a Random Forest model trained on ClinVar data.</li>
          <li style={{ marginBottom: '8px' }}><strong>Image Scan:</strong> Upload synthetic or actual genomic visualizations to identify mutations using a Convolutional Neural Network (CNN).</li>
          <li style={{ marginBottom: '8px' }}><strong>Batch Processing:</strong> Upload CSV files containing multiple variants for high-throughput analysis.</li>
          <li style={{ marginBottom: '8px' }}><strong>Dashboard Analytics:</strong> Visualize prediction distributions and model accuracy.</li>
        </ul>

        <h3 style={{ color: 'var(--text-main)', marginTop: '24px', marginBottom: '12px' }}>Technology Stack</h3>
        <div className="grid-2">
          <div>
            <h4 style={{ color: 'var(--primary)' }}>Frontend</h4>
            <ul>
              <li>React + Vite</li>
              <li>Chart.js</li>
              <li>Vanilla CSS (Glassmorphism)</li>
            </ul>
          </div>
          <div>
            <h4 style={{ color: 'var(--accent)' }}>Backend</h4>
            <ul>
              <li>FastAPI</li>
              <li>TensorFlow / Keras (CNN)</li>
              <li>Scikit-Learn (Random Forest)</li>
              <li>SQLite</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
