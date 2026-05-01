import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom';
import { Moon, Sun, Activity, Home, Search, Upload, PieChart, Clock, Info } from 'lucide-react';
import HomePage from './pages/Home';
import Predictor from './pages/Predictor';
import BatchUpload from './pages/BatchUpload';
import Dashboard from './pages/Dashboard';
import History from './pages/History';
import About from './pages/About';
import './index.css';

function App() {
  const [theme, setTheme] = useState('dark');

  useEffect(() => {
    document.body.setAttribute('data-theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => prev === 'dark' ? 'light' : 'dark');
  };

  return (
    <Router>
      <div className="app-container">
        <div className="blob blob-1"></div>
        <div className="blob blob-2"></div>
        
        {/* Sidebar */}
        <aside className="sidebar">
          <NavLink to="/" className="logo">
            <Activity color="var(--primary)" size={28} />
            <span>GenoVision</span>
          </NavLink>
          
          <nav className="nav-links">
            <NavLink to="/" className={({isActive}) => `nav-link ${isActive ? 'active' : ''}`}>
              <Home size={20} />
              <span>Home</span>
            </NavLink>
            <NavLink to="/predictor" className={({isActive}) => `nav-link ${isActive ? 'active' : ''}`}>
              <Search size={20} />
              <span>Predictor</span>
            </NavLink>
            <NavLink to="/batch" className={({isActive}) => `nav-link ${isActive ? 'active' : ''}`}>
              <Upload size={20} />
              <span>Batch Upload</span>
            </NavLink>
            <NavLink to="/dashboard" className={({isActive}) => `nav-link ${isActive ? 'active' : ''}`}>
              <PieChart size={20} />
              <span>Dashboard</span>
            </NavLink>
            <NavLink to="/history" className={({isActive}) => `nav-link ${isActive ? 'active' : ''}`}>
              <Clock size={20} />
              <span>History</span>
            </NavLink>
            <NavLink to="/about" className={({isActive}) => `nav-link ${isActive ? 'active' : ''}`}>
              <Info size={20} />
              <span>About</span>
            </NavLink>
          </nav>
        </aside>

        {/* Main Content */}
        <main className="main-content">
          <header className="page-header">
            <h2>Genomic Mutation Impact Predictor</h2>
            <button className="theme-toggle" onClick={toggleTheme}>
              {theme === 'dark' ? <Sun size={20} /> : <Moon size={20} />}
            </button>
          </header>
          
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/predictor" element={<Predictor />} />
            <Route path="/batch" element={<BatchUpload />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/history" element={<History />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
