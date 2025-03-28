import React, { useState, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter as Router, Routes, Route, Link, Navigate } from 'react-router-dom';
import './styles.css';

// Import components
import Home from './components/Home';
import Login from './components/Login';
import Wallet from './components/Wallet';
import Staking from './components/Staking';
import Explorer from './components/Explorer';

// Logo component
const Logo = () => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40" width="40" height="40">
    <circle cx="20" cy="20" r="20" fill="#121212" />
    <path d="M20 2L2 12l18 10 18-10L20 2z" fill="#6200EA" />
    <path d="M10 18.5L2 22l18 10 18-10-8-3.5" fill="#00E5FF" opacity="0.8" />
    <path d="M10 28.5L2 32l18 10 18-10-8-3.5" fill="#9D46FF" opacity="0.6" />
    <circle cx="20" cy="20" r="5" fill="#00E5FF" opacity="0.8" />
  </svg>
);

// Main App Component
const App = () => {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [wallet, setWallet] = useState(null);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  useEffect(() => {
    // Check if user has a stored wallet
    const storedWallet = localStorage.getItem('nyxWallet');
    if (storedWallet) {
      setWallet(JSON.parse(storedWallet));
      setIsLoggedIn(true);
    }
  }, []);

  const handleLogout = () => {
    setIsLoggedIn(false);
    setWallet(null);
    localStorage.removeItem('nyxWallet');
    setIsMobileMenuOpen(false);
  };

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
  };

  return (
    <Router>
      <div className="nyxsynth-app">
        <header className="app-header">
          <div className="logo">
            <Logo />
            <h1>NyxSynth</h1>
          </div>
          
          <button className="mobile-menu-btn" onClick={toggleMobileMenu}>
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="3" y1="12" x2="21" y2="12"></line>
              <line x1="3" y1="6" x2="21" y2="6"></line>
              <line x1="3" y1="18" x2="21" y2="18"></line>
            </svg>
          </button>
          
          <nav className={`main-nav ${isMobileMenuOpen ? 'mobile-open' : ''}`}>
            <Link to="/" onClick={() => setIsMobileMenuOpen(false)}>Home</Link>
            {isLoggedIn && (
              <Link to="/wallet" onClick={() => setIsMobileMenuOpen(false)}>Wallet</Link>
            )}
            {isLoggedIn && (
              <Link to="/staking" onClick={() => setIsMobileMenuOpen(false)}>Neural Staking</Link>
            )}
            <Link to="/explorer" onClick={() => setIsMobileMenuOpen(false)}>Explorer</Link>
            
            {isLoggedIn ? (
              <button className="login-button" onClick={handleLogout}>Logout</button>
            ) : (
              <Link to="/login" className="login-button" onClick={() => setIsMobileMenuOpen(false)}>Login</Link>
            )}
          </nav>
        </header>

        <main className="app-content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/login" element={
              isLoggedIn 
                ? <Navigate to="/wallet" /> 
                : <Login setWallet={setWallet} setIsLoggedIn={setIsLoggedIn} />
            } />
            <Route path="/wallet" element={
              isLoggedIn 
                ? <Wallet wallet={wallet} /> 
                : <Navigate to="/login" />
            } />
            <Route path="/staking" element={
              isLoggedIn 
                ? <Staking wallet={wallet} /> 
                : <Navigate to="/login" />
            } />
            <Route path="/explorer" element={<Explorer />} />
          </Routes>
        </main>

        <footer className="app-footer">
          <p>NyxSynth © 2025 - The Biomimetic Neural Cryptocurrency</p>
          <div className="footer-links">
            <a href="/docs">Documentation</a>
            <a href="/about">About</a>
            <a href="https://github.com/nyxsynth/nyxsynth" target="_blank" rel="noopener noreferrer">GitHub</a>
          </div>
        </footer>
      </div>
    </Router>
  );
};

// Home Component
const Home = () => {
  return (
    <div className="home-container">
      <div className="hero-section">
        <h1>Welcome to NyxSynth</h1>
        <p>The world's first biomimetic neural cryptocurrency inspired by deep-sea bioluminescent creatures.</p>
        <div className="cta-buttons">
          <Link to="/wallet" className="primary-button">Get Started</Link>
          <a href="/docs" className="secondary-button">Learn More</a>
        </div>
      </div>

      <div className="features-section">
        <h2>Key Innovations</h2>
        
        <div className="feature-cards">
          <div className="feature-card">
            <h3>Bioluminescent Coordination Protocol</h3>
            <p>Consensus through synchronized illumination patterns, inspired by deep-sea creatures.</p>
          </div>
          
          <div className="feature-card">
            <h3>Neural Validation Networks</h3>
            <p>Self-optimizing blockchain with adaptive neural networks that evolve over time.</p>
          </div>
          
          <div className="feature-card">
            <h3>Symbiotic Smart Contracts</h3>
            <p>Contracts that form mutually beneficial relationships, sharing resources and capabilities.</p>
          </div>
          
          <div className="feature-card">
            <h3>Abyssal Scalability</h3>
            <p>Dynamic scaling to transaction volumes without compromising security or speed.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

// Mount the app
const container = document.getElementById('root');
const root = createRoot(container);
root.render(<App />);
