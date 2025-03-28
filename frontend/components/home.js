import React from 'react';
import { Link } from 'react-router-dom';

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
            <p>Consensus through synchronized illumination patterns, inspired by deep-sea creatures. This revolutionary approach achieves consensus through coordinated light emissions rather than computational waste.</p>
          </div>
          
          <div className="feature-card">
            <h3>Neural Validation Networks</h3>
            <p>Self-optimizing blockchain with adaptive neural networks that evolve over time. Transactions are validated through artificial neural networks that adapt to network conditions, creating a truly living digital ecosystem.</p>
          </div>
          
          <div className="feature-card">
            <h3>Symbiotic Smart Contracts</h3>
            <p>Contracts that form mutually beneficial relationships, sharing resources and capabilities. Unlike traditional smart contracts that operate in isolation, NyxSynth's contracts create emergent functionalities beyond their individual programming.</p>
          </div>
          
          <div className="feature-card">
            <h3>Abyssal Scalability</h3>
            <p>Dynamic scaling to transaction volumes without compromising security or speed. Like deep-sea organisms that adapt to extreme pressures, NyxSynth scales efficiently to meet demand through its adaptive neural architecture.</p>
          </div>
        </div>
      </div>
      
      <div className="technology-section">
        <h2>Evolutionary Blockchain Technology</h2>
        <div className="tech-content">
          <div className="tech-col">
            <h3>Dark Mode Processing</h3>
            <p>Taking inspiration from creatures that thrive in darkness, our blockchain operates with minimal energy consumption while maintaining peak performance.</p>
            
            <h3>Quantum-Resistant Cryptography</h3>
            <p>Just as bioluminescent organisms evolved sophisticated defense mechanisms, NyxSynth employs post-quantum cryptographic solutions to ensure long-term security.</p>
          </div>
          
          <div className="tech-col">
            <h3>NYX Token Economics</h3>
            <p>The NYX token system is designed to mirror the energy efficiency of deep-sea creatures, utilizing minimal resources while maximizing value propagation through the ecosystem.</p>
            
            <h3>Neural Staking Rewards</h3>
            <p>Earn rewards by contributing to the network's neural consensus mechanism. The more aligned your staking patterns are with the network, the higher your rewards.</p>
          </div>
        </div>
      </div>
      
      <div className="cta-section">
        <h2>Join the Evolutionary Blockchain Revolution</h2>
        <p>Experience a blockchain that doesn't just process transactions, but actually evolves, adapts, and coordinates like a living ecosystem.</p>
        <Link to="/login" className="primary-button">Create Your Wallet</Link>
      </div>
    </div>
  );
};

export default Home;
