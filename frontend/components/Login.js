import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const Login = ({ setWallet, setIsLoggedIn }) => {
  const [isCreating, setIsCreating] = useState(true);
  const [privateKey, setPrivateKey] = useState('');
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleCreateWallet = async () => {
    try {
      const response = await axios.post('/api/wallet/create');
      if (response.data.success) {
        const walletData = response.data.wallet;
        localStorage.setItem('nyxWallet', JSON.stringify(walletData));
        setWallet(walletData);
        setIsLoggedIn(true);
        navigate('/wallet');
      } else {
        setError('Failed to create wallet');
      }
    } catch (err) {
      setError('Error creating wallet: ' + err.message);
    }
  };

  const handleImportWallet = async () => {
    if (!privateKey) {
      setError('Please enter your private key');
      return;
    }

    try {
      const response = await axios.post('/api/wallet/import', { privateKey });
      if (response.data.success) {
        const walletData = response.data.wallet;
        localStorage.setItem('nyxWallet', JSON.stringify(walletData));
        setWallet(walletData);
        setIsLoggedIn(true);
        navigate('/wallet');
      } else {
        setError('Failed to import wallet');
      }
    } catch (err) {
      setError('Error importing wallet: ' + err.message);
    }
  };

  return (
    <div className="login-container">
      <h1>Access Your NyxSynth Wallet</h1>
      
      <div className="login-options">
        <div className="option-tabs">
          <button 
            className={isCreating ? 'active' : ''} 
            onClick={() => setIsCreating(true)}
          >
            Create New Wallet
          </button>
          <button 
            className={!isCreating ? 'active' : ''} 
            onClick={() => setIsCreating(false)}
          >
            Import Existing Wallet
          </button>
        </div>
        
        <div className="option-content">
          {isCreating ? (
            <div className="create-wallet">
              <p>Create a new NyxSynth wallet to start your journey into bioluminescent blockchain technology.</p>
              <button className="primary-button" onClick={handleCreateWallet}>
                Create Wallet
              </button>
            </div>
          ) : (
            <div className="import-wallet">
              <p>Import your existing wallet using your private key.</p>
              <input 
                type="password" 
                placeholder="Enter your private key"
                value={privateKey}
                onChange={(e) => setPrivateKey(e.target.value)}
              />
              <button className="primary-button" onClick={handleImportWallet}>
                Import Wallet
              </button>
            </div>
          )}
          
          {error && <div className="error-message">{error}</div>}
        </div>
      </div>
      
      <div className="security-note">
        <h3>Security Note</h3>
        <p>Never share your private key with anyone. NyxSynth will never ask for your private key except when importing a wallet through this secure form.</p>
      </div>
    </div>
  );
};

export default Login;
