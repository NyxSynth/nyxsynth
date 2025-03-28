import React, { useState, useEffect } from 'react';
import axios from 'axios';

const Staking = ({ wallet }) => {
  const [pools, setPools] = useState([
    {
      id: 'pool1',
      name: 'Standard Neural Pool',
      apr: 10.0,
      minStake: 100,
      totalStaked: 1450000,
      myStake: 0
    },
    {
      id: 'pool2',
      name: 'Advanced Neural Pool',
      apr: 15.0,
      minStake: 1000,
      totalStaked: 980000,
      myStake: 0
    },
    {
      id: 'pool3',
      name: 'Expert Neural Pool',
      apr: 20.0,
      minStake: 10000,
      totalStaked: 420000,
      myStake: 0
    }
  ]);
  const [balance, setBalance] = useState(0);
  const [selectedPool, setSelectedPool] = useState(null);
  const [stakeAmount, setStakeAmount] = useState('');
  const [unstakeAmount, setUnstakeAmount] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [successMessage, setSuccessMessage] = useState('');
  
  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch wallet balance
        const balanceResponse = await axios.get(`/api/wallet/balance/${wallet.address}`);
        if (balanceResponse.data.success) {
          setBalance(balanceResponse.data.balance);
        }
        
        // In a real implementation, fetch staking pool data
        // For now, we'll use the mock data initialized above
        
        setLoading(false);
      } catch (err) {
        setError('Error loading data: ' + err.message);
        setLoading(false);
      }
    };
    
    fetchData();
  }, [wallet.address]);
  
  const handleStake = async () => {
    if (!selectedPool) {
      setError('Please select a staking pool');
      return;
    }
    
    if (!stakeAmount || parseFloat(stakeAmount) <= 0) {
      setError('Please enter a valid stake amount');
      return;
    }

    const amount = parseFloat(stakeAmount);
    
    if (amount < selectedPool.minStake) {
      setError(`Minimum stake amount is ${selectedPool.minStake} NYX`);
      return;
    }
    
    if (amount > balance) {
      setError('Insufficient balance');
      return;
    }
    
    try {
      setLoading(true);
      setError('');
      
      // In a real implementation, make an API call to stake tokens
      // For this example, we'll simulate staking
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      // Update UI
      setPools(pools.map(pool => {
        if (pool.id === selectedPool.id) {
          return {
            ...pool,
            myStake: pool.myStake + amount,
            totalStaked: pool.totalStaked + amount
          };
        }
        return pool;
      }));
      
      setBalance(prevBalance => prevBalance - amount);
      setStakeAmount('');
      setSuccessMessage(`Successfully staked ${amount} NYX in ${selectedPool.name}`);
      setLoading(false);
      
      // Clear success message after 5 seconds
      setTimeout(() => setSuccessMessage(''), 5000);
      
    } catch (err) {
      setError('Staking failed: ' + err.message);
      setLoading(false);
    }
  };
  
  const handleUnstake = async () => {
    if (!selectedPool) {
      setError('Please select a staking pool');
      return;
    }
    
    if (!unstakeAmount || parseFloat(unstakeAmount) <= 0) {
      setError('Please enter a valid unstake amount');
      return;
    }
    
    const amount = parseFloat(unstakeAmount);
    const pool = pools.find(p => p.id === selectedPool.id);
    
    if (amount > pool.myStake) {
      setError('Cannot unstake more than your staked amount');
      return;
    }
    
    try {
      setLoading(true);
      setError('');
      
      // In a real implementation, make an API call to unstake tokens
      // For this example, we'll simulate unstaking
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      // Update UI
      setPools(pools.map(pool => {
        if (pool.id === selectedPool.id) {
          return {
            ...pool,
            myStake: pool.myStake - amount,
            totalStaked: pool.totalStaked - amount
          };
        }
        return pool;
      }));
      
      setBalance(prevBalance => prevBalance + amount);
      setUnstakeAmount('');
      setSuccessMessage(`Successfully unstaked ${amount} NYX from ${selectedPool.name}`);
      setLoading(false);
      
      // Clear success message after 5 seconds
      setTimeout(() => setSuccessMessage(''), 5000);
      
    } catch (err) {
      setError('Unstaking failed: ' + err.message);
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="loading">Loading staking data...</div>;
  }

  return (
    <div className="staking-container">
      <div className="staking-header">
        <h1>Neural Staking Pools</h1>
        <p className="staking-intro">
          Stake your NYX tokens in neural pools to earn rewards while supporting
          the network's bioluminescent consensus mechanism.
        </p>
        
        <div className="wallet-balance">
          <span className="balance-label">Available Balance:</span>
          <span className="balance-amount">{balance.toLocaleString()} NYX</span>
        </div>
      </div>
      
      {error && <div className="error-message">{error}</div>}
      {successMessage && <div className="success-message">{successMessage}</div>}
      
      <div className="staking-pools">
        {pools.map(pool => (
          <div 
            key={pool.id} 
            className={`pool-card ${selectedPool?.id === pool.id ? 'selected' : ''}`}
            onClick={() => setSelectedPool(pool)}
          >
            <h3>{pool.name}</h3>
            <div className="pool-stats">
              <div className="stat">
                <span className="stat-label">APR:</span>
                <span className="stat-value">{pool.apr}%</span>
              </div>
              <div className="stat">
                <span className="stat-label">Min. Stake:</span>
                <span className="stat-value">{pool.minStake.toLocaleString()} NYX</span>
              </div>
              <div className="stat">
                <span className="stat-label">Total Staked:</span>
                <span className="stat-value">{pool.totalStaked.toLocaleString()} NYX</span>
              </div>
              <div className="stat">
                <span className="stat-label">Your Stake:</span>
                <span className="stat-value">{pool.myStake.toLocaleString()} NYX</span>
              </div>
            </div>
          </div>
        ))}
      </div>
      
      {selectedPool && (
        <div className="staking-actions">
          <h2>Manage Your Stake in {selectedPool.name}</h2>
          
          <div className="stake-container">
            <div className="stake-form">
              <h3>Stake Tokens</h3>
              <div className="form-group">
                <label>Amount to Stake (NYX)</label>
                <input
                  type="number"
                  placeholder={`Minimum ${selectedPool.minStake} NYX`}
                  value={stakeAmount}
                  onChange={(e) => setStakeAmount(e.target.value)}
                />
              </div>
              <button 
                className="primary-button"
                onClick={handleStake}
                disabled={loading}
              >
                Stake Tokens
              </button>
            </div>
            
            <div className="unstake-form">
              <h3>Unstake Tokens</h3>
              <div className="form-group">
                <label>Amount to Unstake (NYX)</label>
                <input
                  type="number"
                  placeholder={`Maximum ${selectedPool.myStake} NYX`}
                  value={unstakeAmount}
                  onChange={(e) => setUnstakeAmount(e.target.value)}
                />
              </div>
              <button 
                className="primary-button"
                onClick={handleUnstake}
                disabled={loading}
              >
                Unstake Tokens
              </button>
            </div>
          </div>
          
          <div className="staking-info">
            <h3>Pool Information</h3>
            <p>
              The {selectedPool.name} uses neural network patterns to validate transactions
              and contribute to the network's bioluminescent consensus mechanism.
            </p>
            <p>
              Rewards are distributed daily based on your staking proportion and the pool's
              neural efficiency rating.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default Staking;
