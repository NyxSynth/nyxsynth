import React, { useState, useEffect } from 'react';
import axios from 'axios';

const Wallet = ({ wallet }) => {
  const [balance, setBalance] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');
  const [recipient, setRecipient] = useState('');
  const [amount, setAmount] = useState('');
  const [transferStatus, setTransferStatus] = useState(null);

  useEffect(() => {
    const fetchBalance = async () => {
      try {
        const response = await axios.get(`/api/wallet/balance/${wallet.address}`);
        if (response.data.success) {
          setBalance(response.data.balance);
        } else {
          setError('Failed to fetch balance');
        }
        setIsLoading(false);
      } catch (err) {
        setError('Error fetching balance: ' + err.message);
        setIsLoading(false);
      }
    };

    fetchBalance();
    // Poll for balance updates every 30 seconds
    const intervalId = setInterval(fetchBalance, 30000);
    
    return () => clearInterval(intervalId);
  }, [wallet.address]);

  const handleTransfer = async () => {
    if (!recipient || !amount) {
      setTransferStatus({ success: false, message: 'Please enter recipient address and amount' });
      return;
    }

    try {
      setTransferStatus({ loading: true });
      // In a real implementation, this would make an API call to transfer tokens
      // For this example, we'll simulate a transfer
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      setTransferStatus({ success: true, message: 'Transfer successful!' });
      
      // Reset form
      setRecipient('');
      setAmount('');
      
      // Refresh balance
      const response = await axios.get(`/api/wallet/balance/${wallet.address}`);
      if (response.data.success) {
        setBalance(response.data.balance);
      }
    } catch (err) {
      setTransferStatus({ success: false, message: 'Transfer failed: ' + err.message });
    }
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(wallet.address);
    alert('Address copied to clipboard!');
  };

  if (isLoading) {
    return <div className="loading">Loading wallet data...</div>;
  }

  return (
    <div className="wallet-container">
      <div className="wallet-header">
        <h1>Your NyxSynth Wallet</h1>
        <div className="wallet-balance">
          <span className="balance-label">Balance:</span>
          <span className="balance-amount">{balance.toLocaleString()} NYX</span>
        </div>
      </div>
      
      <div className="wallet-address">
        <h3>Your Wallet Address</h3>
        <div className="address-display">
          <code>{wallet.address}</code>
          <button className="copy-button" onClick={handleCopy}>
            Copy
          </button>
        </div>
      </div>
      
      <div className="send-tokens">
        <h3>Send NYX Tokens</h3>
        <div className="transfer-form">
          <div className="form-group">
            <label>Recipient Address</label>
            <input 
              type="text" 
              placeholder="Enter recipient wallet address"
              value={recipient}
              onChange={(e) => setRecipient(e.target.value)}
            />
          </div>
          
          <div className="form-group">
            <label>Amount (NYX)</label>
            <input 
              type="number" 
              placeholder="Enter amount to send"
              value={amount}
              onChange={(e) => setAmount(e.target.value)}
            />
          </div>
          
          <button 
            className="primary-button"
            onClick={handleTransfer}
            disabled={transferStatus?.loading}
          >
            {transferStatus?.loading ? 'Processing...' : 'Send Tokens'}
          </button>
          
          {transferStatus && !transferStatus.loading && (
            <div className={`transfer-status ${transferStatus.success ? 'success' : 'error'}`}>
              {transferStatus.message}
            </div>
          )}
        </div>
      </div>
      
      <div className="transaction-history">
        <h3>Transaction History</h3>
        <p className="no-transactions">No recent transactions to display.</p>
      </div>
    </div>
  );
};

export default Wallet;
