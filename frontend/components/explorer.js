import React, { useState, useEffect } from 'react';
import axios from 'axios';

const Explorer = () => {
  const [activeTab, setActiveTab] = useState('blocks');
  const [blocks, setBlocks] = useState([]);
  const [transactions, setTransactions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError('');
        
        // In a real implementation, fetch data from backend API
        // For this example, we'll use mock data
        
        // Mock blocks data
        const mockBlocks = Array.from({ length: 10 }, (_, i) => ({
          index: 10 - i,
          hash: `0x${Math.random().toString(16).substr(2, 40)}`,
          timestamp: new Date(Date.now() - i * 600000).toISOString(),
          transactions: Math.floor(Math.random() * 10) + 1,
          validator: `0x${Math.random().toString(16).substr(2, 40)}`
        }));
        
        // Mock transactions data
        const mockTransactions = Array.from({ length: 20 }, (_, i) => ({
          id: `0x${Math.random().toString(16).substr(2, 40)}`,
          sender: `0x${Math.random().toString(16).substr(2, 40)}`,
          recipient: `0x${Math.random().toString(16).substr(2, 40)}`,
          amount: parseFloat((Math.random() * 1000).toFixed(2)),
          timestamp: new Date(Date.now() - i * 300000).toISOString(),
          blockIndex: Math.floor(Math.random() * 10) + 1
        }));
        
        setBlocks(mockBlocks);
        setTransactions(mockTransactions);
        setLoading(false);
      } catch (err) {
        setError('Error fetching blockchain data: ' + err.message);
        setLoading(false);
      }
    };
    
    fetchData();
  }, []);
  
  const handleSearch = async (e) => {
    e.preventDefault();
    
    if (!searchQuery.trim()) {
      return;
    }
    
    try {
      setLoading(true);
      setSearchResults(null);
      
      // In a real implementation, perform API search
      // For this example, we'll simulate searching
      
      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      if (searchQuery.startsWith('0x')) {
        // Search for transaction or block hash
        const foundTransaction = transactions.find(tx => tx.id.toLowerCase() === searchQuery.toLowerCase());
        const foundBlock = blocks.find(block => block.hash.toLowerCase() === searchQuery.toLowerCase());
        
        if (foundTransaction) {
          setSearchResults({ type: 'transaction', data: foundTransaction });
        } else if (foundBlock) {
          setSearchResults({ type: 'block', data: foundBlock });
        } else {
          setSearchResults({ type: 'notFound' });
        }
      } else if (!isNaN(parseInt(searchQuery))) {
        // Search for block index
        const blockIndex = parseInt(searchQuery);
        const foundBlock = blocks.find(block => block.index === blockIndex);
        
        if (foundBlock) {
          setSearchResults({ type: 'block', data: foundBlock });
        } else {
          setSearchResults({ type: 'notFound' });
        }
      } else {
        setSearchResults({ type: 'notFound' });
      }
      
      setLoading(false);
    } catch (err) {
      setError('Search failed: ' + err.message);
      setLoading(false);
    }
  };
  
  const formatTime = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleString();
  };
  
  const renderBlocks = () => {
    return (
      <div className="blocks-table">
        <table>
          <thead>
            <tr>
              <th>Block</th>
              <th>Hash</th>
              <th>Time</th>
              <th>Transactions</th>
              <th>Validator</th>
            </tr>
          </thead>
          <tbody>
            {blocks.map(block => (
              <tr key={block.hash}>
                <td>{block.index}</td>
                <td>
                  <a href={`#/block/${block.hash}`}>{`${block.hash.substr(0, 10)}...${block.hash.substr(-8)}`}</a>
                </td>
                <td>{formatTime(block.timestamp)}</td>
                <td>{block.transactions}</td>
                <td>{`${block.validator.substr(0, 10)}...${block.validator.substr(-8)}`}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };
  
  const renderTransactions = () => {
    return (
      <div className="transactions-table">
        <table>
          <thead>
            <tr>
              <th>Transaction</th>
              <th>Block</th>
              <th>Time</th>
              <th>From</th>
              <th>To</th>
              <th>Amount</th>
            </tr>
          </thead>
          <tbody>
            {transactions.map(tx => (
              <tr key={tx.id}>
                <td>
                  <a href={`#/tx/${tx.id}`}>{`${tx.id.substr(0, 10)}...${tx.id.substr(-8)}`}</a>
                </td>
                <td>{tx.blockIndex}</td>
                <td>{formatTime(tx.timestamp)}</td>
                <td>{`${tx.sender.substr(0, 8)}...${tx.sender.substr(-6)}`}</td>
                <td>{`${tx.recipient.substr(0, 8)}...${tx.recipient.substr(-6)}`}</td>
                <td>{tx.amount.toLocaleString()} NYX</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };
  
  const renderSearchResults = () => {
    if (!searchResults) return null;
    
    switch (searchResults.type) {
      case 'block':
        return (
          <div className="search-results">
            <h3>Block Details</h3>
            <table>
              <tbody>
                <tr>
                  <td>Block Number:</td>
                  <td>{searchResults.data.index}</td>
                </tr>
                <tr>
                  <td>Hash:</td>
                  <td>{searchResults.data.hash}</td>
                </tr>
                <tr>
                  <td>Timestamp:</td>
                  <td>{formatTime(searchResults.data.timestamp)}</td>
                </tr>
                <tr>
                  <td>Transactions:</td>
                  <td>{searchResults.data.transactions}</td>
                </tr>
                <tr>
                  <td>Validator:</td>
                  <td>{searchResults.data.validator}</td>
                </tr>
              </tbody>
            </table>
          </div>
        );
      
      case 'transaction':
        return (
          <div className="search-results">
            <h3>Transaction Details</h3>
            <table>
              <tbody>
                <tr>
                  <td>Transaction Hash:</td>
                  <td>{searchResults.data.id}</td>
                </tr>
                <tr>
                  <td>Block:</td>
                  <td>{searchResults.data.blockIndex}</td>
                </tr>
                <tr>
                  <td>Timestamp:</td>
                  <td>{formatTime(searchResults.data.timestamp)}</td>
                </tr>
                <tr>
                  <td>From:</td>
                  <td>{searchResults.data.sender}</td>
                </tr>
                <tr>
                  <td>To:</td>
                  <td>{searchResults.data.recipient}</td>
                </tr>
                <tr>
                  <td>Amount:</td>
                  <td>{searchResults.data.amount.toLocaleString()} NYX</td>
                </tr>
              </tbody>
            </table>
          </div>
        );
      
      case 'notFound':
        return (
          <div className="search-results">
            <p>No results found for "{searchQuery}"</p>
          </div>
        );
      
      default:
        return null;
    }
  };

  return (
    <div className="explorer-container">
      <div className="explorer-header">
        <h1>Blockchain Explorer</h1>
        <p>Explore the NyxSynth blockchain, view blocks, and track transactions</p>
      </div>
      
      <form className="search-form" onSubmit={handleSearch}>
        <input
          type="text"
          placeholder="Search by block number, block hash, or transaction hash"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />
        <button type="submit">Search</button>
      </form>
      
      {error && <div className="error-message">{error}</div>}
      
      {searchResults && renderSearchResults()}
      
      {!searchResults && (
        <>
          <div className="explorer-tabs">
            <button
              className={activeTab === 'blocks' ? 'active' : ''}
              onClick={() => setActiveTab('blocks')}
            >
              Latest Blocks
            </button>
            <button
              className={activeTab === 'transactions' ? 'active' : ''}
              onClick={() => setActiveTab('transactions')}
            >
              Latest Transactions
            </button>
          </div>
          
          {loading ? (
            <div className="loading">Loading blockchain data...</div>
          ) : (
            activeTab === 'blocks' ? renderBlocks() : renderTransactions()
          )}
        </>
      )}
    </div>
  );
};

export default Explorer;
