import numpy as np
import tensorflow as tf
from typing import List, Dict, Any
import time
import hashlib
import json

class NeuralValidator:
    """
    Implements the Neural Validation Network for NyxSynth.
    This system uses neural networks to validate transactions and generate block patterns.
    """
    
    def __init__(self):
        # Initialize the neural network
        self.model = self._build_model()
        self.transaction_history = []
        self.pattern_dimension = 64
        self.adaptation_factor = 0.1
    
    def _build_model(self):
        """Build the neural validation model."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='tanh')  # Pattern output
        ])
        
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def generate_genesis_pattern(self) -> List[float]:
        """
        Generate the initial pattern for the genesis block.
        
        Returns:
            Initial neural pattern
        """
        # Generate a stable, repeatable pattern for genesis
        seed_value = 42
        np.random.seed(seed_value)
        genesis_pattern = np.random.rand(self.pattern_dimension) * 2 - 1
        return genesis_pattern.tolist()
    
    def generate_pattern(self, transactions: List) -> List[float]:
        """
        Generate a neural pattern for a set of transactions.
        
        Args:
            transactions: List of transactions to analyze
            
        Returns:
            Neural pattern representing the transactions
        """
        if not transactions:
            # Return a stable pattern for empty transaction sets
            return np.zeros(self.pattern_dimension).tolist()
        
        # Extract features from transactions
        features = self._extract_transaction_features(transactions)
        
        # Generate pattern using neural network
        pattern = self.model.predict(np.array([features]))[0]
        
        return pattern.tolist()
    
    def _extract_transaction_features(self, transactions) -> np.ndarray:
        """
        Extract relevant features from a set of transactions.
        
        Args:
            transactions: List of transactions
            
        Returns:
            Feature vector
        """
        # This is a simplified feature extraction
        # In a real implementation, this would be more sophisticated
        
        total_value = sum(tx.amount for tx in transactions)
        num_transactions = len(transactions)
        avg_value = total_value / max(num_transactions, 1)
        
        unique_senders = set(tx.sender for tx in transactions)
        unique_recipients = set(tx.recipient for tx in transactions)
        
        timestamp_diffs = []
        for i in range(1, len(transactions)):
            timestamp_diffs.append(transactions[i].timestamp - transactions[i-1].timestamp)
        
        avg_timestamp_diff = sum(timestamp_diffs) / max(len(timestamp_diffs), 1)
        
        # Create a 10-dimensional feature vector
        features = np.array([
            total_value,
            num_transactions,
            avg_value,
            len(unique_senders),
            len(unique_recipients),
            avg_timestamp_diff,
            time.time() % 86400 / 86400,  # Time of day (normalized)
            len(unique_senders & unique_recipients),  # Intersection
            len(self.transaction_history) % 1000,  # Recent history size
            np.random.rand()  # Small random element for uniqueness
        ])
        
        # Normalize features to [-1, 1] range
        features = (features - np.mean(features)) / (np.std(features) + 1e-8)
        
        return features
    
    def adapt_pattern(self, base_pattern: List[float], block_hash: str, nonce: int) -> List[float]:
        """
        Adapt a neural pattern based on mining attempts.
        
        Args:
            base_pattern: Starting pattern
            block_hash: Current block hash
            nonce: Current nonce value
            
        Returns:
            Adapted neural pattern
        """
        pattern_array = np.array(base_pattern)
        
        # Convert hash to a numerical influence
        hash_bytes = bytes.fromhex(block_hash)
        hash_values = np.array([b / 255.0 for b in hash_bytes])
        hash_influence = np.tile(hash_values, self.pattern_dimension // len(hash_values) + 1)[:self.pattern_dimension]
        
        # Use nonce to determine adaptation strength
        adaptation_strength = 0.1 * (1 + np.sin(nonce / 1000))
        
        # Adapt the pattern
        adapted_pattern = pattern_array + adaptation_strength * hash_influence
        
        # Normalize to keep within [-1, 1] range
        adapted_pattern = np.clip(adapted_pattern, -1, 1)
        
        return adapted_pattern.tolist()
    
    def validate_transaction(self, transaction, blockchain) -> bool:
        """
        Validate a transaction using the neural network.
        
        Args:
            transaction: Transaction to validate
            blockchain: Current blockchain state
            
        Returns:
            True if transaction is valid
        """
        # Basic validation
        if not transaction.verify():
            return False
        
        # Check if sender has sufficient balance
        sender_balance = blockchain.get_balance(transaction.sender)
        if transaction.sender != "SYSTEM" and sender_balance < transaction.amount:
            return False
        
        # Add transaction to history for learning
        self.transaction_history.append(transaction)
        if len(self.transaction_history) > 10000:
            self.transaction_history = self.transaction_history[-10000:]
        
        return True
    
    def adapt(self, new_block) -> None:
        """
        Adapt the neural network based on a new block.
        
        Args:
            new_block: New block to learn from
        """
        # This would train the neural network on new data
        # In a full implementation, this would involve gathering
        # training data and updating the model weights
        
        # For this example, we'll simply log that adaptation occurred
        print(f"Neural validator adapting to block {new_block.index}")
        
        # Extract features from the block's transactions
        if new_block.transactions:
            features = self._extract_transaction_features(new_block.transactions)
            
            # Create a simple training example:
            # - Input: Transaction features
            # - Output: The neural pattern that successfully validated the block
            X = np.array([features])
            y = np.array([new_block.neural_pattern])
            
            # Update the model with a single training step
            self.model.train_on_batch(X, y)
    
    def detect_anomalies(self, transactions) -> List[int]:
        """
        Detect potentially fraudulent or anomalous transactions.
        
        Args:
            transactions: List of transactions to check
            
        Returns:
            Indices of suspicious transactions
        """
        suspicious_indices = []
        
        # Extract features for each transaction individually
        for i, tx in enumerate(transactions):
            # Simple anomaly detection based on transaction patterns
            # In a real implementation, this would be more sophisticated
            
            # Check for unusual amounts
            if tx.amount > 1000000:  # Very large transaction
                suspicious_indices.append(i)
            
            # Check for unusual timing patterns
            if i > 0:
                time_diff = tx.timestamp - transactions[i-1].timestamp
                if time_diff < 0.01:  # Suspiciously close timestamps
                    suspicious_indices.append(i)
            
            # Check for repeated transactions
            for prev_tx in self.transaction_history[-100:]:
                if (tx.sender == prev_tx.sender and 
                    tx.recipient == prev_tx.recipient and
                    abs(tx.amount - prev_tx.amount) < 0.001):
                    suspicious_indices.append(i)
                    break
        
        return suspicious_indices
    
    def visualize_pattern(self, pattern: List[float]) -> Dict[str, Any]:
        """
        Generate visualization data for a neural pattern.
        
        Args:
            pattern: Neural pattern to visualize
            
        Returns:
            Visualization data
        """
        # Reshape pattern into an 8x8 grid for visualization
        grid = np.array(pattern).reshape(8, 8)
        
        # Generate color mapping (blue shades)
        colors = []
        for row in grid:
            row_colors = []
            for val in row:
                # Map values from [-1, 1] to [0, 255] for blue intensity
                blue_intensity = int((val + 1) * 127.5)
                row_colors.append(f"rgb(0, 0, {blue_intensity})")
            colors.append(row_colors)
        
        # Create visualization data
        return {
            "grid": grid.tolist(),
            "colors": colors,
            "intensity": np.mean(np.abs(grid)),
            "harmony": np.std(grid),  # Lower std = more harmonious pattern
            "dominant_frequencies": self._extract_dominant_frequencies(pattern)
        }
    
    def _extract_dominant_frequencies(self, pattern: List[float]) -> List[float]:
        """
        Extract dominant frequencies from a pattern using FFT.
        
        Args:
            pattern: Neural pattern
            
        Returns:
            List of dominant frequency components
        """
        # Perform Fast Fourier Transform
        fft_result = np.abs(np.fft.fft(pattern))
        
        # Find peaks (dominant frequencies)
        peaks = []
        threshold = np.mean(fft_result) + np.std(fft_result)
        
        for i in range(1, len(fft_result) - 1):
            if fft_result[i] > threshold:
                if fft_result[i] > fft_result[i-1] and fft_result[i] > fft_result[i+1]:
                    peaks.append((i, fft_result[i]))
        
        # Sort by amplitude and return top 5
        peaks.sort(key=lambda x: x[1], reverse=True)
        return [freq for freq, _ in peaks[:5]]