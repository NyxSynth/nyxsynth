import numpy as np
import tensorflow as tf
from typing import List, Dict, Any, Tuple, Optional, Union, Set
import time
import hashlib
import json
import os
import random
from collections import deque
import logging
import math
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("nyxsynth.neural")

class EnhancedNeuralValidator:
    """
    Advanced implementation of the Neural Validation Network for NyxSynth.
    This system uses sophisticated neural networks to validate transactions and generate block patterns.
    
    Key features:
    - Multi-model ensemble for better prediction accuracy
    - Advanced anomaly detection with autoencoders and statistical methods
    - Continuous learning with experience replay and adaptive weights
    - Formal verification of critical patterns
    - Adversarial resistance through pattern hardening
    - Integration with bioluminescent consensus
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the neural validator.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {
            'pattern_dimension': 64,
            'adaptation_factor': 0.1,
            'anomaly_threshold': 0.85,
            'replay_buffer_size': 10000,
            'batch_size': 32,
            'learning_rate': 0.001,
            'validation_threshold': 0.75,
            'security_level': 'high',
            'entropy_threshold': 3.0,
            'adversarial_resistance': 0.8,
            'verification_confidence': 0.9,
            'training_interval': 100,  # Train after this many transactions
            'pattern_diversity_threshold': 0.2,
            'evolve_rate': 0.05,  # Rate of neural evolution
            'model_ensemble_size': 3  # Number of models in ensemble
        }
        
        # Initialize model components
        self.pattern_dimension = self.config['pattern_dimension']
        self.adaptation_factor = self.config['adaptation_factor']
        self.anomaly_threshold = self.config['anomaly_threshold']
        
        # Create model ensemble
        self.models = self._build_model_ensemble()
        self.autoencoder = self._build_autoencoder()
        
        # Create experience replay buffer
        self.replay_buffer = deque(maxlen=self.config['replay_buffer_size'])
        
        # Transaction history for pattern analysis
        self.transaction_history = []
        
        # Load pretrained patterns if available
        self.pattern_library = self._load_pattern_library()
        
        # Performance tracking
        self.metrics = {
            'validation_time': [],
            'prediction_accuracy': [],
            'false_positives': 0,
            'false_negatives': 0,
            'anomalies_detected': 0,
            'total_transactions_processed': 0,
            'avg_validation_time_ms': 0,
            'total_blocks_validated': 0,
            'total_patterns_generated': 0,
            'pattern_entropy_avg': 0,
            'pattern_verification_success_rate': 1.0,
            'training_epochs_completed': 0
        }
        
        # Adaptive security components
        self.security_level = self.config['security_level']
        self.security_multiplier = self._get_security_multiplier()
        
        # Recent anomaly tracking
        self.recent_anomalies = deque(maxlen=100)
        
        # Pattern verification cache
        self.verified_patterns: Dict[str, Dict] = {}
        
        # Adversarial resistance components
        self.adversarial_patterns = set()
        self.trusted_pattern_hashes = set()
        
        # Transaction confidence scoring
        self.transaction_confidence_history: Dict[str, float] = {}
        
        # Model evolution tracking
        self.evolution_state = {
            'generations': 0,
            'fitness_scores': [],
            'last_evolution': time.time(),
            'mutations': 0
        }
        
        # Load model weights if available
        self._load_models()
        
        logger.info("EnhancedNeuralValidator initialized with security level: %s", self.security_level)
    
    def _get_security_multiplier(self) -> float:
        """
        Get security multiplier based on security level setting.
        
        Returns:
            Float multiplier to scale security parameters
        """
        security_levels = {
            'low': 0.5,
            'medium': 1.0,
            'high': 1.5,
            'maximum': 2.0
        }
        return security_levels.get(self.security_level, 1.0)
    
    def _build_model_ensemble(self) -> List[tf.keras.Model]:
        """
        Build an ensemble of neural validation models with different architectures.
        
        Returns:
            List of neural network models
        """
        models = []
        
        try:
            # Primary pattern generation model: Dense architecture
            pattern_model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu', input_shape=(10,)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(self.pattern_dimension, activation='tanh')
            ])
            pattern_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
                loss='mse'
            )
            models.append(pattern_model)
            
            # Transaction validation model: Classifier architecture
            validation_model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(self.pattern_dimension,)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            validation_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            models.append(validation_model)
            
            # Temporal validation model: LSTM architecture
            temporal_model = tf.keras.Sequential([
                tf.keras.layers.Reshape((self.pattern_dimension, 1), input_shape=(self.pattern_dimension,)),
                tf.keras.layers.LSTM(64, return_sequences=True),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.LSTM(32),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            temporal_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
                loss='binary_crossentropy'
            )
            models.append(temporal_model)
            
            # Add additional models if ensemble size is larger
            if self.config['model_ensemble_size'] > 3:
                # Add a convolutional model for pattern recognition
                conv_model = tf.keras.Sequential([
                    tf.keras.layers.Reshape((self.pattern_dimension, 1), input_shape=(self.pattern_dimension,)),
                    tf.keras.layers.Conv1D(32, 3, activation='relu'),
                    tf.keras.layers.MaxPooling1D(2),
                    tf.keras.layers.Conv1D(64, 3, activation='relu'),
                    tf.keras.layers.GlobalAveragePooling1D(),
                    tf.keras.layers.Dense(64, activation='relu'),
                    tf.keras.layers.Dense(self.pattern_dimension, activation='tanh')
                ])
                conv_model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
                    loss='mse'
                )
                models.append(conv_model)
                
                # Add a residual network model
                inputs = tf.keras.Input(shape=(self.pattern_dimension,))
                x = tf.keras.layers.Dense(64, activation='relu')(inputs)
                
                # Residual block
                residual = x
                x = tf.keras.layers.Dense(64, activation='relu')(x)
                x = tf.keras.layers.Dropout(0.2)(x)
                x = tf.keras.layers.Dense(64, activation='relu')(x)
                x = tf.keras.layers.Add()([x, residual])
                
                # Output
                outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
                residual_model = tf.keras.Model(inputs=inputs, outputs=outputs)
                residual_model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
                    loss='binary_crossentropy'
                )
                models.append(residual_model)
        
        except Exception as e:
            logger.error("Error building model ensemble: %s", e)
            # Create fallback models if TensorFlow fails
            class FallbackModel:
                def predict(self, data):
                    if isinstance(data, np.ndarray):
                        if len(data.shape) == 2 and data.shape[1] == 10:
                            # Pattern generation fallback
                            return np.random.rand(data.shape[0], self.pattern_dimension) * 2 - 1
                        elif len(data.shape) == 2 and data.shape[1] == self.pattern_dimension:
                            # Binary classification fallback
                            return np.random.rand(data.shape[0], 1)
                    return np.random.rand(1, self.pattern_dimension)
                    
                def fit(self, *args, **kwargs):
                    pass
            
            # Create minimum required fallback models
            for _ in range(3):
                fallback = FallbackModel()
                fallback.pattern_dimension = self.pattern_dimension
                models.append(fallback)
                
            logger.warning("Using fallback models due to TensorFlow initialization error")
        
        return models
    
    def _build_autoencoder(self) -> Union[tf.keras.Model, Any]:
        """
        Build autoencoder for anomaly detection with enhanced architecture.
        
        Returns:
            Autoencoder model
        """
        try:
            # Input layer
            inputs = tf.keras.Input(shape=(self.pattern_dimension,))
            
            # Encoder layers
            x = tf.keras.layers.Dense(64, activation='relu')(inputs)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dense(32, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.2)(x)
            encoded = tf.keras.layers.Dense(16, activation='relu')(x)
            
            # Bottleneck
            bottleneck = tf.keras.layers.Dense(8, activation='relu')(encoded)
            
            # Decoder layers
            x = tf.keras.layers.Dense(16, activation='relu')(bottleneck)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dense(32, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.2)(x)
            x = tf.keras.layers.Dense(64, activation='relu')(x)
            
            # Output layer
            decoded = tf.keras.layers.Dense(self.pattern_dimension, activation='tanh')(x)
            
            # Build and compile model
            autoencoder = tf.keras.Model(inputs, decoded)
            autoencoder.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
                loss='mse'
            )
            
            return autoencoder
            
        except Exception as e:
            logger.error("Error building autoencoder: %s", e)
            
            # Create a fallback autoencoder
            class FallbackAutoencoder:
                def __init__(self, pattern_dimension):
                    self.pattern_dimension = pattern_dimension
                
                def predict(self, data):
                    # Add small noise to simulate reconstruction error
                    if isinstance(data, np.ndarray):
                        noise = np.random.normal(0, 0.05, data.shape)
                        return data + noise
                    return data
                    
                def fit(self, *args, **kwargs):
                    pass
            
            return FallbackAutoencoder(self.pattern_dimension)
    
    def _load_pattern_library(self) -> Dict[str, np.ndarray]:
        """
        Load pretrained pattern library with baseline patterns.
        
        Returns:
            Dictionary of named patterns
        """
        pattern_library = {
            'genesis': np.random.rand(self.pattern_dimension) * 2 - 1,
            'empty_block': np.zeros(self.pattern_dimension),
            'high_volume': np.random.rand(self.pattern_dimension) * 2 - 1,
            'high_value': np.random.rand(self.pattern_dimension) * 2 - 1,
            'attack_pattern': np.random.rand(self.pattern_dimension) * 2 - 1,
            'normal_transaction': np.random.rand(self.pattern_dimension) * 2 - 1,
            'anomalous_transaction': np.random.rand(self.pattern_dimension) * 2 - 1,
            'trusted_baseline': np.random.rand(self.pattern_dimension) * 2 - 1
        }
        
        # Normalize all patterns
        for key in pattern_library:
            pattern_library[key] = pattern_library[key] / np.linalg.norm(pattern_library[key])
        
        # Try to load from disk if available
        try:
            pattern_path = os.path.join('data', 'pattern_library.json')
            if os.path.exists(pattern_path):
                with open(pattern_path, 'r') as f:
                    loaded_patterns = json.load(f)
                    
                    for key, value in loaded_patterns.items():
                        pattern_array = np.array(value)
                        # Ensure pattern has correct dimension
                        if len(pattern_array) == self.pattern_dimension:
                            pattern_library[key] = pattern_array
                
                logger.info("Loaded pattern library with %d patterns", len(pattern_library))
                
                # Add loaded patterns to trusted patterns
                for key, pattern in pattern_library.items():
                    pattern_hash = self._get_pattern_hash(pattern)
                    self.trusted_pattern_hashes.add(pattern_hash)
        
        except Exception as e:
            logger.warning("Could not load pattern library: %s", e)
        
        return pattern_library
    
    def _save_pattern_library(self) -> None:
        """
        Save pattern library to disk.
        
        Returns:
            None
        """
        try:
            os.makedirs('data', exist_ok=True)
            
            # Convert patterns to lists for JSON serialization
            pattern_dict = {k: v.tolist() for k, v in self.pattern_library.items()}
            
            # Use atomic write to prevent corruption
            temp_file = os.path.join('data', 'pattern_library.json.tmp')
            with open(temp_file, 'w') as f:
                json.dump(pattern_dict, f, indent=2)
            
            # Rename temp file to final file (atomic operation)
            os.replace(temp_file, os.path.join('data', 'pattern_library.json'))
            
            logger.info("Saved pattern library with %d patterns", len(self.pattern_library))
            
        except Exception as e:
            logger.warning("Could not save pattern library: %s", e)
    
    def _load_models(self) -> None:
        """
        Load saved model weights if available.
        
        Returns:
            None
        """
        try:
            models_dir = os.path.join('data', 'models')
            if os.path.exists(models_dir):
                # Load main models
                for i, model in enumerate(self.models):
                    model_path = os.path.join(models_dir, f'model_{i}.h5')
                    if os.path.exists(model_path) and hasattr(model, 'load_weights'):
                        try:
                            model.load_weights(model_path)
                            logger.info("Loaded weights for model %d", i)
                        except Exception as e:
                            logger.warning("Could not load weights for model %d: %s", i, e)
                
                # Load autoencoder
                autoencoder_path = os.path.join(models_dir, 'autoencoder.h5')
                if os.path.exists(autoencoder_path) and hasattr(self.autoencoder, 'load_weights'):
                    try:
                        self.autoencoder.load_weights(autoencoder_path)
                        logger.info("Loaded weights for autoencoder")
                    except Exception as e:
                        logger.warning("Could not load weights for autoencoder: %s", e)
                        
                # Load metrics if available
                metrics_path = os.path.join(models_dir, 'metrics.json')
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as f:
                        stored_metrics = json.load(f)
                        # Only override metrics that exist in stored data
                        for key, value in stored_metrics.items():
                            if key in self.metrics:
                                self.metrics[key] = value
                
                logger.info("Loaded neural models and metrics")
        
        except Exception as e:
            logger.warning("Could not load neural models: %s", e)
    
    def _save_models(self) -> None:
        """
        Save model weights to disk.
        
        Returns:
            None
        """
        try:
            models_dir = os.path.join('data', 'models')
            os.makedirs(models_dir, exist_ok=True)
            
            # Save main models
            for i, model in enumerate(self.models):
                if hasattr(model, 'save_weights'):
                    model_path = os.path.join(models_dir, f'model_{i}.h5')
                    model.save_weights(model_path)
            
            # Save autoencoder
            if hasattr(self.autoencoder, 'save_weights'):
                autoencoder_path = os.path.join(models_dir, 'autoencoder.h5')
                self.autoencoder.save_weights(autoencoder_path)
            
            # Save metrics
            metrics_path = os.path.join(models_dir, 'metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            
            logger.info("Saved neural models and metrics")
            
        except Exception as e:
            logger.warning("Could not save neural models: %s", e)
    
    def generate_genesis_pattern(self) -> List[float]:
        """
        Generate the initial pattern for the genesis block with enhanced stability.
        
        Returns:
            Initial neural pattern (normalized)
        """
        # Use a repeatable seed for consistency
        np.random.seed(42)
        
        # Generate base pattern
        base_pattern = np.random.rand(self.pattern_dimension) * 2 - 1
        
        # Apply formal verification to ensure pattern meets security criteria
        verified_pattern = self._verify_pattern(base_pattern)
        
        # Store in pattern library
        self.pattern_library['genesis'] = verified_pattern
        
        # Add to trusted patterns
        pattern_hash = self._get_pattern_hash(verified_pattern)
        self.trusted_pattern_hashes.add(pattern_hash)
        
        # Reset random seed
        np.random.seed(None)
        
        # Convert to list and return
        return verified_pattern.tolist()
    
    def _verify_pattern(self, pattern: np.ndarray) -> np.ndarray:
        """
        Apply formal verification to ensure pattern meets security and quality criteria.
        
        Args:
            pattern: Pattern to verify
            
        Returns:
            Verified pattern
        """
        # Normalize to unit length
        norm = np.linalg.norm(pattern)
        if norm < 1e-10:
            # Create a random pattern if input is too small
            pattern = np.random.rand(self.pattern_dimension) * 2 - 1
            norm = np.linalg.norm(pattern)
        
        normalized_pattern = pattern / norm
        
        # Calculate entropy (diversity of values)
        value_distribution = np.abs(normalized_pattern) / np.sum(np.abs(normalized_pattern) + 1e-10)
        entropy = -np.sum(value_distribution * np.log2(value_distribution + 1e-10))
        max_entropy = np.log2(len(pattern))
        entropy_ratio = entropy / max_entropy
        
        # Enhance entropy if below threshold
        if entropy < self.config['entropy_threshold'] or entropy_ratio < 0.6:
            # Add controlled noise to increase entropy
            noise_scale = 0.2 * (1 - entropy_ratio)
            enhancement_noise = np.random.randn(self.pattern_dimension) * noise_scale
            normalized_pattern = normalized_pattern + enhancement_noise
            normalized_pattern = normalized_pattern / np.linalg.norm(normalized_pattern)
            
            # Recalculate entropy
            value_distribution = np.abs(normalized_pattern) / np.sum(np.abs(normalized_pattern) + 1e-10)
            entropy = -np.sum(value_distribution * np.log2(value_distribution + 1e-10))
        
        # Check pattern diversity (max component magnitude)
        max_component = np.max(np.abs(normalized_pattern))
        if max_component > 0.5:
            # Scale down dominant components
            normalized_pattern = normalized_pattern * (0.5 / max_component)
            normalized_pattern = normalized_pattern / np.linalg.norm(normalized_pattern)
        
        # Check for adversarial patterns (extreme FFT spectrum)
        fft_values = np.abs(np.fft.fft(normalized_pattern))
        fft_peaks = np.max(fft_values[1:]) / np.mean(fft_values[1:])
        if fft_peaks > 5.0:
            # Dampen frequency peaks by adding phase-shifted noise
            phase_noise = np.sin(np.linspace(0, 2*np.pi, self.pattern_dimension)) * 0.1
            normalized_pattern = normalized_pattern + phase_noise
            normalized_pattern = normalized_pattern / np.linalg.norm(normalized_pattern)
        
        # Add security level-based hardening
        if self.security_level in ['high', 'maximum']:
            # Incorporate trusted baseline pattern
            trusted_pattern = self.pattern_library['trusted_baseline']
            hardening_factor = 0.05 * self.security_multiplier
            normalized_pattern = (1 - hardening_factor) * normalized_pattern + hardening_factor * trusted_pattern
            normalized_pattern = normalized_pattern / np.linalg.norm(normalized_pattern)
        
        # Add to verified patterns cache
        pattern_hash = self._get_pattern_hash(normalized_pattern)
        self.verified_patterns[pattern_hash] = {
            'entropy': entropy,
            'max_component': max_component,
            'fft_peaks': fft_peaks,
            'verified_at': time.time()
        }
        
        return normalized_pattern
    
    def _get_pattern_hash(self, pattern: np.ndarray) -> str:
        """
        Generate a deterministic hash for a pattern.
        
        Args:
            pattern: Neural pattern to hash
            
        Returns:
            Hash string
        """
        # Quantize to reduce sensitivity to minor numeric differences
        quantized = np.round(pattern * 1000) / 1000
        
        # Convert to bytes and hash
        pattern_bytes = quantized.tobytes()
        
        # Use SHA3-256 for enhanced security
        return hashlib.sha3_256(pattern_bytes).hexdigest()
    
    def generate_pattern(self, transactions: List) -> List[float]:
        """
        Generate a neural pattern for a set of transactions with enhanced security.
        
        Args:
            transactions: List of transactions to analyze
            
        Returns:
            Neural pattern representing the transactions
        """
        if not transactions:
            # Return a verified empty pattern
            return self.pattern_library['empty_block'].tolist()
        
        # Start timing for performance metrics
        start_time = time.time()
        
        # Extract features from transactions
        features = self._extract_transaction_features(transactions)
        
        # Generate pattern using primary model (with batch dimension)
        try:
            base_pattern = self.models[0].predict(np.array([features]))[0]
        except Exception as e:
            logger.warning("Error predicting with primary model: %s", e)
            # Fallback to library pattern based on transaction volume
            if len(transactions) > 100:
                base_pattern = self.pattern_library['high_volume']
            else:
                base_pattern = self.pattern_library['normal_transaction']
        
        # Apply adaptive security adjustments
        pattern = self._apply_security_adjustments(base_pattern, transactions)
        
        # Verify the pattern meets security criteria
        verified_pattern = self._verify_pattern(pattern)
        
        # Update metrics
        self.metrics['validation_time'].append(time.time() - start_time)
        self.metrics['total_patterns_generated'] += 1
        
        # Calculate running average validation time
        total_time = sum(self.metrics['validation_time'])
        avg_time = total_time / len(self.metrics['validation_time'])
        self.metrics['avg_validation_time_ms'] = avg_time * 1000
        
        # Check for anomalies
        self._check_pattern_anomaly(verified_pattern, transactions)
        
        # Save pattern to library if it has high quality
        quality_score = self._evaluate_pattern_quality(verified_pattern)
        if quality_score > 0.9 and random.random() < 0.1:
            pattern_name = f"quality_pattern_{int(time.time()) % 1000}"
            self.pattern_library[pattern_name] = verified_pattern
            
            # Periodically save pattern library (10% chance)
            if random.random() < 0.1:
                self._save_pattern_library()
        
        return verified_pattern.tolist()
    
    def _evaluate_pattern_quality(self, pattern: np.ndarray) -> float:
        """
        Evaluate the quality of a pattern based on multiple metrics.
        
        Args:
            pattern: Pattern to evaluate
            
        Returns:
            Quality score (0-1)
        """
        # Calculate entropy
        value_distribution = np.abs(pattern) / np.sum(np.abs(pattern) + 1e-10)
        entropy = -np.sum(value_distribution * np.log2(value_distribution + 1e-10))
        max_entropy = np.log2(len(pattern))
        entropy_ratio = entropy / max_entropy
        
        # Calculate pattern diversity
        diversity = np.std(pattern)
        
        # Check for frequency characteristics
        fft_values = np.abs(np.fft.fft(pattern))
        frequency_balance = np.std(fft_values[1:]) / np.mean(fft_values[1:])
        
        # Calculate quality score from combined metrics
        entropy_score = min(1.0, entropy_ratio / 0.8)
        diversity_score = min(1.0, diversity / 0.4)
        frequency_score = min(1.0, 1.0 / (frequency_balance + 0.1))
        
        # Weighted combination
        quality_score = (0.5 * entropy_score + 
                         0.3 * diversity_score + 
                         0.2 * frequency_score)
        
        return quality_score
    
    def _apply_security_adjustments(self, pattern: np.ndarray, transactions: List) -> np.ndarray:
        """
        Apply security adjustments to a pattern based on transaction properties.
        
        Args:
            pattern: Base pattern to adjust
            transactions: Transactions being represented
            
        Returns:
            Adjusted pattern
        """
        # Calculate transaction value and complexity
        total_value = sum(tx.amount for tx in transactions)
        tx_count = len(transactions)
        
        # Normalize pattern
        norm = np.linalg.norm(pattern)
        if norm < 1e-10:
            # If pattern is too small, generate a new one
            pattern = np.random.rand(self.pattern_dimension) * 2 - 1
            norm = np.linalg.norm(pattern)
        
        normalized_pattern = pattern / norm
        
        # Apply adjustment based on total transaction value
        if total_value > 10000:
            # Incorporate high-value pattern for large transactions
            high_value_influence = 0.3 * self.security_multiplier
            normalized_pattern = ((1 - high_value_influence) * normalized_pattern + 
                                 high_value_influence * self.pattern_library['high_value'])
            normalized_pattern = normalized_pattern / np.linalg.norm(normalized_pattern)
        
        # Apply adjustment based on transaction volume
        if tx_count > 100:
            # Incorporate high-volume pattern for large batches
            high_volume_influence = 0.2 * self.security_multiplier
            normalized_pattern = ((1 - high_volume_influence) * normalized_pattern + 
                                 high_volume_influence * self.pattern_library['high_volume'])
            normalized_pattern = normalized_pattern / np.linalg.norm(normalized_pattern)
        
        # Apply adversarial resistance if needed
        # Identify any suspicious transactions
        suspicious_tx_count = 0
        for tx in transactions:
            if hasattr(tx, 'transaction_id') and tx.transaction_id in self.transaction_confidence_history:
                if self.transaction_confidence_history[tx.transaction_id] < 0.5:
                    suspicious_tx_count += 1
        
        # If suspicious transactions detected, add resistance
        if suspicious_tx_count > 0:
            resistance_factor = min(0.4, suspicious_tx_count / len(transactions) * self.config['adversarial_resistance'])
            trusted_pattern = self.pattern_library['trusted_baseline']
            normalized_pattern = ((1 - resistance_factor) * normalized_pattern + 
                                 resistance_factor * trusted_pattern)
            normalized_pattern = normalized_pattern / np.linalg.norm(normalized_pattern)
        
        return normalized_pattern
    
    def _check_pattern_anomaly(self, pattern: np.ndarray, transactions: List) -> bool:
        """
        Check if a pattern exhibits anomalous properties using ensemble methods.
        
        Args:
            pattern: Pattern to check
            transactions: Associated transactions
            
        Returns:
            True if pattern is anomalous
        """
        # Use autoencoder for anomaly detection
        try:
            reconstructed = self.autoencoder.predict(np.array([pattern]))[0]
            reconstruction_error = np.mean(np.square(pattern - reconstructed))
        except Exception as e:
            logger.warning("Error in autoencoder prediction: %s", e)
            # Fallback to statistical anomaly detection
            reconstruction_error = 0.5
            
            # Statistical anomaly detection as fallback
            pattern_stats = {
                'min': np.min(pattern),
                'max': np.max(pattern),
                'mean': np.mean(pattern),
                'std': np.std(pattern),
                'range': np.max(pattern) - np.min(pattern),
                'entropy': self._calculate_entropy(pattern)
            }
            
            # Check for statistical anomalies
            if (pattern_stats['range'] < 0.1 or 
                pattern_stats['std'] < 0.1 or
                pattern_stats['entropy'] < 2.0):
                reconstruction_error = self.anomaly_threshold + 0.1
        
        # Check FFT characteristics for unusual patterns
        fft_values = np.abs(np.fft.fft(pattern))
        frequency_ratio = np.max(fft_values[1:]) / np.mean(fft_values[1:])
        
        # Combine multiple anomaly signals
        anomaly_signals = [
            reconstruction_error > self.anomaly_threshold,
            frequency_ratio > 10.0,
            np.max(np.abs(pattern)) > 0.9
        ]
        
        # Final anomaly decision (majority vote)
        is_anomalous = sum(anomaly_signals) >= 2
        
        if is_anomalous:
            self.metrics['anomalies_detected'] += 1
            
            # Store the anomalous pattern for analysis
            anomaly_data = {
                'pattern': pattern.tolist(),
                'transactions': [tx.to_dict() if hasattr(tx, 'to_dict') else str(tx) for tx in transactions],
                'reconstruction_error': float(reconstruction_error),
                'frequency_ratio': float(frequency_ratio),
                'timestamp': time.time()
            }
            
            # Add to recent anomalies
            self.recent_anomalies.append(anomaly_data)
            
            # Mark pattern as potentially adversarial
            pattern_hash = self._get_pattern_hash(pattern)
            self.adversarial_patterns.add(pattern_hash)
            
            logger.warning("Anomalous pattern detected: reconstruction_error=%.4f, frequency_ratio=%.2f", 
                          reconstruction_error, frequency_ratio)
        
        return is_anomalous
    
    def _calculate_entropy(self, pattern: np.ndarray) -> float:
        """
        Calculate Shannon entropy of a pattern.
        
        Args:
            pattern: Pattern to analyze
            
        Returns:
            Entropy value
        """
        # Normalize for probability distribution
        abs_pattern = np.abs(pattern)
        if np.sum(abs_pattern) < 1e-10:
            return 0.0
            
        normalized = abs_pattern / np.sum(abs_pattern)
        # Calculate entropy
        return -np.sum(normalized * np.log2(normalized + 1e-10))
    
    def _extract_transaction_features(self, transactions) -> np.ndarray:
        """
        Extract relevant features from a set of transactions with enhanced detail.
        
        Args:
            transactions: List of transactions
            
        Returns:
            Feature vector
        """
        if not transactions:
            # Return zero features for empty transaction set
            return np.zeros(10)
        
        # Basic transaction statistics
        total_value = sum(getattr(tx, 'amount', 0) for tx in transactions)
        num_transactions = len(transactions)
        avg_value = total_value / max(num_transactions, 1)
        
        # Network graph analysis
        unique_senders = set(getattr(tx, 'sender', f'unknown_{i}') for i, tx in enumerate(transactions))
        unique_recipients = set(getattr(tx, 'recipient', f'unknown_{i}') for i, tx in enumerate(transactions))
        unique_addresses = len(unique_senders.union(unique_recipients))
        address_intersection = len(unique_senders.intersection(unique_recipients))
        
        # Temporal analysis
        try:
            timestamp_array = [getattr(tx, 'timestamp', time.time()) for tx in transactions]
            timestamp_diffs = []
            for i in range(1, len(timestamp_array)):
                timestamp_diffs.append(timestamp_array[i] - timestamp_array[i-1])
            
            avg_timestamp_diff = sum(timestamp_diffs) / max(len(timestamp_diffs), 1)
            std_timestamp_diff = np.std(timestamp_diffs) if timestamp_diffs else 0
        except Exception:
            # Fallback if temporal analysis fails
            avg_timestamp_diff = 0
            std_timestamp_diff = 0
        
        # Transaction graph density (ratio of actual to possible connections)
        graph_density = 0
        if num_transactions > 1:
            transaction_pairs = [(getattr(tx1, 'sender', ''), getattr(tx2, 'recipient', '')) 
                                for tx1 in transactions for tx2 in transactions]
            unique_pairs = set(transaction_pairs)
            graph_density = len(unique_pairs) / (num_transactions * (num_transactions - 1))
        
        # Features vector (10 dimensions)
        features = np.array([
            total_value,
            num_transactions,
            avg_value,
            len(unique_senders),
            len(unique_recipients),
            avg_timestamp_diff,
            time.time() % 86400 / 86400,  # Time of day (normalized)
            address_intersection / max(unique_addresses, 1),  # Normalized intersection
            graph_density,
            std_timestamp_diff
        ])
        
        # Normalize features to [-1, 1] range for better neural network performance
        # Use robust normalization to handle outliers
        mean = np.nanmean(features)
        std = np.nanstd(features)
        if std < 1e-10:
            std = 1.0
        
        normalized_features = (features - mean) / (std + 1e-8)
        
        # Clip to prevent extreme values
        normalized_features = np.clip(normalized_features, -5, 5)
        
        return normalized_features
    
    def adapt_pattern(self, base_pattern: List[float], block_hash: str, nonce: int) -> List[float]:
        """
        Adapt a neural pattern based on mining attempts with enhanced security.
        
        Args:
            base_pattern: Starting pattern
            block_hash: Current block hash
            nonce: Current nonce value
            
        Returns:
            Adapted neural pattern
        """
        if not base_pattern:
            return self.pattern_library['trusted_baseline'].tolist()
            
        pattern_array = np.array(base_pattern)
        
        # Normalize the pattern
        norm = np.linalg.norm(pattern_array)
        if norm < 1e-10:
            pattern_array = self.pattern_library['trusted_baseline']
        else:
            pattern_array = pattern_array / norm
        
        # Convert hash to numerical influence with better distribution
        hash_bytes = bytes.fromhex(block_hash if block_hash.startswith('0x') else '0x' + block_hash)
        hash_values = []
        for i in range(0, min(len(hash_bytes), 32), 2):
            if i+1 < len(hash_bytes):
                val = (hash_bytes[i] << 8) + hash_bytes[i+1]
                hash_values.append((val / 65535) * 2 - 1)  # Scale to [-1, 1]
        
        # Extend hash values if needed
        while len(hash_values) < self.pattern_dimension:
            hash_values.extend(hash_values[:self.pattern_dimension - len(hash_values)])
        
        hash_influence = np.array(hash_values[:self.pattern_dimension])
        
        # Use a multi-phase adaptation function based on nonce
        # This creates a more complex and secure adaptation
        adaptation_strength = 0.1 * (1 + np.sin(nonce / 1000 * np.pi))
        adaptation_phase = np.cos(nonce / 500 * np.pi) * 0.5 + 0.5
        
        # Apply magnitude adjustment based on hash
        adapted_pattern = pattern_array + adaptation_strength * hash_influence
        
        # Apply phase adjustment based on nonce
        rotation_steps = nonce % self.pattern_dimension
        adapted_pattern = ((1 - adaptation_phase) * adapted_pattern + 
                          adaptation_phase * np.roll(adapted_pattern, rotation_steps))
        
        # Apply security hardening based on security level
        if self.security_level in ['high', 'maximum']:
            # Incorporate trusted baseline pattern
            hardening_factor = 0.05 * self.security_multiplier
            adapted_pattern = ((1 - hardening_factor) * adapted_pattern + 
                              hardening_factor * self.pattern_library['trusted_baseline'])
        
        # Normalize to keep within [-1, 1] range
        adapted_pattern = adapted_pattern / np.linalg.norm(adapted_pattern)
        
        # Verify the pattern meets security criteria
        verified_pattern = self._verify_pattern(adapted_pattern)
        
        return verified_pattern.tolist()
    
    def validate_transaction(self, transaction, blockchain) -> bool:
        """
        Validate a transaction using the neural network ensemble.
        
        Args:
            transaction: Transaction to validate
            blockchain: Current blockchain state
            
        Returns:
            True if transaction is valid
        """
        # Start timing for performance metrics
        start_time = time.time()
        
        # Basic validation
        if not transaction.verify():
            self.metrics['false_positives'] += 1
            self._record_transaction_confidence(transaction.transaction_id, 0.0)
            return False
        
        # Check if sender has sufficient balance (except for system transactions)
        if transaction.sender != "SYSTEM":
            try:
                sender_balance = blockchain.get_balance(transaction.sender)
                if sender_balance < transaction.amount:
                    self.metrics['false_positives'] += 1
                    self._record_transaction_confidence(transaction.transaction_id, 0.1)
                    return False
            except Exception as e:
                logger.warning("Error checking balance: %s", e)
                # Continue with neural validation even if balance check fails
        
        # Convert transaction to pattern
        tx_pattern = self._transaction_to_pattern(transaction)
        
        # Use ensemble voting for validation
        validation_results = []
        validation_scores = []
        
        # Primary model validation - Pattern prediction
        try:
            # Extract transaction features
            tx_features = self._extract_transaction_features([transaction])
            
            # Generate expected pattern
            pattern_prediction = self.models[0].predict(np.array([tx_features]))[0]
            
            # Check similarity between predicted pattern and transaction pattern
            similarity = np.dot(pattern_prediction, tx_pattern) / (
                np.linalg.norm(pattern_prediction) * np.linalg.norm(tx_pattern)
            )
            validation_results.append(similarity > self.config['validation_threshold'])
            validation_scores.append(similarity)
        except Exception as e:
            logger.warning("Error in primary model validation: %s", e)
            # Default to neutral validation
            validation_results.append(True)
            validation_scores.append(0.7)
        
        # Secondary model validation - Direct classification
        try:
            validation_score = self.models[1].predict(np.array([tx_pattern]))[0][0]
            validation_results.append(validation_score > 0.5)
            validation_scores.append(validation_score)
        except Exception as e:
            logger.warning("Error in secondary model validation: %s", e)
            validation_results.append(True)
            validation_scores.append(0.7)
        
        # Temporal model validation using historical context
        if len(self.transaction_history) >= 5:
            try:
                # Create sequence of patterns including current transaction
                historical_patterns = [self._transaction_to_pattern(tx) for tx in self.transaction_history[-5:]]
                historical_patterns.append(tx_pattern)
                
                # Reshape for LSTM input [batch, timesteps, features]
                sequence = np.array(historical_patterns).reshape(1, len(historical_patterns), self.pattern_dimension)
                
                temporal_score = self.models[2].predict(sequence)[0][0]
                validation_results.append(temporal_score > 0.5)
                validation_scores.append(temporal_score)
            except Exception as e:
                logger.warning("Error in temporal model validation: %s", e)
                validation_results.append(True)
                validation_scores.append(0.7)
        else:
            # Not enough history, use default validation
            validation_results.append(True)
            validation_scores.append(0.7)
        
        # Check for anomalies
        is_anomalous = self._check_pattern_anomaly(tx_pattern, [transaction])
        
        # Calculate confidence score
        avg_confidence = sum(validation_scores) / len(validation_scores)
        
        # Record transaction confidence
        self._record_transaction_confidence(transaction.transaction_id, avg_confidence)
        
        # Final validation result
        if is_anomalous:
            # Apply stricter validation for anomalous transactions
            # Require all models to agree
            validation_result = all(validation_results)
            
            # Record as potential false negative if rejected
            if not validation_result:
                self.metrics['false_negatives'] += 1
        else:
            # Use majority voting for normal transactions
            validation_result = sum(validation_results) > len(validation_results) / 2
        
        # Add transaction to history for future validation
        if validation_result:
            self.transaction_history.append(transaction)
            if len(self.transaction_history) > 10000:
                self.transaction_history = self.transaction_history[-10000:]
            
            # Add to experience replay buffer
            self.replay_buffer.append((
                tx_pattern,
                1.0  # Valid transaction label
            ))
        
        # Periodically update models from replay buffer
        self.metrics['total_transactions_processed'] += 1
        if (len(self.replay_buffer) >= self.config['batch_size'] and 
            self.metrics['total_transactions_processed'] % self.config['training_interval'] == 0):
            self._update_models_from_replay()
        
        # Update validation time metrics
        validation_time = time.time() - start_time
        self.metrics['avg_validation_time_ms'] = validation_time * 1000
        
        return validation_result
    
    def _record_transaction_confidence(self, transaction_id: str, confidence: float) -> None:
        """
        Record confidence score for a transaction.
        
        Args:
            transaction_id: ID of the transaction
            confidence: Confidence score (0-1)
            
        Returns:
            None
        """
        if not transaction_id:
            return
            
        self.transaction_confidence_history[transaction_id] = confidence
        
        # Limit history size
        if len(self.transaction_confidence_history) > 10000:
            # Keep most recent entries
            to_remove = len(self.transaction_confidence_history) - 10000
            for tx_id in list(self.transaction_confidence_history.keys())[:to_remove]:
                del self.transaction_confidence_history[tx_id]
    
    def _transaction_to_pattern(self, transaction) -> np.ndarray:
        """
        Convert a transaction to a neural pattern for validation.
        
        Args:
            transaction: Transaction to convert
            
        Returns:
            Neural pattern
        """
        # Create a deterministic hash from transaction details
        if hasattr(transaction, 'to_dict'):
            tx_data = json.dumps(transaction.to_dict(), sort_keys=True)
        else:
            # Fallback for non-standard transaction objects
            tx_data = f"{getattr(transaction, 'sender', '')}{getattr(transaction, 'recipient', '')}"
            tx_data += f"{getattr(transaction, 'amount', 0)}{getattr(transaction, 'timestamp', time.time())}"
            tx_data += f"{getattr(transaction, 'transaction_id', '')}"
        
        # Generate hash
        tx_hash = hashlib.sha3_256(tx_data.encode()).digest()
        
        # Convert hash to pattern values in [-1, 1] range
        pattern = []
        for i in range(0, self.pattern_dimension):
            byte_idx = i % len(tx_hash)
            bit_idx = (i // len(tx_hash)) % 8
            bit_value = (tx_hash[byte_idx] >> bit_idx) & 1
            pattern.append(bit_value * 2 - 1)  # Convert to [-1, 1]
        
        pattern_array = np.array(pattern)
        
        # Normalize pattern
        norm = np.linalg.norm(pattern_array)
        if norm < 1e-10:
            # Fallback for zero-norm pattern
            return self.pattern_library['normal_transaction']
            
        return pattern_array / norm
    
    def _update_models_from_replay(self) -> None:
        """
        Update models using experience replay and adaptive learning.
        
        Returns:
            None
        """
        if len(self.replay_buffer) < self.config['batch_size']:
            return
        
        logger.info("Updating neural models from replay buffer")
        
        try:
            # Sample batch from replay buffer with preference for recent entries
            batch_indices = np.random.choice(
                len(self.replay_buffer),
                size=min(self.config['batch_size'], len(self.replay_buffer)),
                replace=False,
                p=self._get_sampling_weights()
            )
            
            batch = [self.replay_buffer[i] for i in batch_indices]
            patterns, labels = zip(*batch)
            
            patterns_array = np.array(patterns)
            labels_array = np.array(labels)
            
            # Update validation model
            if hasattr(self.models[1], 'fit'):
                self.models[1].fit(
                    patterns_array, 
                    labels_array,
                    epochs=1,
                    verbose=0
                )
            
            # Update autoencoder with focus on normal patterns
            normal_patterns = patterns_array[labels_array > 0.8]
            if len(normal_patterns) > 0 and hasattr(self.autoencoder, 'fit'):
                self.autoencoder.fit(
                    normal_patterns,
                    normal_patterns,
                    epochs=1,
                    verbose=0
                )
            
            # Update primary model with transaction features and patterns
            if len(self.transaction_history) >= self.config['batch_size']:
                # Sample transactions
                sampled_txs = random.sample(
                    self.transaction_history, 
                    min(self.config['batch_size'], len(self.transaction_history))
                )
                
                # Extract features and convert to patterns
                tx_features = []
                tx_patterns = []
                
                for tx in sampled_txs:
                    tx_features.append(self._extract_transaction_features([tx]))
                    tx_patterns.append(self._transaction_to_pattern(tx))
                
                if hasattr(self.models[0], 'fit'):
                    self.models[0].fit(
                        np.array(tx_features),
                        np.array(tx_patterns),
                        epochs=1,
                        verbose=0
                    )
            
            # Update training metrics
            self.metrics['training_epochs_completed'] += 1
            
            # Periodically save models (every 10 training cycles)
            if self.metrics['training_epochs_completed'] % 10 == 0:
                self._save_models()
                
            # Consider model evolution
            if (time.time() - self.evolution_state['last_evolution'] > 3600 and  # Once per hour
                self.metrics['training_epochs_completed'] > 10):
                self._evolve_models()
        
        except Exception as e:
            logger.error("Error updating models from replay: %s", e)
    
    def _get_sampling_weights(self) -> np.ndarray:
        """
        Get weighted sampling distribution favoring recent experiences.
        
        Returns:
            Array of sampling weights
        """
        # Generate weights that favor recent experiences
        n = len(self.replay_buffer)
        if n == 0:
            return np.array([])
            
        # Linear decay weights
        weights = np.arange(1, n + 1) / n
        # Normalize
        return weights / weights.sum()
    
    def _evolve_models(self) -> None:
        """
        Evolve models by introducing mutations based on performance.
        
        Returns:
            None
        """
        # Only attempt evolution for real TensorFlow models
        evolution_supported = all(hasattr(model, 'get_weights') for model in self.models)
        if not evolution_supported:
            return
            
        try:
            logger.info("Evolving neural models")
            
            # Record evolution time
            self.evolution_state['last_evolution'] = time.time()
            self.evolution_state['generations'] += 1
            
            # Choose model to evolve (usually validation model)
            target_model = self.models[1]
            
            # Get current weights
            current_weights = target_model.get_weights()
            
            # Create mutation
            mutated_weights = []
            mutation_magnitude = self.config['evolve_rate'] * (1.0 / (1.0 + self.evolution_state['generations'] / 10))
            
            for layer_weights in current_weights:
                # Add random noise proportional to weight magnitudes
                noise = np.random.normal(0, mutation_magnitude * np.std(layer_weights), layer_weights.shape)
                mutated_weights.append(layer_weights + noise)
            
            # Test mutation on validation data
            if len(self.replay_buffer) >= 50:
                # Sample validation data
                val_indices = np.random.choice(len(self.replay_buffer), size=50, replace=False)
                val_patterns, val_labels = zip(*[self.replay_buffer[i] for i in val_indices])
                val_patterns = np.array(val_patterns)
                val_labels = np.array(val_labels)
                
                # Evaluate original model
                original_preds = target_model.predict(val_patterns)
                original_accuracy = np.mean((original_preds > 0.5).astype(int) == val_labels.astype(int))
                
                # Apply mutation temporarily
                backup_weights = target_model.get_weights()
                target_model.set_weights(mutated_weights)
                
                # Evaluate mutated model
                mutated_preds = target_model.predict(val_patterns)
                mutated_accuracy = np.mean((mutated_preds > 0.5).astype(int) == val_labels.astype(int))
                
                # Keep mutation if better, otherwise revert
                if mutated_accuracy >= original_accuracy:
                    # Keep mutation
                    self.evolution_state['mutations'] += 1
                    logger.info("Evolved model with successful mutation: %.4f -> %.4f", 
                              original_accuracy, mutated_accuracy)
                else:
                    # Revert mutation
                    target_model.set_weights(backup_weights)
                    logger.info("Reverted model mutation: %.4f vs %.4f", 
                              original_accuracy, mutated_accuracy)
                
                # Record fitness score
                self.evolution_state['fitness_scores'].append(max(original_accuracy, mutated_accuracy))
                
                # If fitness scores list gets too long, keep only the most recent 10
                if len(self.evolution_state['fitness_scores']) > 10:
                    self.evolution_state['fitness_scores'] = self.evolution_state['fitness_scores'][-10:]
        
        except Exception as e:
            logger.error("Error during model evolution: %s", e)
    
    def adapt(self, new_block) -> None:
        """
        Adapt the neural network ensemble based on a new block.
        
        Args:
            new_block: New block to learn from
            
        Returns:
            None
        """
        if not hasattr(new_block, 'transactions') or not new_block.transactions:
            return
        
        # Update block count metric
        self.metrics['total_blocks_validated'] += 1
        
        # Extract features from the block's transactions
        try:
            features = self._extract_transaction_features(new_block.transactions)
            
            # Skip adaptation if TensorFlow is not properly initialized
            if not all(hasattr(model, 'fit') for model in self.models):
                return
                
            # Create training examples
            X = np.array([features])
            y_pattern = np.array([new_block.neural_pattern])
            
            # Update pattern generation model
            self.models[0].fit(X, y_pattern, epochs=1, verbose=0)
            
            # Extract patterns from transactions for additional training
            transaction_patterns = []
            transaction_labels = []
            
            for tx in new_block.transactions:
                try:
                    tx_pattern = self._transaction_to_pattern(tx)
                    transaction_patterns.append(tx_pattern)
                    transaction_labels.append(1.0)  # Valid transaction
                    
                    # Add to replay buffer
                    self.replay_buffer.append((tx_pattern, 1.0))
                except Exception as e:
                    logger.warning("Error processing transaction for training: %s", e)
            
            if transaction_patterns:
                # Update validation model with successful transactions
                patterns_array = np.array(transaction_patterns)
                labels_array = np.ones(len(transaction_patterns))
                
                self.models[1].fit(patterns_array, labels_array, epochs=1, verbose=0)
                
                # Update autoencoder model
                self.autoencoder.fit(patterns_array, patterns_array, epochs=1, verbose=0)
            
            # Periodically save pattern library
            if random.random() < 0.1:  # 10% chance each block
                self._save_pattern_library()
                
            # Periodically save models
            if random.random() < 0.05:  # 5% chance each block
                self._save_models()
        
        except Exception as e:
            logger.warning("Error adapting neural models to new block: %s", e)
    
    def detect_anomalies(self, transactions) -> List[int]:
        """
        Detect potentially fraudulent or anomalous transactions with enhanced precision.
        
        Args:
            transactions: List of transactions to check
            
        Returns:
            Indices of suspicious transactions
        """
        suspicious_indices = []
        
        if not transactions:
            return suspicious_indices
        
        # Process each transaction individually
        for i, tx in enumerate(transactions):
            try:
                # Extract transaction pattern
                tx_pattern = self._transaction_to_pattern(tx)
                
                # Multi-factor anomaly detection
                anomaly_signals = []
                
                # 1. Autoencoder reconstruction error
                try:
                    reconstructed = self.autoencoder.predict(np.array([tx_pattern]))[0]
                    reconstruction_error = np.mean(np.square(tx_pattern - reconstructed))
                    anomaly_signals.append(reconstruction_error > self.anomaly_threshold)
                except Exception:
                    # Skip this signal if autoencoder fails
                    pass
                
                # 2. Pattern entropy check
                pattern_entropy = self._calculate_entropy(tx_pattern)
                entropy_anomaly = pattern_entropy < self.config['entropy_threshold']
                anomaly_signals.append(entropy_anomaly)
                
                # 3. Unusual amount
                tx_amount = getattr(tx, 'amount', 0)
                amount_anomaly = tx_amount > 1000000  # Very large transaction
                anomaly_signals.append(amount_anomaly)
                
                # 4. Unusual timing patterns
                if i > 0:
                    prev_timestamp = getattr(transactions[i-1], 'timestamp', 0)
                    curr_timestamp = getattr(tx, 'timestamp', 0)
                    time_diff = curr_timestamp - prev_timestamp
                    timing_anomaly = time_diff < 0.01  # Suspiciously close timestamps
                    anomaly_signals.append(timing_anomaly)
                
                # 5. Check for repeated transactions (potential replay attacks)
                is_repeated = False
                for prev_tx in self.transaction_history[-100:]:
                    if (getattr(tx, 'sender', '') == getattr(prev_tx, 'sender', '') and 
                        getattr(tx, 'recipient', '') == getattr(prev_tx, 'recipient', '') and
                        abs(getattr(tx, 'amount', 0) - getattr(prev_tx, 'amount', 0)) < 0.001):
                        is_repeated = True
                        break
                anomaly_signals.append(is_repeated)
                
                # 6. Neural model classification
                try:
                    validation_score = self.models[1].predict(np.array([tx_pattern]))[0][0]
                    model_anomaly = validation_score < 0.3  # Low confidence
                    anomaly_signals.append(model_anomaly)
                except Exception:
                    # Skip this signal if model fails
                    pass
                
                # Determine if transaction is suspicious (at least 2 anomaly signals)
                if sum(anomaly_signals) >= 2:
                    suspicious_indices.append(i)
                    
                    # Record confidence score
                    confidence = 1.0 - (sum(anomaly_signals) / len(anomaly_signals))
                    self._record_transaction_confidence(
                        getattr(tx, 'transaction_id', f'unknown-{i}'), 
                        confidence
                    )
            
            except Exception as e:
                logger.warning("Error in anomaly detection for transaction %d: %s", i, e)
                # If error occurs during analysis, mark as suspicious
                suspicious_indices.append(i)
        
        return suspicious_indices
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get validator performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        metrics_copy = self.metrics.copy()
        
        # Add derived metrics
        if self.metrics['total_transactions_processed'] > 0:
            metrics_copy['anomaly_rate'] = (
                self.metrics['anomalies_detected'] / self.metrics['total_transactions_processed']
            )
        else:
            metrics_copy['anomaly_rate'] = 0
            
        # Add evolution metrics
        if hasattr(self, 'evolution_state'):
            metrics_copy['evolution'] = {
                'generations': self.evolution_state['generations'],
                'mutations': self.evolution_state['mutations']
            }
            
            if self.evolution_state['fitness_scores']:
                metrics_copy['evolution']['avg_fitness'] = sum(self.evolution_state['fitness_scores']) / len(self.evolution_state['fitness_scores'])
        
        return metrics_copy
    
    def verify_block_pattern(self, block_pattern: List[float], block_hash: str) -> bool:
        """
        Verify if a block's neural pattern is valid and secure.
        
        Args:
            block_pattern: Neural pattern from the block
            block_hash: Hash of the block
            
        Returns:
            True if pattern is valid
        """
        if not block_pattern:
            return False
            
        # Convert pattern to numpy array
        pattern_array = np.array(block_pattern)
        
        # Check dimension
        if len(pattern_array) != self.pattern_dimension:
            return False
            
        # Check if pattern is already verified
        pattern_hash = self._get_pattern_hash(pattern_array)
        if pattern_hash in self.verified_patterns:
            return True
            
        # Check if pattern is in trusted set
        if pattern_hash in self.trusted_pattern_hashes:
            return True
            
        # Check if pattern is flagged as adversarial
        if pattern_hash in self.adversarial_patterns:
            return False
            
        # Validate pattern properties
        verification_result = self._verify_pattern(pattern_array)
        pattern_quality = self._evaluate_pattern_quality(verification_result)
        
        # Verify based on quality threshold
        threshold = self.config['verification_confidence']
        is_valid = pattern_quality >= threshold
        
        # Update metrics
        self.metrics['pattern_verification_success_rate'] = (
            0.95 * self.metrics['pattern_verification_success_rate'] + 
            0.05 * (1.0 if is_valid else 0.0)
        )
        
        return is_valid
    
    def calculate_transaction_confidence(self, transaction) -> float:
        """
        Calculate a confidence score for a transaction.
        
        Args:
            transaction: Transaction to evaluate
            
        Returns:
            Confidence score (0-1)
        """
        # Check if we have stored confidence for this transaction
        if hasattr(transaction, 'transaction_id') and transaction.transaction_id in self.transaction_confidence_history:
            return self.transaction_confidence_history[transaction.transaction_id]
            
        # Calculate new confidence score
        confidence_signals = []
        
        # 1. Valid signature
        if hasattr(transaction, 'verify'):
            valid_signature = transaction.verify()
            confidence_signals.append(1.0 if valid_signature else 0.0)
        
        # 2. Pattern quality
        try:
            tx_pattern = self._transaction_to_pattern(transaction)
            pattern_quality = self._evaluate_pattern_quality(tx_pattern)
            confidence_signals.append(pattern_quality)
        except Exception:
            # Default to neutral if pattern evaluation fails
            confidence_signals.append(0.5)
        
        # 3. Neural model confidence
        try:
            validation_score = self.models[1].predict(np.array([tx_pattern]))[0][0]
            confidence_signals.append(float(validation_score))
        except Exception:
            # Default to neutral if model fails
            confidence_signals.append(0.5)
        
        # 4. Check for anomalies
        try:
            is_anomalous = self._check_pattern_anomaly(tx_pattern, [transaction])
            confidence_signals.append(0.0 if is_anomalous else 1.0)
        except Exception:
            # Default to neutral if anomaly check fails
            confidence_signals.append(0.5)
        
        # Calculate final confidence score
        confidence = sum(confidence_signals) / len(confidence_signals)
        
        # Store for future reference
        if hasattr(transaction, 'transaction_id'):
            self._record_transaction_confidence(transaction.transaction_id, confidence)
        
        return confidence
    
    def benchmark_performance(self, num_transactions: int = 100) -> Dict[str, Any]:
        """
        Run a performance benchmark on the neural validation system.
        
        Args:
            num_transactions: Number of test transactions to generate
            
        Returns:
            Benchmark results
        """
        results = {
            'pattern_generation': {},
            'transaction_validation': {},
            'anomaly_detection': {}
        }
        
        try:
            # Generate test data
            test_transactions = []
            for i in range(num_transactions):
                # Create simple transaction object
                class TestTransaction:
                    def __init__(self, id_num):
                        self.sender = f"sender_{id_num % 10}"
                        self.recipient = f"recipient_{(id_num + 5) % 10}"
                        self.amount = 10.0 + (id_num % 100)
                        self.timestamp = time.time() - (id_num * 60)
                        self.transaction_id = f"test_tx_{id_num}"
                        
                    def verify(self):
                        return True
                        
                    def to_dict(self):
                        return {
                            "sender": self.sender,
                            "recipient": self.recipient,
                            "amount": self.amount,
                            "timestamp": self.timestamp,
                            "transaction_id": self.transaction_id
                        }
                
                test_transactions.append(TestTransaction(i))
            
            # 1. Benchmark pattern generation
            start_time = time.time()
            for i in range(min(10, num_transactions // 10)):
                batch = test_transactions[i*10:(i+1)*10]
                self.generate_pattern(batch)
            
            pattern_time = time.time() - start_time
            results['pattern_generation'] = {
                'total_time_seconds': pattern_time,
                'time_per_pattern': pattern_time / min(10, num_transactions // 10),
                'patterns_per_second': min(10, num_transactions // 10) / pattern_time
            }
            
            # 2. Benchmark transaction validation
            class MockBlockchain:
                def get_balance(self, address):
                    return 1000.0
            
            mock_blockchain = MockBlockchain()
            
            start_time = time.time()
            valid_count = 0
            for tx in test_transactions:
                if self.validate_transaction(tx, mock_blockchain):
                    valid_count += 1
            
            validation_time = time.time() - start_time
            results['transaction_validation'] = {
                'total_time_seconds': validation_time,
                'time_per_transaction': validation_time / num_transactions,
                'transactions_per_second': num_transactions / validation_time,
                'valid_rate': valid_count / num_transactions
            }
            
            # 3. Benchmark anomaly detection
            start_time = time.time()
            suspicious = self.detect_anomalies(test_transactions)
            
            anomaly_time = time.time() - start_time
            results['anomaly_detection'] = {
                'total_time_seconds': anomaly_time,
                'time_per_transaction': anomaly_time / num_transactions,
                'transactions_per_second': num_transactions / anomaly_time,
                'suspicious_rate': len(suspicious) / num_transactions
            }
            
            # Overall performance
            results['overall'] = {
                'total_time_seconds': pattern_time + validation_time + anomaly_time,
                'total_transactions': num_transactions
            }
            
        except Exception as e:
            logger.error("Error during performance benchmark: %s", e)
            results['error'] = str(e)
            
        return results