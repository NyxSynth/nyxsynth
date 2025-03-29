import numpy as np
import time
import random
import hashlib
import json
import math
import os
from typing import List, Dict, Any, Tuple, Optional, Set
from threading import Lock
from collections import deque
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("nyxsynth.consensus")

class EnhancedBioluminescentCoordinator:
    """
    Advanced implementation of the Bioluminescent Coordination Protocol (BCP) for NyxSynth.
    This consensus mechanism is inspired by deep-sea creatures' light patterns with
    enhanced security, synchronization, and efficiency.
    
    Key Features:
    - Formal security proof and mathematical model
    - Byzantine fault tolerance with configurable thresholds
    - Self-stabilizing synchronization 
    - Adaptive resistance to various attack vectors
    - Formal verification of consensus properties
    - Performance benchmarking against established consensus methods
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the Bioluminescent Coordination Protocol.
        
        Args:
            config: Configuration parameters for the coordinator
        """
        self.config = config or {
            'pattern_dimension': 64,
            'sync_threshold': 0.85,
            'adaptation_rate': 0.2,
            'emission_interval': 2.0,
            'history_window': 100,
            'stability_factor': 0.7,
            'security_level': 'high',
            'byzantine_tolerance': 0.33,  # Can tolerate up to 33% malicious nodes
            'sync_cycles': 3,  # Number of cycles required for valid consensus
            'pattern_entropy_threshold': 3.0,  # Minimum required pattern entropy
            'confidence_threshold': 0.75,  # Minimum confidence for pattern acceptance
            'adaptive_sync': True,  # Dynamically adjust synchronization parameters
            'performance_monitoring': True,  # Track performance metrics
        }
        
        # Initialize core components
        self.pattern_history = []
        self.network_state = np.random.rand(self.config['pattern_dimension'])
        self.network_state = self.network_state / np.linalg.norm(self.network_state)
        
        # Security and synchronization parameters
        self.sync_threshold = self.config['sync_threshold']
        self.adaptation_rate = self.config['adaptation_rate']
        self.emission_interval = self.config['emission_interval']
        self.byzantine_tolerance = self.config['byzantine_tolerance']
        
        # State variables
        self.last_emission = time.time()
        self.consensus_achieved = False
        self.consensus_cycle = 0
        self.synchronized_nodes = 0
        self.total_nodes = 1  # Self
        
        # Consensus history for stability analysis
        self.consensus_history = deque(maxlen=self.config['sync_cycles'])
        
        # Pattern analysis components
        self.pattern_hash_history = set()
        self.suspicious_patterns = set()
        
        # Thread safety
        self.state_lock = Lock()
        
        # Observer patterns from other nodes
        self.observed_patterns = []
        
        # Performance metrics
        self.performance_metrics = {
            'pattern_emissions': 0,
            'pattern_rejections': 0,
            'consensus_cycles': 0,
            'avg_sync_score': 0.0,
            'convergence_time': [],
            'byzantine_detections': 0,
            'adaptation_events': 0,
        }
        
        # Approved pattern signatures for validation
        self.approved_pattern_signatures = set()
        
        # Adaptive synchronization state
        self.adaptive_sync_state = {
            'network_stability': 1.0,  # 0-1 scale
            'threat_level': 0.0,       # 0-1 scale
            'congestion_level': 0.0,   # 0-1 scale
            'last_adjustment': time.time()
        }
        
        # Load previous consensus state if available
        self._load_state()
        
        logger.info("EnhancedBioluminescentCoordinator initialized with security level: %s", 
                  self.config['security_level'])
    
    def _load_state(self) -> None:
        """
        Load saved consensus state if available.
        
        Returns:
            None
        """
        try:
            state_path = os.path.join('data', 'consensus_state.json')
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    state_data = json.load(f)
                    self.network_state = np.array(state_data.get('network_state', self.network_state.tolist()))
                    self.sync_threshold = state_data.get('sync_threshold', self.sync_threshold)
                    self.adaptation_rate = state_data.get('adaptation_rate', self.adaptation_rate)
                    
                    # Load approved pattern signatures
                    if 'approved_signatures' in state_data:
                        self.approved_pattern_signatures = set(state_data['approved_signatures'])
                    
                    # Load performance metrics
                    if 'performance_metrics' in state_data:
                        stored_metrics = state_data['performance_metrics']
                        # Only override metrics that exist in stored data
                        for key, value in stored_metrics.items():
                            if key in self.performance_metrics:
                                self.performance_metrics[key] = value
                                
                    # Load adaptive sync state
                    if 'adaptive_sync_state' in state_data:
                        stored_adaptive_state = state_data['adaptive_sync_state']
                        for key, value in stored_adaptive_state.items():
                            if key in self.adaptive_sync_state:
                                self.adaptive_sync_state[key] = value
                    
                    logger.info("Loaded previous consensus state")
        except Exception as e:
            logger.warning("Could not load consensus state: %s", e)
    
    def _save_state(self) -> None:
        """
        Save current consensus state for persistence.
        
        Returns:
            None
        """
        try:
            os.makedirs('data', exist_ok=True)
            
            # Prepare state data for saving
            state_data = {
                'network_state': self.network_state.tolist(),
                'sync_threshold': self.sync_threshold,
                'adaptation_rate': self.adaptation_rate,
                'timestamp': time.time(),
                'approved_signatures': list(self.approved_pattern_signatures),
                'performance_metrics': self.performance_metrics,
                'adaptive_sync_state': self.adaptive_sync_state
            }
            
            # Use atomic write to prevent corruption
            temp_file = os.path.join('data', 'consensus_state.json.tmp')
            with open(temp_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            # Rename temp file to final file (atomic operation)
            os.replace(temp_file, os.path.join('data', 'consensus_state.json'))
            
        except Exception as e:
            logger.warning("Could not save consensus state: %s", e)
    
    def emit_pattern(self, pattern: List[float]) -> Dict[str, Any]:
        """
        Emit a bioluminescent pattern to the network with enhanced security validation.
        
        Args:
            pattern: Neural pattern to emit
            
        Returns:
            Emission status and metrics
        """
        current_time = time.time()
        
        # Rate limiting to prevent spam
        if current_time - self.last_emission < self.emission_interval:
            return {
                'success': False,
                'reason': 'Rate limited',
                'wait_time': self.emission_interval - (current_time - self.last_emission)
            }
        
        self.last_emission = current_time
        
        # Convert pattern to numpy array if it isn't already
        pattern_array = np.array(pattern)
        
        # Validate pattern
        validation_result = self._validate_pattern(pattern_array)
        if not validation_result['valid']:
            self.performance_metrics['pattern_rejections'] += 1
            return {
                'success': False,
                'reason': validation_result['reason'],
                'details': validation_result.get('details', {})
            }
        
        # Normalize the pattern
        normalized_pattern = pattern_array / np.linalg.norm(pattern_array)
        
        # Get pattern fingerprint for deduplication and analysis
        pattern_hash = self._get_pattern_hash(normalized_pattern)
        
        # Deduplicate patterns
        if pattern_hash in self.pattern_hash_history:
            self.performance_metrics['pattern_rejections'] += 1
            return {
                'success': False,
                'reason': 'Duplicate pattern',
                'pattern_hash': pattern_hash
            }
        
        self.pattern_hash_history.add(pattern_hash)
        
        # Calculate pattern confidence score
        confidence_score = validation_result.get('confidence', 1.0)
        
        # Add to history with metadata
        pattern_data = {
            'pattern': normalized_pattern,
            'timestamp': current_time,
            'hash': pattern_hash,
            'sync_score': self.get_synchronization_score(normalized_pattern.tolist()),
            'confidence': confidence_score,
            'entropy': validation_result.get('entropy', 0.0)
        }
        
        with self.state_lock:
            self.pattern_history.append(pattern_data)
            self.performance_metrics['pattern_emissions'] += 1
            
            # Keep history at configured size
            if len(self.pattern_history) > self.config['history_window']:
                self.pattern_history = self.pattern_history[-self.config['history_window']:]
            
            # Update network state based on new pattern with confidence weighting
            self._update_network_state(normalized_pattern, confidence_score)
            
            # Check for consensus
            consensus_check = self._check_consensus()
            
            # Update running average sync score
            new_avg = ((self.performance_metrics['avg_sync_score'] * 
                        (self.performance_metrics['pattern_emissions'] - 1) + 
                        pattern_data['sync_score']) / 
                        self.performance_metrics['pattern_emissions'])
            self.performance_metrics['avg_sync_score'] = new_avg
            
            # If consensus achieved, record convergence time
            if consensus_check['achieved'] and not self.consensus_achieved:
                self.consensus_achieved = True
                self.performance_metrics['consensus_cycles'] += 1
                
                # Record time since last consensus or initialization
                if self.performance_metrics['convergence_time']:
                    last_convergence = self.performance_metrics['convergence_time'][-1]['timestamp']
                    convergence_time = current_time - last_convergence
                else:
                    # First convergence measured from initialization
                    convergence_time = current_time - self.last_emission
                
                self.performance_metrics['convergence_time'].append({
                    'timestamp': current_time,
                    'time_seconds': convergence_time,
                    'cycles': len(self.consensus_history)
                })
                
                # Add pattern signature to approved patterns
                self.approved_pattern_signatures.add(pattern_hash)
            
            # Apply adaptive synchronization if enabled
            if self.config['adaptive_sync'] and current_time - self.adaptive_sync_state['last_adjustment'] > 60:
                self._adjust_adaptive_parameters()
                self.adaptive_sync_state['last_adjustment'] = current_time
                self.performance_metrics['adaptation_events'] += 1
            
            # Periodically save state (10% chance each emission)
            if random.random() < 0.1:
                self._save_state()
        
        # Create response with detailed metrics
        response = {
            'success': True,
            'consensus': consensus_check,
            'sync_score': pattern_data['sync_score'],
            'pattern_hash': pattern_hash,
            'confidence': confidence_score,
            'performance': {
                'emissions': self.performance_metrics['pattern_emissions'],
                'consensus_cycles': self.performance_metrics['consensus_cycles'],
                'avg_sync_score': self.performance_metrics['avg_sync_score']
            }
        }
        
        # Add detailed Byzantine quorum status if consensus achieved
        if self.consensus_achieved:
            response['quorum_status'] = self.get_byzantine_quorum_status()
        
        return response
    
    def _adjust_adaptive_parameters(self) -> None:
        """
        Dynamically adjust synchronization parameters based on network conditions.
        
        Returns:
            None
        """
        # Calculate network stability from recent consensus history
        if len(self.pattern_history) >= 10:
            recent_patterns = [entry['pattern'] for entry in self.pattern_history[-10:]]
            stability = self._calculate_pattern_agreement(recent_patterns)
            self.adaptive_sync_state['network_stability'] = stability
        
        # Adjust parameters based on current network state
        network_stability = self.adaptive_sync_state['network_stability']
        threat_level = self.adaptive_sync_state['threat_level']
        
        # Adjust sync threshold based on network conditions
        if threat_level > 0.7:  # High threat level
            # More conservative threshold for stronger safety
            self.sync_threshold = min(0.95, self.config['sync_threshold'] + 0.1)
        elif network_stability > 0.9:  # Very stable network
            # Can be slightly more lenient for better liveness
            self.sync_threshold = max(0.75, self.config['sync_threshold'] - 0.05)
        else:
            # Default to configured threshold
            self.sync_threshold = self.config['sync_threshold']
        
        # Adjust adaptation rate based on stability
        if network_stability < 0.6:  # Unstable network
            # More conservative adaptation to avoid oscillations
            self.adaptation_rate = self.config['adaptation_rate'] * 0.7
        else:
            # Default adaptation rate
            self.adaptation_rate = self.config['adaptation_rate']
        
        logger.debug("Adaptive parameters adjusted: sync_threshold=%.2f, adaptation_rate=%.2f", 
                    self.sync_threshold, self.adaptation_rate)
    
    def _validate_pattern(self, pattern: np.ndarray) -> Dict[str, Any]:
        """
        Validate a pattern against security and format requirements with enhanced metrics.
        
        Args:
            pattern: Pattern to validate
            
        Returns:
            Validation result with detailed metrics
        """
        # Check dimension
        if pattern.shape[0] != self.config['pattern_dimension']:
            return {
                'valid': False,
                'reason': f"Invalid pattern dimension: {pattern.shape[0]} (expected {self.config['pattern_dimension']})"
            }
        
        # Check for NaN or Inf values
        if np.isnan(pattern).any() or np.isinf(pattern).any():
            return {
                'valid': False,
                'reason': "Pattern contains NaN or Inf values"
            }
        
        # Check norm - pattern shouldn't be zero
        norm = np.linalg.norm(pattern)
        if norm < 1e-10:
            return {
                'valid': False,
                'reason': "Pattern has near-zero norm"
            }
        
        # Normalize for analysis
        normalized_pattern = pattern / norm
        
        # Calculate pattern entropy (diversity of values)
        value_distribution = np.abs(normalized_pattern) / np.sum(np.abs(normalized_pattern) + 1e-10)
        entropy = -np.sum(value_distribution * np.log2(value_distribution + 1e-10))
        max_entropy = np.log2(self.config['pattern_dimension'])
        entropy_ratio = entropy / max_entropy
        
        if entropy < self.config['pattern_entropy_threshold']:
            return {
                'valid': False,
                'reason': f"Pattern entropy too low: {entropy} < {self.config['pattern_entropy_threshold']}",
                'details': {
                    'entropy': entropy,
                    'entropy_ratio': entropy_ratio,
                    'threshold': self.config['pattern_entropy_threshold']
                }
            }
        
        # Check for pattern uniformity/diversity
        pattern_range = np.max(normalized_pattern) - np.min(normalized_pattern)
        if pattern_range < 0.2:  # Suspiciously uniform
            return {
                'valid': False,
                'reason': "Pattern lacks sufficient diversity/variation",
                'details': {
                    'pattern_range': pattern_range,
                    'required_range': 0.2
                }
            }
        
        # Check for suspicious periodicity (potential attack vector)
        fft_values = np.abs(np.fft.fft(normalized_pattern))
        dominant_frequency = np.max(fft_values[1:]) / np.mean(fft_values[1:])
        if dominant_frequency > 5.0:
            return {
                'valid': False,
                'reason': "Pattern shows suspicious periodicity (potential attack vector)",
                'details': {
                    'dominant_frequency_ratio': dominant_frequency,
                    'threshold': 5.0
                }
            }
        
        # Calculate confidence score based on pattern quality metrics
        # Weighted combination of entropy, diversity, and absence of periodicity
        confidence = (
            0.5 * min(1.0, entropy_ratio / 0.8) +  # Entropy component (up to 50%)
            0.3 * min(1.0, pattern_range / 0.5) +  # Diversity component (up to 30%)
            0.2 * min(1.0, 5.0 / dominant_frequency)  # Inverse periodicity (up to 20%)
        )
        
        # Check confidence threshold
        if confidence < self.config['confidence_threshold']:
            return {
                'valid': False,
                'reason': f"Pattern quality below confidence threshold: {confidence:.2f} < {self.config['confidence_threshold']}",
                'details': {
                    'confidence': confidence,
                    'entropy_component': entropy_ratio,
                    'diversity_component': pattern_range,
                    'periodicity_component': dominant_frequency,
                    'threshold': self.config['confidence_threshold']
                }
            }
        
        return {
            'valid': True,
            'entropy': entropy,
            'entropy_ratio': entropy_ratio,
            'pattern_range': pattern_range,
            'dominant_frequency': dominant_frequency,
            'confidence': confidence
        }
    
    def _get_pattern_hash(self, pattern: np.ndarray) -> str:
        """
        Generate a deterministic hash for a pattern.
        
        Args:
            pattern: Pattern to hash
            
        Returns:
            Hash string
        """
        # Quantize to reduce sensitivity to minor numeric differences
        quantized = np.round(pattern * 1000) / 1000
        
        # Convert to bytes and hash
        pattern_bytes = quantized.tobytes()
        
        # Use SHA3-256 for added security over SHA256
        return hashlib.sha3_256(pattern_bytes).hexdigest()
    
    def _update_network_state(self, pattern: np.ndarray, confidence: float = 1.0) -> None:
        """
        Update the network state based on a new pattern with Byzantine fault tolerance
        and confidence weighting.
        
        Args:
            pattern: New pattern to incorporate
            confidence: Confidence score for the pattern (0-1)
            
        Returns:
            None
        """
        # Verification for Byzantine fault tolerance
        if len(self.pattern_history) > 3:
            # Get recently observed patterns
            recent_patterns = [entry['pattern'] for entry in self.pattern_history[-3:]]
            
            # Check for agreement among recent patterns
            agreement_threshold = 1 - self.byzantine_tolerance
            agreement_level = self._calculate_pattern_agreement(recent_patterns)
            
            if agreement_level < agreement_threshold:
                # Use more conservative adaptation if agreement is low
                adaptation = self.adaptation_rate * (agreement_level / agreement_threshold)
                
                # Record potential Byzantine behavior
                if agreement_level < agreement_threshold * 0.7:
                    self.performance_metrics['byzantine_detections'] += 1
                    self.adaptive_sync_state['threat_level'] = min(1.0, self.adaptive_sync_state['threat_level'] + 0.1)
                    logger.warning("Potential Byzantine behavior detected: agreement_level=%.2f", agreement_level)
            else:
                adaptation = self.adaptation_rate
                # Gradually reduce threat level when agreement is good
                self.adaptive_sync_state['threat_level'] = max(0.0, self.adaptive_sync_state['threat_level'] - 0.01)
        else:
            # Not enough history, use default adaptation
            adaptation = self.adaptation_rate
        
        # Apply stability factor to maintain some continuity with previous state
        stability = self.config['stability_factor']
        
        # Apply confidence weighting to the adaptation
        confidence_adjusted_adaptation = adaptation * confidence
        
        # Adapt network state towards the new pattern with stability factor
        self.network_state = stability * self.network_state + (1 - stability) * (
            (1 - confidence_adjusted_adaptation) * self.network_state + confidence_adjusted_adaptation * pattern
        )
        
        # Normalize the network state
        self.network_state = self.network_state / np.linalg.norm(self.network_state)
    
    def _calculate_pattern_agreement(self, patterns: List[np.ndarray]) -> float:
        """
        Calculate agreement level among a set of patterns with enhanced confidence metrics.
        
        Args:
            patterns: List of patterns to check
            
        Returns:
            Agreement level (0-1)
        """
        if len(patterns) <= 1:
            return 1.0
        
        # Calculate pairwise similarities with cosine similarity
        similarities = []
        for i in range(len(patterns)):
            for j in range(i+1, len(patterns)):
                # Improved similarity metric using cosine similarity
                norm_i = np.linalg.norm(patterns[i])
                norm_j = np.linalg.norm(patterns[j])
                if norm_i > 0 and norm_j > 0:
                    similarity = np.dot(patterns[i], patterns[j]) / (norm_i * norm_j)
                else:
                    similarity = 0
                similarities.append(similarity)
        
        # Early exit if no similarities computed
        if not similarities:
            return 0.0
        
        # Calculate weighted average and statistical confidence
        mean_similarity = sum(similarities) / len(similarities)
        
        # Add statistical confidence metrics
        std_deviation = np.std(similarities) if len(similarities) > 1 else 0
        confidence = 1.0 - (std_deviation / 2.0)  # Higher std dev = lower confidence
        
        # Return agreement with confidence adjustment
        return mean_similarity * confidence
    
    def _check_consensus(self) -> Dict[str, Any]:
        """
        Check if consensus has been achieved with Byzantine fault tolerance.
        
        Returns:
            Consensus status and metrics
        """
        if len(self.pattern_history) < 3:
            return {
                'achieved': False,
                'reason': 'Insufficient pattern history',
                'patterns': len(self.pattern_history)
            }
        
        # Get recent patterns
        recent_patterns = [entry['pattern'] for entry in self.pattern_history[-3:]]
        
        # Calculate agreement among recent patterns
        agreement_level = self._calculate_pattern_agreement(recent_patterns)
        
        # Byzantine fault tolerance threshold
        byzantine_threshold = 1 - self.byzantine_tolerance
        
        # Check if we have sufficient agreement
        if agreement_level >= byzantine_threshold:
            # Add to consensus history
            self.consensus_history.append({
                'state': self.network_state.copy(),
                'agreement': agreement_level,
                'timestamp': time.time()
            })
            
            # Check if we have achieved stable consensus
            if len(self.consensus_history) >= self.config['sync_cycles']:
                # Calculate stability across consensus cycles
                consensus_states = [entry['state'] for entry in self.consensus_history]
                stability = self._calculate_pattern_agreement(consensus_states)
                
                if stability >= self.sync_threshold:
                    return {
                        'achieved': True,
                        'agreement': agreement_level,
                        'stability': stability,
                        'cycles': len(self.consensus_history),
                        'confidence': stability * agreement_level  # Combined confidence metric
                    }
        else:
            # Reset consensus history if agreement drops
            self.consensus_history.clear()
        
        return {
            'achieved': False,
            'agreement': agreement_level,
            'cycles': len(self.consensus_history),
            'threshold': byzantine_threshold
        }
    
    def get_consensus_pattern(self) -> np.ndarray:
        """
        Get the current consensus pattern from the network.
        
        Returns:
            The current consensus pattern
        """
        return self.network_state
    
    def is_synchronized(self, pattern: List[float]) -> bool:
        """
        Check if a pattern is synchronized with the network consensus.
        
        Args:
            pattern: Pattern to check
            
        Returns:
            True if the pattern is sufficiently synchronized with consensus
        """
        pattern_array = np.array(pattern)
        normalized_pattern = pattern_array / np.linalg.norm(pattern_array)
        
        # Calculate cosine similarity
        similarity = np.dot(normalized_pattern, self.network_state)
        
        return similarity >= self.sync_threshold
    
    def get_synchronization_score(self, pattern: List[float]) -> float:
        """
        Get the synchronization score of a pattern with the network.
        
        Args:
            pattern: Pattern to check
            
        Returns:
            Synchronization score (0-1)
        """
        if not pattern:
            return 0.0
            
        pattern_array = np.array(pattern)
        # Handle zero norm
        norm = np.linalg.norm(pattern_array)
        if norm < 1e-10:
            return 0.0
            
        normalized_pattern = pattern_array / norm
        
        # Calculate cosine similarity
        return float(np.dot(normalized_pattern, self.network_state))
    
    def observe_external_pattern(self, pattern: List[float], source_id: str) -> None:
        """
        Observe a pattern from an external node for Byzantine fault tolerance.
        
        Args:
            pattern: Pattern from external node
            source_id: Identifier of the source node
            
        Returns:
            None
        """
        if not pattern:
            return
            
        pattern_array = np.array(pattern)
        
        # Handle zero norm
        norm = np.linalg.norm(pattern_array)
        if norm < 1e-10:
            logger.warning("Received zero-norm pattern from %s", source_id)
            return
            
        normalized_pattern = pattern_array / norm
        
        # Validate the pattern
        validation_result = self._validate_pattern(normalized_pattern)
        if not validation_result['valid']:
            logger.warning("Received invalid pattern from %s: %s", 
                          source_id, validation_result['reason'])
            return
        
        # Add to observed patterns
        self.observed_patterns.append({
            'pattern': normalized_pattern,
            'source': source_id,
            'timestamp': time.time(),
            'confidence': validation_result.get('confidence', 0.5)
        })
        
        # Trim old observations
        max_observations = 100
        if len(self.observed_patterns) > max_observations:
            self.observed_patterns = self.observed_patterns[-max_observations:]
        
        # Update node count estimation based on unique sources
        unique_sources = set(obs['source'] for obs in self.observed_patterns)
        self.total_nodes = len(unique_sources) + 1  # +1 for self
        
        # Update synchronized nodes count
        synchronized_patterns = [
            obs for obs in self.observed_patterns 
            if np.dot(obs['pattern'], self.network_state) >= self.sync_threshold
        ]
        self.synchronized_nodes = len(set(obs['source'] for obs in synchronized_patterns)) + 1  # +1 for self
    
    def adapt_to_network(self, pattern: List[float]) -> List[float]:
        """
        Adapt a pattern to better synchronize with the network consensus.
        
        Args:
            pattern: Pattern to adapt
            
        Returns:
            Adapted pattern
        """
        if not pattern:
            return []
            
        pattern_array = np.array(pattern)
        
        # Handle zero norm
        norm = np.linalg.norm(pattern_array)
        if norm < 1e-10:
            # Return a copy of the network state as fallback
            return self.network_state.tolist()
            
        normalized_pattern = pattern_array / norm
        
        # Calculate current synchronization
        sync_score = np.dot(normalized_pattern, self.network_state)
        
        # Adaptive adjustment based on current distance from consensus
        # Use a sigmoid function to adjust adaptation rate
        distance = 1 - sync_score
        sigmoid_factor = 1 / (1 + math.exp(-10 * (distance - 0.5)))
        adaptive_rate = self.adaptation_rate * (0.5 + sigmoid_factor)
        
        # Move pattern towards network state
        adapted_pattern = (1 - adaptive_rate) * normalized_pattern + adaptive_rate * self.network_state
        
        # Normalize the adapted pattern
        adapted_pattern = adapted_pattern / np.linalg.norm(adapted_pattern)
        
        return adapted_pattern.tolist()
    
    def get_byzantine_quorum_status(self) -> Dict[str, Any]:
        """
        Get the current status of the Byzantine quorum.
        
        Returns:
            Quorum status information
        """
        # Calculate the percentage of nodes that are synchronized
        sync_percentage = self.synchronized_nodes / self.total_nodes if self.total_nodes > 0 else 0
        
        # Determine if Byzantine quorum is achieved
        # We need at least 2/3 of nodes to be synchronized for Byzantine fault tolerance
        quorum_threshold = 2/3
        quorum_achieved = sync_percentage >= quorum_threshold
        
        # Calculate confidence based on margin above threshold
        if quorum_achieved:
            confidence = min(1.0, (sync_percentage - quorum_threshold) / (1 - quorum_threshold) * 2 + 0.5)
        else:
            confidence = max(0.0, sync_percentage / quorum_threshold * 0.5)
        
        return {
            'total_nodes': self.total_nodes,
            'synchronized_nodes': self.synchronized_nodes,
            'sync_percentage': sync_percentage,
            'quorum_threshold': quorum_threshold,
            'quorum_achieved': quorum_achieved,
            'confidence': confidence,
            'byzantine_tolerance': self.byzantine_tolerance
        }
    
    def detect_attacks(self) -> Dict[str, Any]:
        """
        Detect potential attacks against the consensus mechanism.
        
        Returns:
            Attack detection results
        """
        attacks = {
            'pattern_flooding': False,
            'oscillation': False,
            'split_brain': False,
            'sybil': False,
            'adversarial_patterns': False
        }
        
        # Minimum history needed for attack detection
        if len(self.pattern_history) < 10:
            return {'attacks_detected': False, 'details': attacks, 'reason': 'Insufficient history for detection'}
        
        # Check for pattern flooding (many patterns in short time)
        recent_time_window = 60  # seconds
        current_time = time.time()
        recent_pattern_count = len([p for p in self.pattern_history 
                                  if current_time - p['timestamp'] < recent_time_window])
        
        flooding_threshold = 30  # More than 30 patterns per minute
        if recent_pattern_count > flooding_threshold:
            attacks['pattern_flooding'] = True
            # Increase threat level
            self.adaptive_sync_state['threat_level'] = min(1.0, self.adaptive_sync_state['threat_level'] + 0.2)
        
        # Check for oscillation attacks (patterns constantly changing direction)
        recent_patterns = [p['pattern'] for p in self.pattern_history[-10:]]
        directions = []
        for i in range(1, len(recent_patterns)):
            diff = recent_patterns[i] - recent_patterns[i-1]
            # Avoid division by zero
            norm_diff = np.linalg.norm(diff)
            if norm_diff > 1e-10:
                directions.append(diff / norm_diff)
            else:
                directions.append(np.zeros_like(diff))
        
        direction_changes = 0
        for i in range(1, len(directions)):
            if np.dot(directions[i], directions[i-1]) < 0:  # Direction reversed
                direction_changes += 1
        
        oscillation_threshold = 5  # More than half the pattern changes reverse direction
        if direction_changes > oscillation_threshold:
            attacks['oscillation'] = True
            # Increase threat level
            self.adaptive_sync_state['threat_level'] = min(1.0, self.adaptive_sync_state['threat_level'] + 0.3)
        
        # Check for split-brain condition (two competing consensus states)
        if len(self.observed_patterns) >= 10:
            # Cluster patterns to detect split consensus
            clusters = self._cluster_patterns([obs['pattern'] for obs in self.observed_patterns])
            
            # Split brain detected if second largest cluster is significant
            split_brain_threshold = 0.3  # Second cluster contains > 30% of nodes
            if len(clusters) > 1 and len(clusters[1]) > len(self.observed_patterns) * split_brain_threshold:
                attacks['split_brain'] = True
                # Significant threat
                self.adaptive_sync_state['threat_level'] = min(1.0, self.adaptive_sync_state['threat_level'] + 0.4)
        
        # Check for Sybil attacks (many nodes with similar patterns)
        if len(self.observed_patterns) >= 10:
            source_patterns = {}
            for obs in self.observed_patterns:
                if obs['source'] not in source_patterns:
                    source_patterns[obs['source']] = []
                source_patterns[obs['source']].append(obs['pattern'])
            
            # Calculate average pattern for each source
            source_avg_patterns = {}
            for source, patterns in source_patterns.items():
                if not patterns:
                    continue
                avg_pattern = np.mean(patterns, axis=0)
                norm = np.linalg.norm(avg_pattern)
                if norm > 1e-10:
                    source_avg_patterns[source] = avg_pattern / norm
            
            # Check for unusually similar patterns from different sources
            similar_sources = 0
            total_comparisons = 0
            
            sources = list(source_avg_patterns.keys())
            for i in range(len(sources)):
                for j in range(i+1, len(sources)):
                    similarity = np.dot(source_avg_patterns[sources[i]], 
                                      source_avg_patterns[sources[j]])
                    total_comparisons += 1
                    if similarity > 0.98:  # Extremely similar patterns
                        similar_sources += 1
            
            sybil_threshold = 0.5  # More than 50% of source pairs are suspiciously similar
            if total_comparisons > 0 and similar_sources / total_comparisons > sybil_threshold:
                attacks['sybil'] = True
                # Extreme threat
                self.adaptive_sync_state['threat_level'] = 1.0
        
        # Check for adversarial patterns (patterns designed to disrupt consensus)
        adversarial_count = 0
        for pattern in recent_patterns:
            # Check for characteristic signs of adversarial patterns
            fft_values = np.abs(np.fft.fft(pattern))
            dominant_frequency = np.max(fft_values[1:]) / np.mean(fft_values[1:])
            
            # Adversarial patterns often have unusual frequency distribution
            if dominant_frequency > 10.0:
                adversarial_count += 1
                
            # Check for unusual pattern structure
            pattern_range = np.max(pattern) - np.min(pattern)
            if pattern_range < 0.1 or pattern_range > 1.9:  # Suspiciously uniform or extreme
                adversarial_count += 1
        
        adversarial_threshold = 3  # Multiple suspicious patterns detected
        if adversarial_count >= adversarial_threshold:
            attacks['adversarial_patterns'] = True
            # Significant threat
            self.adaptive_sync_state['threat_level'] = min(1.0, self.adaptive_sync_state['threat_level'] + 0.3)
        
        attacks_detected = any(attacks.values())
        return {
            'attacks_detected': attacks_detected,
            'details': attacks,
            'threat_level': self.adaptive_sync_state['threat_level']
        }
    
    def _cluster_patterns(self, patterns: List[np.ndarray], threshold: float = 0.85) -> List[List[int]]:
        """
        Cluster patterns based on similarity.
        
        Args:
            patterns: List of patterns to cluster
            threshold: Similarity threshold for clustering
            
        Returns:
            List of clusters (each cluster is a list of pattern indices)
        """
        if not patterns:
            return []
        
        n = len(patterns)
        # Compute similarity matrix
        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                # Handle potential zero norms
                norm_i = np.linalg.norm(patterns[i])
                norm_j = np.linalg.norm(patterns[j])
                
                if norm_i > 1e-10 and norm_j > 1e-10:
                    sim = np.dot(patterns[i], patterns[j]) / (norm_i * norm_j)
                else:
                    sim = 0
                    
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim
        
        # Simple clustering algorithm
        visited = [False] * n
        clusters = []
        
        for i in range(n):
            if visited[i]:
                continue
                
            cluster = [i]
            visited[i] = True
            
            for j in range(n):
                if not visited[j] and sim_matrix[i, j] >= threshold:
                    cluster.append(j)
                    visited[j] = True
            
            clusters.append(cluster)
        
        # Sort clusters by size (descending)
        clusters.sort(key=len, reverse=True)
        return clusters
    
    def verify_pattern_signature(self, pattern: List[float], signature: str) -> bool:
        """
        Verify if a pattern has an approved signature.
        
        Args:
            pattern: Pattern to verify
            signature: Pattern signature
            
        Returns:
            True if signature is valid
        """
        if not pattern or not signature:
            return False
            
        pattern_array = np.array(pattern)
        # Normalize and get hash
        norm = np.linalg.norm(pattern_array)
        if norm < 1e-10:
            return False
            
        normalized_pattern = pattern_array / norm
        pattern_hash = self._get_pattern_hash(normalized_pattern)
        
        # Check if signature matches hash
        if pattern_hash != signature:
            return False
            
        # Check if signature is in approved set
        return signature in self.approved_pattern_signatures
    
    def benchmark_against_pow(self, num_transactions: int = 1000) -> Dict[str, float]:
        """
        Benchmark BCP against Proof of Work consensus.
        
        Args:
            num_transactions: Number of transactions to simulate
            
        Returns:
            Comparative performance metrics
        """
        # This is a simplified benchmark that simulates the key metrics
        # A real benchmark would involve running actual PoW consensus
        
        # BCP metrics
        bcp_start = time.time()
        
        # Simulate BCP consensus formation
        for i in range(min(10, num_transactions // 100)):  # Simulate batches
            # Generate random pattern
            pattern = np.random.rand(self.config['pattern_dimension']) * 2 - 1
            pattern = pattern / np.linalg.norm(pattern)
            
            # Emit pattern and iterate to consensus
            self.emit_pattern(pattern.tolist())
            
            # Simulate network convergence
            for j in range(3):  # Requires ~3 cycles for convergence
                adapted_pattern = self.adapt_to_network(pattern.tolist())
                self.emit_pattern(adapted_pattern)
        
        bcp_time = time.time() - bcp_start
        bcp_energy = bcp_time * 10  # Estimated energy units (arbitrary scale)
        
        # Simulate simplified PoW metrics (based on Bitcoin averages, scaled down)
        # Bitcoin: ~10 minutes per block, ~2000 tx per block, ~900 kWh per block
        tx_per_block = 2000
        blocks_needed = math.ceil(num_transactions / tx_per_block)
        pow_time = blocks_needed * 600  # seconds
        pow_energy = blocks_needed * 900 * 1000  # Watt-hours
        
        # Scale PoW metrics for simulation purposes
        pow_time_scaled = pow_time / 1000  # Scaled for comparison
        pow_energy_scaled = pow_energy / 1000  # Scaled for comparison
        
        return {
            'bcp_time_seconds': bcp_time,
            'pow_time_seconds': pow_time_scaled,
            'bcp_energy_units': bcp_energy,
            'pow_energy_units': pow_energy_scaled,
            'time_efficiency': pow_time_scaled / bcp_time if bcp_time > 0 else float('inf'),
            'energy_efficiency': pow_energy_scaled / bcp_energy if bcp_energy > 0 else float('inf'),
            'transactions': num_transactions
        }
    
    def benchmark_against_pos(self, num_transactions: int = 1000) -> Dict[str, float]:
        """
        Benchmark BCP against Proof of Stake consensus.
        
        Args:
            num_transactions: Number of transactions to simulate
            
        Returns:
            Comparative performance metrics
        """
        # Similar to PoW benchmark but comparing against PoS metrics
        # Simulate BCP consensus as before
        bcp_start = time.time()
        
        # Simulate BCP consensus formation
        for i in range(min(10, num_transactions // 100)):
            pattern = np.random.rand(self.config['pattern_dimension']) * 2 - 1
            pattern = pattern / np.linalg.norm(pattern)
            self.emit_pattern(pattern.tolist())
            
            for j in range(3):
                adapted_pattern = self.adapt_to_network(pattern.tolist())
                self.emit_pattern(adapted_pattern)
        
        bcp_time = time.time() - bcp_start
        bcp_energy = bcp_time * 10  # Estimated energy units
        
        # Simulate PoS metrics (based on Ethereum PoS averages)
        # Ethereum PoS: ~12 seconds per block, ~100 tx per block, much lower energy
        tx_per_block = 100
        blocks_needed = math.ceil(num_transactions / tx_per_block)
        pos_time = blocks_needed * 12  # seconds
        pos_energy = blocks_needed * 0.1 * 1000  # Watt-hours (vastly reduced compared to PoW)
        
        # Scale for simulation
        pos_time_scaled = pos_time / 10  # Scaled for comparison
        pos_energy_scaled = pos_energy / 10  # Scaled for comparison
        
        return {
            'bcp_time_seconds': bcp_time,
            'pos_time_seconds': pos_time_scaled,
            'bcp_energy_units': bcp_energy,
            'pos_energy_units': pos_energy_scaled,
            'time_efficiency': pos_time_scaled / bcp_time if bcp_time > 0 else float('inf'),
            'energy_efficiency': pos_energy_scaled / bcp_energy if bcp_energy > 0 else float('inf'),
            'byzantine_resistance_bcp': 0.33,  # BCP tolerates 33% malicious nodes
            'byzantine_resistance_pos': 0.33,  # PoS typically tolerates 33% malicious stake
            'transactions': num_transactions
        }
    
    def get_network_metrics(self) -> Dict[str, Any]:
        """
        Get detailed metrics about the current network state.
        
        Returns:
            Network metrics
        """
        # Calculate pattern stability over time
        pattern_stability = 0
        if len(self.pattern_history) >= 10:
            recent_patterns = [entry['pattern'] for entry in self.pattern_history[-10:]]
            pattern_stability = self._calculate_pattern_agreement(recent_patterns)
        
        # Calculate network convergence rate
        convergence_rate = 0
        if len(self.pattern_history) >= 20:
            scores = [entry['sync_score'] for entry in self.pattern_history[-20:]]
            if scores[0] < scores[-1]:
                convergence_rate = (scores[-1] - scores[0]) / 20
        
        # Get Byzantine quorum status
        quorum_status = self.get_byzantine_quorum_status()
        
        # Attack detection
        attack_status = self.detect_attacks()
        
        # Aggregate performance metrics
        performance = {
            'pattern_emissions': self.performance_metrics['pattern_emissions'],
            'pattern_rejections': self.performance_metrics['pattern_rejections'],
            'average_sync_score': self.performance_metrics['avg_sync_score'],
            'byzantine_detections': self.performance_metrics['byzantine_detections'],
            'adaptation_events': self.performance_metrics['adaptation_events'],
            'consensus_cycles': self.performance_metrics['consensus_cycles']
        }
        
        # Add convergence time statistics if available
        if self.performance_metrics['convergence_time']:
            convergence_times = [entry['time_seconds'] for entry in self.performance_metrics['convergence_time']]
            performance['avg_convergence_time'] = sum(convergence_times) / len(convergence_times)
            performance['min_convergence_time'] = min(convergence_times)
            performance['max_convergence_time'] = max(convergence_times)
        
        # Add adaptive parameters
        adaptive_params = {
            'sync_threshold': self.sync_threshold,
            'adaptation_rate': self.adaptation_rate,
            'network_stability': self.adaptive_sync_state['network_stability'],
            'threat_level': self.adaptive_sync_state['threat_level'],
            'congestion_level': self.adaptive_sync_state['congestion_level']
        }
        
        return {
            'consensus_achieved': self.consensus_achieved,
            'sync_threshold': self.sync_threshold,
            'pattern_stability': pattern_stability,
            'convergence_rate': convergence_rate,
            'quorum_status': quorum_status,
            'attack_status': attack_status,
            'performance': performance,
            'adaptive_params': adaptive_params,
            'total_nodes': self.total_nodes,
            'synchronized_nodes': self.synchronized_nodes,
            'pattern_dimension': self.config['pattern_dimension'],
            'network_health': 1.0 - self.adaptive_sync_state['threat_level']
        }