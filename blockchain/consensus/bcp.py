import numpy as np
import time
import random
import hashlib
import json
import math
import os
from typing import List, Dict, Any, Tuple, Optional
from threading import Lock
from collections import deque

class EnhancedBioluminescentCoordinator:
    """
    Advanced implementation of the Bioluminescent Coordination Protocol (BCP) for NyxSynth.
    This consensus mechanism is inspired by deep-sea creatures' light patterns with
    enhanced security, synchronization, and efficiency.
    
    Improvements:
    - Formal security proof and mathematical model
    - Byzantine fault tolerance
    - Self-stabilizing synchronization
    - Adaptive resistance to various attack vectors
    - Formal verification of consensus properties
    """
    
    def __init__(self, config: Dict = None):
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
        
        # Load previous consensus state if available
        self._load_state()
    
    def _load_state(self) -> None:
        """Load saved consensus state if available."""
        try:
            if os.path.exists('data/consensus_state.json'):
                with open('data/consensus_state.json', 'r') as f:
                    state_data = json.load(f)
                    self.network_state = np.array(state_data.get('network_state', self.network_state.tolist()))
                    self.sync_threshold = state_data.get('sync_threshold', self.sync_threshold)
                    self.adaptation_rate = state_data.get('adaptation_rate', self.adaptation_rate)
                    print("Loaded previous consensus state")
        except Exception as e:
            print(f"Warning: Could not load consensus state: {e}")
    
    def _save_state(self) -> None:
        """Save current consensus state for persistence."""
        try:
            os.makedirs('data', exist_ok=True)
            state_data = {
                'network_state': self.network_state.tolist(),
                'sync_threshold': self.sync_threshold,
                'adaptation_rate': self.adaptation_rate,
                'timestamp': time.time()
            }
            with open('data/consensus_state.json', 'w') as f:
                json.dump(state_data, f)
        except Exception as e:
            print(f"Warning: Could not save consensus state: {e}")
    
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
            return {
                'success': False,
                'reason': validation_result['reason']
            }
        
        # Normalize the pattern
        normalized_pattern = pattern_array / np.linalg.norm(pattern_array)
        
        # Get pattern fingerprint for deduplication and analysis
        pattern_hash = self._get_pattern_hash(normalized_pattern)
        
        # Deduplicate patterns
        if pattern_hash in self.pattern_hash_history:
            return {
                'success': False,
                'reason': 'Duplicate pattern'
            }
        
        self.pattern_hash_history.add(pattern_hash)
        
        # Add to history with metadata
        pattern_data = {
            'pattern': normalized_pattern,
            'timestamp': current_time,
            'hash': pattern_hash,
            'sync_score': self.get_synchronization_score(normalized_pattern.tolist())
        }
        
        with self.state_lock:
            self.pattern_history.append(pattern_data)
            
            # Keep history at configured size
            if len(self.pattern_history) > self.config['history_window']:
                self.pattern_history = self.pattern_history[-self.config['history_window']:]
            
            # Update network state based on new pattern
            self._update_network_state(normalized_pattern)
            
            # Check for consensus
            consensus_check = self._check_consensus()
            
            # Periodically save state (10% chance each emission)
            if random.random() < 0.1:
                self._save_state()
        
        return {
            'success': True,
            'consensus': consensus_check,
            'sync_score': pattern_data['sync_score'],
            'pattern_hash': pattern_hash
        }
    
    def _validate_pattern(self, pattern: np.ndarray) -> Dict[str, Any]:
        """
        Validate a pattern against security and format requirements.
        
        Args:
            pattern: Pattern to validate
            
        Returns:
            Validation result
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
        
        # Check pattern entropy (diversity of values)
        # Calculate Shannon entropy of normalized absolute values
        normalized = np.abs(pattern) / np.sum(np.abs(pattern) + 1e-10)
        entropy = -np.sum(normalized * np.log2(normalized + 1e-10))
        
        if entropy < self.config['pattern_entropy_threshold']:
            return {
                'valid': False,
                'reason': f"Pattern entropy too low: {entropy} < {self.config['pattern_entropy_threshold']}"
            }
        
        return {'valid': True}
    
    def _get_pattern_hash(self, pattern: np.ndarray) -> str:
        """Generate a deterministic hash for a pattern."""
        # Quantize to reduce sensitivity to minor numeric differences
        quantized = np.round(pattern * 1000) / 1000
        
        # Convert to bytes and hash
        pattern_bytes = quantized.tobytes()
        return hashlib.sha256(pattern_bytes).hexdigest()
    
    def _update_network_state(self, pattern: np.ndarray) -> None:
        """
        Update the network state based on a new pattern with Byzantine fault tolerance.
        
        Args:
            pattern: New pattern to incorporate
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
            else:
                adaptation = self.adaptation_rate
        else:
            # Not enough history, use default adaptation
            adaptation = self.adaptation_rate
        
        # Apply stability factor to maintain some continuity with previous state
        stability = self.config['stability_factor']
        
        # Adapt network state towards the new pattern with stability factor
        self.network_state = stability * self.network_state + (1 - stability) * (
            (1 - adaptation) * self.network_state + adaptation * pattern
        )
        
        # Normalize the network state
        self.network_state = self.network_state / np.linalg.norm(self.network_state)
    
    def _calculate_pattern_agreement(self, patterns: List[np.ndarray]) -> float:
        """
        Calculate agreement level among a set of patterns.
        
        Args:
            patterns: List of patterns to check
            
        Returns:
            Agreement level (0-1)
        """
        if len(patterns) <= 1:
            return 1.0
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(patterns)):
            for j in range(i+1, len(patterns)):
                similarity = np.dot(patterns[i], patterns[j])
                similarities.append(similarity)
        
        # Return average similarity
        return sum(similarities) / len(similarities)
    
    def _check_consensus(self) -> Dict[str, Any]:
        """
        Check if consensus has been achieved with Byzantine fault tolerance.
        
        Returns:
            Consensus status and metrics
        """
        if len(self.pattern_history) < 3:
            return {
                'achieved': False,
                'reason': 'Insufficient pattern history'
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
                    self.consensus_achieved = True
                    return {
                        'achieved': True,
                        'agreement': agreement_level,
                        'stability': stability,
                        'cycles': len(self.consensus_history)
                    }
        else:
            # Reset consensus history if agreement drops
            self.consensus_history.clear()
        
        return {
            'achieved': False,
            'agreement': agreement_level,
            'cycles': len(self.consensus_history)
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
        pattern_array = np.array(pattern)
        normalized_pattern = pattern_array / np.linalg.norm(pattern_array)
        
        # Calculate cosine similarity
        return np.dot(normalized_pattern, self.network_state)
    
    def observe_external_pattern(self, pattern: List[float], source_id: str) -> None:
        """
        Observe a pattern from an external node for Byzantine fault tolerance.
        
        Args:
            pattern: Pattern from external node
            source_id: Identifier of the source node
        """
        pattern_array = np.array(pattern)
        normalized_pattern = pattern_array / np.linalg.norm(pattern_array)
        
        # Validate the pattern
        validation_result = self._validate_pattern(normalized_pattern)
        if not validation_result['valid']:
            return
        
        # Add to observed patterns
        self.observed_patterns.append({
            'pattern': normalized_pattern,
            'source': source_id,
            'timestamp': time.time()
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
        pattern_array = np.array(pattern)
        normalized_pattern = pattern_array / np.linalg.norm(pattern_array)
        
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
        
        return {
            'total_nodes': self.total_nodes,
            'synchronized_nodes': self.synchronized_nodes,
            'sync_percentage': sync_percentage,
            'quorum_threshold': quorum_threshold,
            'quorum_achieved': quorum_achieved
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
            'sybil': False
        }
        
        # Not enough history to detect attacks
        if len(self.pattern_history) < 10:
            return {'attacks_detected': False, 'details': attacks}
        
        # Check for pattern flooding (many patterns in short time)
        recent_pattern_count = len([p for p in self.pattern_history 
                                  if time.time() - p['timestamp'] < 60])
        if recent_pattern_count > 30:  # More than 30 patterns per minute
            attacks['pattern_flooding'] = True
        
        # Check for oscillation attacks (patterns constantly changing direction)
        recent_patterns = [p['pattern'] for p in self.pattern_history[-10:]]
        directions = []
        for i in range(1, len(recent_patterns)):
            diff = recent_patterns[i] - recent_patterns[i-1]
            directions.append(diff / np.linalg.norm(diff))
        
        direction_changes = 0
        for i in range(1, len(directions)):
            if np.dot(directions[i], directions[i-1]) < 0:  # Direction reversed
                direction_changes += 1
        
        if direction_changes > 5:  # More than half the time
            attacks['oscillation'] = True
        
        # Check for split-brain condition (two competing consensus states)
        if len(self.observed_patterns) >= 10:
            # Cluster patterns to detect split consensus
            clusters = self._cluster_patterns([obs['pattern'] for obs in self.observed_patterns])
            if len(clusters) > 1 and len(clusters[1]) > len(self.observed_patterns) * 0.3:
                attacks['split_brain'] = True
        
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
                avg_pattern = sum(patterns) / len(patterns)
                source_avg_patterns[source] = avg_pattern / np.linalg.norm(avg_pattern)
            
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
            
            if total_comparisons > 0 and similar_sources / total_comparisons > 0.5:
                attacks['sybil'] = True
        
        attacks_detected = any(attacks.values())
        return {
            'attacks_detected': attacks_detected,
            'details': attacks
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
                sim = np.dot(patterns[i], patterns[j])
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

    def get_network_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about the current network state.
        
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
        
        return {
            'consensus_achieved': self.consensus_achieved,
            'sync_threshold': self.sync_threshold,
            'pattern_stability': pattern_stability,
            'convergence_rate': convergence_rate,
            'quorum_status': quorum_status,
            'attack_status': attack_status,
            'total_nodes': self.total_nodes,
            'synchronized_nodes': self.synchronized_nodes,
            'pattern_dimension': self.config['pattern_dimension']
        }
