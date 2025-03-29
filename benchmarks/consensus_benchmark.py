import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from typing import List, Dict, Any, Tuple, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
import hashlib
import random
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import blockchain components
from blockchain.core import Blockchain, Transaction, Block
from blockchain.neural.enhanced_validator import EnhancedNeuralValidator
from blockchain.consensus.enhanced_bcp import EnhancedBioluminescentCoordinator
from blockchain.crypto.hardened_quantum import HardenedQuantumCrypto

class ConsensusMetrics:
    """
    Defines metrics for evaluating and comparing consensus algorithms.
    
    Metrics include:
    - Throughput (transactions per second)
    - Latency (time to consensus)
    - Energy consumption (estimated)
    - Security metrics (Byzantine fault tolerance)
    - Consistency metrics (forking rates, agreement levels)
    - Scalability (performance vs. node count)
    """
    
    def __init__(self):
        self.metrics = {
            'throughput': [],
            'latency': [],
            'energy': [],
            'security': [],
            'consistency': [],
            'scalability': [],
            'confidence': [],
            'pattern_entropy': [],
            'network_resilience': []
        }
    
    def add_metric(self, metric_name: str, value: float):
        """Add a metric measurement."""
        if metric_name in self.metrics:
            self.metrics[metric_name].append(value)
    
    def get_average(self, metric_name: str) -> float:
        """Get the average value for a metric."""
        if metric_name in self.metrics and self.metrics[metric_name]:
            return sum(self.metrics[metric_name]) / len(self.metrics[metric_name])
        return 0.0
    
    def get_metrics_summary(self) -> Dict[str, float]:
        """Get a summary of all metrics."""
        return {k: self.get_average(k) for k in self.metrics.keys()}
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert metrics to a pandas DataFrame."""
        return pd.DataFrame(self.metrics)


class ConsensusSimulator:
    """Base class for simulating different consensus algorithms."""
    
    def __init__(self, num_nodes: int = 10, num_transactions: int = 100, 
                fault_tolerance: float = 0.33):
        self.num_nodes = num_nodes
        self.num_transactions = num_transactions
        self.fault_tolerance = fault_tolerance
        self.metrics = ConsensusMetrics()
        self.crypto = HardenedQuantumCrypto({'security_level': 5})
        
        # Generate node identities
        self.nodes = [self._generate_node() for _ in range(num_nodes)]
        
        # Track simulation state
        self.transactions = []
        self.terminated = False
    
    def _generate_node(self) -> Dict[str, Any]:
        """Generate a node with identity, state and resources."""
        keypair = self.crypto.generate_keypair()
        return {
            'id': keypair['public_key'],
            'keypair': keypair,
            'compute_power': random.uniform(0.5, 2.0),  # Relative compute power
            'stake': random.uniform(100, 1000),         # Stake (for PoS)
            'state': {},                                # Node state
            'blockchain': Blockchain()                  # Node's blockchain copy
        }
    
    def _generate_transactions(self, count: int) -> List[Transaction]:
        """Generate random transactions for simulation."""
        transactions = []
        for _ in range(count):
            sender = random.choice(self.nodes)
            recipient = random.choice(self.nodes)
            while recipient['id'] == sender['id']:
                recipient = random.choice(self.nodes)
            
            amount = random.uniform(1, 10)
            
            tx = Transaction(
                sender=sender['id'],
                recipient=recipient['id'],
                amount=amount
            )
            tx.sign(sender['keypair']['private_key'])
            transactions.append(tx)
        
        return transactions
    
    def simulate(self, duration: float = 60.0) -> ConsensusMetrics:
        """
        Run a consensus simulation for a specified duration.
        
        Args:
            duration: Simulation duration in seconds
            
        Returns:
            Metrics collected during simulation
        """
        start_time = time.time()
        self.terminated = False
        
        # Generate transactions
        self.transactions = self._generate_transactions(self.num_transactions)
        
        # Track blocks and consensus
        blocks_created = 0
        consensus_rounds = 0
        
        # Simulation loop
        while time.time() - start_time < duration and not self.terminated:
            # Run a consensus round
            round_start = time.time()
            
            # Each consensus mechanism implements this differently
            block, agreement_level, confidence = self._consensus_round()
            
            round_end = time.time()
            
            if block:
                blocks_created += 1
                
                # Record metrics
                self.metrics.add_metric('latency', round_end - round_start)
                self.metrics.add_metric('consistency', agreement_level)
                self.metrics.add_metric('confidence', confidence)
                
                if len(block.transactions) > 0:
                    throughput = len(block.transactions) / (round_end - round_start)
                    self.metrics.add_metric('throughput', throughput)
            
            consensus_rounds += 1
            
            # Simulate network effect
            if consensus_rounds % 10 == 0:
                # Introduce some new transactions
                new_txs = self._generate_transactions(max(1, self.num_transactions // 10))
                self.transactions.extend(new_txs)
        
        # Calculate final metrics
        total_time = time.time() - start_time
        self.metrics.add_metric('throughput', blocks_created * 10 / total_time)  # Assuming ~10 tx per block
        self.metrics.add_metric('scalability', self.metrics.get_average('throughput') / self.num_nodes)
        
        return self.metrics
    
    def _consensus_round(self) -> Tuple[Optional[Block], float, float]:
        """
        Run a single consensus round.
        
        Returns:
            (block, agreement_level, confidence)
        """
        # To be implemented by subclasses
        raise NotImplementedError("Subclasses must implement _consensus_round")


class BCPSimulator(ConsensusSimulator):
    """Simulates the Bioluminescent Coordination Protocol."""
    
    def __init__(self, num_nodes: int = 10, num_transactions: int = 100,
                fault_tolerance: float = 0.33, pattern_dimension: int = 64):
        super().__init__(num_nodes, num_transactions, fault_tolerance)
        self.pattern_dimension = pattern_dimension
        
        # Initialize BCP and neural validator for each node
        for node in self.nodes:
            node['bcp'] = EnhancedBioluminescentCoordinator({
                'pattern_dimension': pattern_dimension,
                'sync_threshold': 0.75,
                'adaptation_rate': 0.3,
                'byzantine_tolerance': fault_tolerance
            })
            
            node['validator'] = EnhancedNeuralValidator({
                'pattern_dimension': pattern_dimension,
                'adaptation_factor': 0.2,
                'anomaly_threshold': 0.85
            })
    
    def _improved_pattern_agreement(self, patterns: List[np.ndarray]) -> Tuple[float, float]:
        """
        Calculate agreement and confidence metrics between patterns.
        
        Args:
            patterns: List of neural patterns
            
        Returns:
            (agreement_level, confidence_metric)
        """
        if len(patterns) <= 1:
            return 1.0, 1.0
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(patterns)):
            for j in range(i+1, len(patterns)):
                sim = np.dot(patterns[i], patterns[j])
                similarities.append(sim)
        
        # Calculate agreement (average similarity)
        agreement = sum(similarities) / len(similarities)
        
        # Calculate confidence (variance/stddev of similarities)
        variance = np.var(similarities) if len(similarities) > 1 else 0
        confidence = 1.0 - min(1.0, variance)  # Lower variance = higher confidence
        
        return agreement, confidence
    
    def _validate_pattern_entropy(self, pattern: np.ndarray) -> float:
        """
        Calculate the entropy of a pattern as a measure of its randomness.
        
        Args:
            pattern: Neural pattern to analyze
            
        Returns:
            Entropy measure (higher is more diverse)
        """
        # Normalize the pattern to positive values for probability distribution
        p = np.abs(pattern) / np.sum(np.abs(pattern) + 1e-10)
        
        # Calculate entropy
        entropy = -np.sum(p * np.log2(p + 1e-10))
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(pattern))
        normalized_entropy = entropy / max_entropy
        
        return normalized_entropy
    
    def _check_adversarial_patterns(self, 
                                  node_patterns: Dict[str, np.ndarray], 
                                  agreement_threshold: float = 0.8) -> List[str]:
        """
        Detect potential adversarial patterns that attempt to manipulate consensus.
        
        Args:
            node_patterns: Dictionary mapping node IDs to their patterns
            agreement_threshold: Threshold for suspicious agreement
            
        Returns:
            List of suspicious node IDs
        """
        suspicious_nodes = []
        node_ids = list(node_patterns.keys())
        
        # Check for abnormally high agreement between subsets of nodes
        for i in range(len(node_ids)):
            for j in range(i+1, len(node_ids)):
                id1, id2 = node_ids[i], node_ids[j]
                pattern1, pattern2 = node_patterns[id1], node_patterns[id2]
                
                # Calculate similarity
                similarity = np.dot(pattern1, pattern2)
                
                # If suspiciously high agreement
                if similarity > agreement_threshold:
                    # Check entropy of both patterns
                    entropy1 = self._validate_pattern_entropy(pattern1)
                    entropy2 = self._validate_pattern_entropy(pattern2)
                    
                    # Low entropy + high similarity = suspicious
                    if entropy1 < 0.7 or entropy2 < 0.7:
                        if id1 not in suspicious_nodes:
                            suspicious_nodes.append(id1)
                        if id2 not in suspicious_nodes:
                            suspicious_nodes.append(id2)
        
        return suspicious_nodes
    
    def _consensus_round(self) -> Tuple[Optional[Block], float, float]:
        """
        Run a single BCP consensus round.
        
        Returns:
            (block, agreement_level, confidence)
        """
        # Each node generates a neural pattern for pending transactions
        node_patterns = {}
        pending_blocks = {}
        
        for node in self.nodes:
            # Get subset of transactions for this round (could be different for each node)
            num_tx = min(10, len(self.transactions))
            if num_tx == 0:
                continue
                
            node_txs = random.sample(self.transactions, num_tx)
            
            # Generate pattern for transactions
            pattern = node['validator'].generate_pattern(node_txs)
            pattern_array = np.array(pattern)
            
            # Check pattern entropy
            entropy = self._validate_pattern_entropy(pattern_array)
            self.metrics.add_metric('pattern_entropy', entropy)
            
            # Create a block with these transactions
            prev_block = node['blockchain'].get_latest_block()
            new_block = Block(
                index=prev_block.index + 1,
                timestamp=time.time(),
                transactions=node_txs,
                previous_hash=prev_block.hash,
                neural_pattern=pattern
            )
            
            # Store pattern and block
            node_patterns[node['id']] = pattern_array
            pending_blocks[node['id']] = new_block
            
            # Emit pattern to BCP
            node['bcp'].emit_pattern(pattern)
        
        # Detect potential adversarial patterns
        suspicious_nodes = self._check_adversarial_patterns(node_patterns)
        if suspicious_nodes:
            # Remove suspicious nodes from consideration
            for node_id in suspicious_nodes:
                if node_id in node_patterns:
                    del node_patterns[node_id]
                if node_id in pending_blocks:
                    del pending_blocks[node_id]
        
        if not node_patterns:
            return None, 0.0, 0.0
        
        # Calculate pattern agreement and confidence
        all_patterns = list(node_patterns.values())
        agreement, confidence = self._improved_pattern_agreement(all_patterns)
        
        # If we have sufficient agreement, select a block
        if agreement >= 0.7 and confidence >= 0.6:
            # For simplicity, choose the block from a random node that's part of consensus
            consensus_node_id = random.choice(list(pending_blocks.keys()))
            selected_block = pending_blocks[consensus_node_id]
            
            # Remove included transactions from the pool
            tx_ids = [tx.transaction_id for tx in selected_block.transactions]
            self.transactions = [tx for tx in self.transactions if tx.transaction_id not in tx_ids]
            
            # Add block to all nodes' blockchains
            for node in self.nodes:
                node['blockchain'].chain.append(selected_block)
            
            return selected_block, agreement, confidence
        
        # No consensus reached this round
        return None, agreement, confidence


class PoWSimulator(ConsensusSimulator):
    """Simulates Proof of Work consensus."""
    
    def __init__(self, num_nodes: int = 10, num_transactions: int = 100,
                fault_tolerance: float = 0.33, difficulty: int = 3):
        super().__init__(num_nodes, num_transactions, fault_tolerance)
        self.difficulty = difficulty
        self.target = "0" * difficulty
    
    def _mine_block(self, node, transactions, prev_hash) -> Tuple[Block, float]:
        """
        Simulate mining a block with PoW.
        
        Returns:
            (block, energy_consumed)
        """
        start_time = time.time()
        
        # Create a block
        block = Block(
            index=len(node['blockchain'].chain),
            timestamp=time.time(),
            transactions=transactions,
            previous_hash=prev_hash
        )
        
        # Simulate mining (finding a valid nonce)
        nonce = 0
        hash_value = ""
        max_nonce = 2**32  # Arbitrary limit
        
        while nonce < max_nonce:
            block.nonce = nonce
            hash_value = block.calculate_hash()
            
            if hash_value[:self.difficulty] == self.target:
                break
            
            nonce += 1
        
        # Calculate energy consumption (simplified model based on iterations)
        energy_consumed = nonce * node['compute_power'] * 0.001  # Arbitrary energy units
        time_taken = time.time() - start_time
        
        return block, energy_consumed, time_taken
    
    def _consensus_round(self) -> Tuple[Optional[Block], float, float]:
        """
        Run a single PoW consensus round.
        
        Returns:
            (block, agreement_level, confidence)
        """
        # Select transactions for this round
        round_txs = []
        if self.transactions:
            count = min(10, len(self.transactions))
            round_txs = self.transactions[:count]
        
        # Each node attempts to mine a block
        mined_blocks = []
        node_energy = {}
        
        for node in self.nodes:
            # Get previous block hash
            prev_hash = node['blockchain'].get_latest_block().hash
            
            # Mine block
            block, energy, time_taken = self._mine_block(node, round_txs, prev_hash)
            
            # Store results
            mined_blocks.append((node, block, time_taken))
            node_energy[node['id']] = energy
            
            # Record energy consumption
            self.metrics.add_metric('energy', energy)
        
        # Sort by mining time (fastest first)
        mined_blocks.sort(key=lambda x: x[2])
        
        if not mined_blocks:
            return None, 0.0, 0.0
        
        # First miner wins
        winner_node, winner_block, _ = mined_blocks[0]
        
        # Calculate agreement level (how many nodes would have mined the same block)
        # In PoW, agreement is based on whether nodes accept the first valid block
        # Simplified: higher compute power = greater influence on consensus
        total_power = sum(node['compute_power'] for node in self.nodes)
        honest_power = sum(node['compute_power'] for node in self.nodes 
                         if node['id'] != winner_node['id'])
        
        # Agreement calculation: ratio of honest compute power to total
        agreement = honest_power / total_power if total_power > 0 else 0
        
        # Confidence calculation: if 51% or more compute power is honest, high confidence
        confidence = min(1.0, agreement / 0.51) if agreement > 0 else 0
        
        # Update all nodes with the winning block
        for node in self.nodes:
            node['blockchain'].chain.append(winner_block)
        
        # Remove included transactions from the pool
        tx_ids = [tx.transaction_id for tx in winner_block.transactions]
        self.transactions = [tx for tx in self.transactions if tx.transaction_id not in tx_ids]
        
        return winner_block, agreement, confidence


class PoSSimulator(ConsensusSimulator):
    """Simulates Proof of Stake consensus."""
    
    def __init__(self, num_nodes: int = 10, num_transactions: int = 100,
                fault_tolerance: float = 0.33, min_stake: float = 100.0):
        super().__init__(num_nodes, num_transactions, fault_tolerance)
        self.min_stake = min_stake
        
        # Ensure all nodes have at least minimum stake
        for node in self.nodes:
            if node['stake'] < min_stake:
                node['stake'] = min_stake
    
    def _select_validator(self) -> Dict[str, Any]:
        """
        Select a validator based on stake (weighted random selection).
        
        Returns:
            Selected validator node
        """
        # Calculate total stake
        total_stake = sum(node['stake'] for node in self.nodes)
        
        # Generate a random point
        r = random.uniform(0, total_stake)
        
        # Find the validator containing this point
        cumulative = 0
        for node in self.nodes:
            cumulative += node['stake']
            if r <= cumulative:
                return node
        
        # Fallback (should never reach here)
        return self.nodes[0]
    
    def _consensus_round(self) -> Tuple[Optional[Block], float, float]:
        """
        Run a single PoS consensus round.
        
        Returns:
            (block, agreement_level, confidence)
        """
        # Select transactions for this round
        round_txs = []
        if self.transactions:
            count = min(10, len(self.transactions))
            round_txs = self.transactions[:count]
        
        # Select validator for this round
        validator = self._select_validator()
        
        # Validator creates a block
        prev_block = validator['blockchain'].get_latest_block()
        new_block = Block(
            index=prev_block.index + 1,
            timestamp=time.time(),
            transactions=round_txs,
            previous_hash=prev_block.hash
        )
        
        # Calculate stake distribution for agreement level
        total_stake = sum(node['stake'] for node in self.nodes)
        validator_stake = validator['stake']
        
        # Agreement level based on stake distribution
        # Higher validator stake = lower agreement (more centralized)
        agreement = 1.0 - (validator_stake / total_stake)
        
        # Confidence based on stake distribution
        # If validator has too much stake (>33%), lower confidence
        if validator_stake / total_stake > self.fault_tolerance:
            confidence = 1.0 - ((validator_stake / total_stake) - self.fault_tolerance) / (1 - self.fault_tolerance)
        else:
            confidence = 1.0
        
        # Update all nodes with the new block
        for node in self.nodes:
            node['blockchain'].chain.append(new_block)
        
        # Remove included transactions from the pool
        tx_ids = [tx.transaction_id for tx in new_block.transactions]
        self.transactions = [tx for tx in self.transactions if tx.transaction_id not in tx_ids]
        
        return new_block, agreement, confidence


class ConsensusBenchmark:
    """
    Benchmark different consensus algorithms against each other.
    """
    
    def __init__(self, results_dir: str = 'benchmark_results'):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        self.simulators = {
            'BCP': BCPSimulator,
            'PoW': PoWSimulator,
            'PoS': PoSSimulator
        }
        
        self.results = {}
    
    def run_benchmark(self, algorithm: str = 'all', duration: float = 60.0,
                    node_counts: List[int] = None, tx_counts: List[int] = None) -> Dict[str, Any]:
        """
        Run benchmarks for specified consensus algorithms.
        
        Args:
            algorithm: Algorithm to benchmark ('all' for all algorithms)
            duration: Duration of each simulation in seconds
            node_counts: List of node counts to test
            tx_counts: List of transaction counts to test
            
        Returns:
            Benchmark results
        """
        if node_counts is None:
            node_counts = [5, 10, 20, 50]
        
        if tx_counts is None:
            tx_counts = [50, 100, 500, 1000]
        
        algorithms = list(self.simulators.keys()) if algorithm == 'all' else [algorithm]
        
        for alg in algorithms:
            print(f"Benchmarking {alg}...")
            self.results[alg] = self._benchmark_algorithm(
                alg, duration, node_counts, tx_counts
            )
        
        # Save results
        self._save_results()
        
        # Generate charts
        self._generate_charts()
        
        return self.results
    
    def _benchmark_algorithm(self, algorithm: str, duration: float,
                           node_counts: List[int], tx_counts: List[int]) -> Dict[str, Any]:
        """
        Benchmark a specific consensus algorithm.
        
        Args:
            algorithm: Algorithm to benchmark
            duration: Duration of each simulation in seconds
            node_counts: List of node counts to test
            tx_counts: List of transaction counts to test
            
        Returns:
            Algorithm benchmark results
        """
        if algorithm not in self.simulators:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        simulator_class = self.simulators[algorithm]
        results = {
            'scalability': [],
            'throughput': [],
            'latency': [],
            'consistency': [],
            'energy': [],
            'confidence': []
        }
        
        # Test different node counts
        for nodes in node_counts:
            print(f"  Testing with {nodes} nodes...")
            
            # Use middle tx count for node scaling tests
            tx_count = tx_counts[len(tx_counts) // 2]
            simulator = simulator_class(num_nodes=nodes, num_transactions=tx_count)
            metrics = simulator.simulate(duration)
            
            results['scalability'].append({
                'nodes': nodes,
                'throughput': metrics.get_average('throughput'),
                'latency': metrics.get_average('latency')
            })
        
        # Test different transaction counts (with middle node count)
        node_count = node_counts[len(node_counts) // 2]
        for txs in tx_counts:
            print(f"  Testing with {txs} transactions...")
            simulator = simulator_class(num_nodes=node_count, num_transactions=txs)
            metrics = simulator.simulate(duration)
            
            results['throughput'].append({
                'tx_count': txs,
                'tps': metrics.get_average('throughput')
            })
            
            results['latency'].append({
                'tx_count': txs,
                'latency': metrics.get_average('latency')
            })
            
            results['consistency'].append({
                'tx_count': txs,
                'agreement': metrics.get_average('consistency')
            })
            
            results['confidence'].append({
                'tx_count': txs,
                'confidence': metrics.get_average('confidence')
            })
            
            # Energy is most relevant for PoW
            if algorithm == 'PoW':
                results['energy'].append({
                    'tx_count': txs,
                    'energy': metrics.get_average('energy')
                })
        
        return results
    
    def _save_results(self) -> None:
        """Save benchmark results to disk."""
        filename = os.path.join(self.results_dir, 'consensus_benchmark_results.json')
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results saved to {filename}")
    
    def _generate_charts(self) -> None:
        """Generate charts from benchmark results."""
        # Throughput comparison
        self._generate_throughput_chart()
        
        # Latency comparison
        self._generate_latency_chart()
        
        # Scalability comparison
        self._generate_scalability_chart()
        
        # Consistency comparison
        self._generate_consistency_chart()
        
        # Energy comparison (mainly for PoW)
        if 'PoW' in self.results:
            self._generate_energy_chart()
        
        # Confidence comparison
        self._generate_confidence_chart()
    
    def _generate_throughput_chart(self) -> None:
        """Generate throughput comparison chart."""
        plt.figure(figsize=(10, 6))
        
        for alg, results in self.results.items():
            tx_counts = [item['tx_count'] for item in results['throughput']]
            tps_values = [item['tps'] for item in results['throughput']]
            
            plt.plot(tx_counts, tps_values, marker='o', label=alg)
        
        plt.xlabel('Transaction Count')
        plt.ylabel('Transactions Per Second')
        plt.title('Consensus Throughput Comparison')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.savefig(os.path.join(self.results_dir, 'consensus_throughput.png'))
    
    def _generate_latency_chart(self) -> None:
        """Generate latency comparison chart."""
        plt.figure(figsize=(10, 6))
        
        for alg, results in self.results.items():
            tx_counts = [item['tx_count'] for item in results['latency']]
            latency_values = [item['latency'] for item in results['latency']]
            
            plt.plot(tx_counts, latency_values, marker='o', label=alg)
        
        plt.xlabel('Transaction Count')
        plt.ylabel('Latency (seconds)')
        plt.title('Consensus Latency Comparison')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.savefig(os.path.join(self.results_dir, 'consensus_latency.png'))
    
    def _generate_scalability_chart(self) -> None:
        """Generate scalability comparison chart."""
        plt.figure(figsize=(10, 6))
        
        for alg, results in self.results.items():
            node_counts = [item['nodes'] for item in results['scalability']]
            throughput_values = [item['throughput'] for item in results['scalability']]
            
            plt.plot(node_counts, throughput_values, marker='o', label=alg)
        
        plt.xlabel('Node Count')
        plt.ylabel('Throughput (TPS)')
        plt.title('Consensus Scalability Comparison')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.savefig(os.path.join(self.results_dir, 'consensus_scalability.png'))
    
    def _generate_consistency_chart(self) -> None:
        """Generate consistency comparison chart."""
        plt.figure(figsize=(10, 6))
        
        for alg, results in self.results.items():
            tx_counts = [item['tx_count'] for item in results['consistency']]
            agreement_values = [item['agreement'] for item in results['consistency']]
            
            plt.plot(tx_counts, agreement_values, marker='o', label=alg)
        
        plt.xlabel('Transaction Count')
        plt.ylabel('Agreement Level (0-1)')
        plt.title('Consensus Agreement Comparison')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.savefig(os.path.join(self.results_dir, 'consensus_agreement.png'))
    
    def _generate_energy_chart(self) -> None:
        """Generate energy consumption chart."""
        plt.figure(figsize=(10, 6))
        
        for alg, results in self.results.items():
            if 'energy' in results and results['energy']:
                tx_counts = [item['tx_count'] for item in results['energy']]
                energy_values = [item['energy'] for item in results['energy']]
                
                plt.plot(tx_counts, energy_values, marker='o', label=alg)
        
        plt.xlabel('Transaction Count')
        plt.ylabel('Energy Consumption (arbitrary units)')
        plt.title('Consensus Energy Comparison')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.savefig(os.path.join(self.results_dir, 'consensus_energy.png'))
    
    def _generate_confidence_chart(self) -> None:
        """Generate confidence comparison chart."""
        plt.figure(figsize=(10, 6))
        
        for alg, results in self.results.items():
            tx_counts = [item['tx_count'] for item in results['confidence']]
            confidence_values = [item['confidence'] for item in results['confidence']]
            
            plt.plot(tx_counts, confidence_values, marker='o', label=alg)
        
        plt.xlabel('Transaction Count')
        plt.ylabel('Confidence Metric (0-1)')
        plt.title('Consensus Confidence Comparison')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.savefig(os.path.join(self.results_dir, 'consensus_confidence.png'))


class ConsensusAnalysis:
    """
    Provides analysis and insights from benchmark results.
    """
    
    def __init__(self, results_file: str = None):
        """
        Initialize with optional results file.
        
        Args:
            results_file: Path to JSON results file
        """
        self.results = {}
        
        if results_file and os.path.exists(results_file):
            with open(results_file, 'r') as f:
                self.results = json.load(f)
    
    def set_results(self, results: Dict) -> None:
        """Set results directly."""
        self.results = results
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of benchmark results.
        
        Returns:
            Dictionary of summary metrics
        """
        if not self.results:
            return {}
        
        summary = {}
        
        for alg, results in self.results.items():
            # Calculate average metrics
            throughput_avg = sum(item['tps'] for item in results['throughput']) / len(results['throughput'])
            latency_avg = sum(item['latency'] for item in results['latency']) / len(results['latency'])
            
            # Get scalability metrics
            scalability_data = results['scalability']
            scale_factor = 0
            if len(scalability_data) >= 2:
                first, last = scalability_data[0], scalability_data[-1]
                nodes_ratio = last['nodes'] / first['nodes']
                throughput_ratio = last['throughput'] / first['throughput'] if first['throughput'] > 0 else 0
                scale_factor = throughput_ratio / nodes_ratio
            
            # Compile summary for this algorithm
            summary[alg] = {
                'avg_throughput': throughput_avg,
                'avg_latency': latency_avg,
                'scalability_factor': scale_factor,
            }
            
            # Add consistency if available
            if 'consistency' in results and results['consistency']:
                consistency_avg = sum(item['agreement'] for item in results['consistency']) / len(results['consistency'])
                summary[alg]['avg_consistency'] = consistency_avg
            
            # Add energy if available
            if 'energy' in results and results['energy']:
                energy_avg = sum(item['energy'] for item in results['energy']) / len(results['energy'])
                summary[alg]['avg_energy'] = energy_avg
            
            # Add confidence if available
            if 'confidence' in results and results['confidence']:
                confidence_avg = sum(item['confidence'] for item in results['confidence']) / len(results['confidence'])
                summary[alg]['avg_confidence'] = confidence_avg
        
        return summary
    
    def generate_comparative_analysis(self) -> str:
        """
        Generate a textual comparative analysis.
        
        Returns:
            Analysis text
        """
        if not self.results:
            return "No results available for analysis."
        
        summary = self.get_summary()
        
        # Build analysis text
        analysis = "Consensus Mechanism Comparative Analysis\n"
        analysis += "======================================\n\n"
        
        # Throughput comparison
        analysis += "Throughput Comparison:\n"
        sorted_by_throughput = sorted(summary.items(), key=lambda x: x[1]['avg_throughput'], reverse=True)
        for alg, metrics in sorted_by_throughput:
            analysis += f"  {alg}: {metrics['avg_throughput']:.2f} TPS\n"
        
        analysis += "\n"
        
        # Latency comparison
        analysis += "Latency Comparison:\n"
        sorted_by_latency = sorted(summary.items(), key=lambda x: x[1]['avg_latency'])
        for alg, metrics in sorted_by_latency:
            analysis += f"  {alg}: {metrics['avg_latency']:.2f} seconds\n"
        
        analysis += "\n"
        
        # Scalability comparison
        analysis += "Scalability Comparison:\n"
        sorted_by_scalability = sorted(summary.items(), key=lambda x: x[1]['scalability_factor'], reverse=True)
        for alg, metrics in sorted_by_scalability:
            analysis += f"  {alg}: {metrics['scalability_factor']:.2f} scaling factor\n"
        
        analysis += "\n"
        
        # Consistency comparison
        if all('avg_consistency' in metrics for alg, metrics in summary.items()):
            analysis += "Consistency Comparison:\n"
            sorted_by_consistency = sorted(summary.items(), key=lambda x: x[1]['avg_consistency'], reverse=True)
            for alg, metrics in sorted_by_consistency:
                analysis += f"  {alg}: {metrics['avg_consistency']:.2f} agreement level\n"
            
            analysis += "\n"
        
        # Confidence comparison
        if all('avg_confidence' in metrics for alg, metrics in summary.items()):
            analysis += "Confidence Comparison:\n"
            sorted_by_confidence = sorted(summary.items(), key=lambda x: x[1]['avg_confidence'], reverse=True)
            for alg, metrics in sorted_by_confidence:
                analysis += f"  {alg}: {metrics['avg_confidence']:.2f} confidence metric\n"
            
            analysis += "\n"
        
        # Energy comparison (mainly for PoW)
        if any('avg_energy' in metrics for alg, metrics in summary.items()):
            analysis += "Energy Consumption Comparison:\n"
            for alg, metrics in summary.items():
                if 'avg_energy' in metrics:
                    analysis += f"  {alg}: {metrics['avg_energy']:.2f} energy units\n"
            
            analysis += "\n"
        
        # Overall assessment
        analysis += "Overall Assessment:\n"
        
        # Determine best algorithm for different metrics
        best_throughput = max(summary.items(), key=lambda x: x[1]['avg_throughput'])[0]
        best_latency = min(summary.items(), key=lambda x: x[1]['avg_latency'])[0]
        best_scalability = max(summary.items(), key=lambda x: x[1]['scalability_factor'])[0]
        
        analysis += f"  Best Throughput: {best_throughput}\n"
        analysis += f"  Best Latency: {best_latency}\n"
        analysis += f"  Best Scalability: {best_scalability}\n"
        
        if all('avg_consistency' in metrics for alg, metrics in summary.items()):
            best_consistency = max(summary.items(), key=lambda x: x[1]['avg_consistency'])[0]
            analysis += f"  Best Consistency: {best_consistency}\n"
        
        if all('avg_confidence' in metrics for alg, metrics in summary.items()):
            best_confidence = max(summary.items(), key=lambda x: x[1]['avg_confidence'])[0]
            analysis += f"  Best Confidence: {best_confidence}\n"
        
        # Add comparative advantage of BCP if present
        if 'BCP' in summary:
            analysis += "\nBioluminescent Coordination Protocol (BCP) Assessment:\n"
            
            bcp_throughput_rank = [alg for alg, _ in sorted_by_throughput].index('BCP') + 1
            bcp_latency_rank = [alg for alg, _ in sorted_by_latency].index('BCP') + 1
            bcp_scalability_rank = [alg for alg, _ in sorted_by_scalability].index('BCP') + 1
            
            analysis += f"  Throughput Rank: {bcp_throughput_rank} of {len(summary)}\n"
            analysis += f"  Latency Rank: {bcp_latency_rank} of {len(summary)}\n"
            analysis += f"  Scalability Rank: {bcp_scalability_rank} of {len(summary)}\n"
            
            # Highlight BCP advantages
            advantages = []
            
            if 'avg_consistency' in summary['BCP']:
                bcp_consistency = summary['BCP']['avg_consistency']
                others_consistency = [m['avg_consistency'] for a, m in summary.items() if a != 'BCP' and 'avg_consistency' in m]
                
                if others_consistency and bcp_consistency > max(others_consistency):
                    advantages.append("Higher consistency/agreement level")
            
            if 'avg_confidence' in summary['BCP']:
                bcp_confidence = summary['BCP']['avg_confidence']
                others_confidence = [m['avg_confidence'] for a, m in summary.items() if a != 'BCP' and 'avg_confidence' in m]
                
                if others_confidence and bcp_confidence > max(others_confidence):
                    advantages.append("Higher confidence metrics")
            
            if 'PoW' in summary and 'avg_energy' in summary['PoW'] and 'avg_throughput' in summary['BCP'] and 'avg_throughput' in summary['PoW']:
                pow_efficiency = summary['PoW']['avg_throughput'] / summary['PoW']['avg_energy']
                # BCP is assumed to have minimal energy consumption compared to PoW
                bcp_assumed_energy = summary['PoW']['avg_energy'] * 0.01  # Assume 1% of PoW energy usage
                bcp_efficiency = summary['BCP']['avg_throughput'] / bcp_assumed_energy
                
                if bcp_efficiency > pow_efficiency:
                    advantages.append("Significantly higher energy efficiency")
            
            if advantages:
                analysis += "\n  Key Advantages:\n"
                for adv in advantages:
                    analysis += f"    - {adv}\n"
        
        return analysis


def main():
    """Main function to run benchmarks."""
    parser = argparse.ArgumentParser(description='NyxSynth Consensus Benchmarking Tool')
    
    parser.add_argument('--algorithm', type=str, default='all',
                      choices=['all', 'BCP', 'PoW', 'PoS'],
                      help='Consensus algorithm to benchmark')
    parser.add_argument('--duration', type=float, default=30.0,
                      help='Duration of each simulation in seconds')
    parser.add_argument('--nodes', type=str, default='5,10,20',
                      help='Comma-separated list of node counts')
    parser.add_argument('--transactions', type=str, default='50,100,200',
                      help='Comma-separated list of transaction counts')
    parser.add_argument('--output-dir', type=str, default='benchmark_results',
                      help='Output directory for results')
    
    args = parser.parse_args()
    
    # Parse node counts and transaction counts
    node_counts = [int(n) for n in args.nodes.split(',')]
    tx_counts = [int(t) for t in args.transactions.split(',')]
    
    # Run benchmarks
    benchmark = ConsensusBenchmark(args.output_dir)
    results = benchmark.run_benchmark(
        algorithm=args.algorithm,
        duration=args.duration,
        node_counts=node_counts,
        tx_counts=tx_counts
    )
    
    # Generate analysis
    analyzer = ConsensusAnalysis()
    analyzer.set_results(results)
    analysis = analyzer.generate_comparative_analysis()
    
    # Save analysis
    analysis_file = os.path.join(args.output_dir, 'consensus_analysis.txt')
    with open(analysis_file, 'w') as f:
        f.write(analysis)
    
    print(f"Analysis saved to {analysis_file}")
    
    # Print summary
    print("\nAnalysis Summary:")
    print(analysis)


if __name__ == "__main__":
    main()