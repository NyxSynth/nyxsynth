import os
import sys
import time
import random
import json
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
import argparse
from typing import List, Dict, Any, Tuple, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import blockchain components
from blockchain.core import Blockchain, Transaction, Block
from blockchain.crypto.hardened_quantum import HardenedQuantumCrypto
from blockchain.neural.enhanced_validator import EnhancedNeuralValidator
from blockchain.consensus.enhanced_bcp import EnhancedBioluminescentCoordinator

class NyxSynthBenchmark:
    """
    Comprehensive benchmark suite for NyxSynth blockchain performance testing.
    
    Features:
    - Transaction throughput measurement
    - Scalability testing with various loads
    - Consensus mechanism efficiency analysis
    - Neural validation performance metrics
    - Cryptographic operations benchmarking
    - Network simulation with multiple nodes
    - Resource usage measurement
    - Comparative analysis with baseline
    """
    
    def __init__(self, config: Dict = None):
        """Initialize the benchmark suite."""
        self.config = config or {
            'test_duration': 60,  # seconds
            'transaction_sizes': [1, 10, 100, 1000, 10000],
            'transaction_batch_sizes': [1, 10, 50, 100, 500],
            'block_sizes': [1, 10, 50, 100, 500],
            'network_sizes': [1, 5, 10, 20, 50],
            'security_levels': [1, 3, 5],
            'output_dir': 'benchmark_results',
            'verbose': True
        }
        
        # Create output directory
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # Initialize components
        self.crypto = HardenedQuantumCrypto()
        self.validator = EnhancedNeuralValidator()
        self.coordinator = EnhancedBioluminescentCoordinator()
        self.blockchain = Blockchain()
        
        # Test wallets
        self.test_wallets = self._generate_test_wallets(10)
        
        # Results storage
        self.results = {
            'transaction_throughput': {},
            'block_creation': {},
            'consensus_formation': {},
            'neural_validation': {},
            'cryptographic_operations': {},
            'resource_usage': {},
            'scalability': {}
        }
    
    def _generate_test_wallets(self, count: int) -> List[Dict]:
        """Generate test wallets for benchmarking."""
        wallets = []
        for _ in range(count):
            wallet = self.crypto.generate_keypair()
            wallets.append(wallet)
        return wallets
    
    def _generate_test_transaction(self, size_bytes: int = 100) -> Transaction:
        """Generate a test transaction with specified data size."""
        sender = random.choice(self.test_wallets)
        recipient = random.choice(self.test_wallets)
        while recipient['public_key'] == sender['public_key']:
            recipient = random.choice(self.test_wallets)
        
        # Generate random amount
        amount = random.uniform(0.1, 100.0)
        
        # Create transaction
        tx = Transaction(
            sender=sender['public_key'],
            recipient=recipient['public_key'],
            amount=amount
        )
        
        # Sign transaction
        tx.sign(sender['private_key'])
        
        return tx
    
    def _generate_transaction_batch(self, batch_size: int) -> List[Transaction]:
        """Generate a batch of test transactions."""
        return [self._generate_test_transaction() for _ in range(batch_size)]
    
    def benchmark_transaction_throughput(self):
        """Benchmark transaction processing throughput."""
        print("\nBenchmarking Transaction Throughput...")
        results = {}
        
        for batch_size in self.config['transaction_batch_sizes']:
            print(f"Testing batch size: {batch_size}")
            
            # Clear pending transactions
            self.blockchain.pending_transactions = []
            
            # Generate transactions
            transactions = self._generate_transaction_batch(batch_size)
            
            # Measure time to add transactions
            start_time = time.time()
            for tx in transactions:
                self.blockchain.add_transaction(tx)
            end_time = time.time()
            
            elapsed = end_time - start_time
            throughput = batch_size / elapsed if elapsed > 0 else 0
            
            results[batch_size] = {
                'time_seconds': elapsed,
                'transactions_per_second': throughput,
                'total_transactions': batch_size
            }
            
            print(f"  Throughput: {throughput:.2f} tx/s")
        
        self.results['transaction_throughput'] = results
        return results
    
    def benchmark_block_creation(self):
        """Benchmark block creation and mining performance."""
        print("\nBenchmarking Block Creation...")
        results = {}
        
        for block_size in self.config['block_sizes']:
            print(f"Testing block size: {block_size}")
            
            # Clear pending transactions
            self.blockchain.pending_transactions = []
            
            # Add transactions
            transactions = self._generate_transaction_batch(block_size)
            for tx in transactions:
                self.blockchain.pending_transactions.append(tx)
            
            # Mine a block and measure time
            start_time = time.time()
            miner_wallet = self.test_wallets[0]
            block = self.blockchain.mine_pending_transactions(miner_wallet['public_key'])
            end_time = time.time()
            
            elapsed = end_time - start_time
            
            results[block_size] = {
                'time_seconds': elapsed,
                'blocks_per_second': 1.0 / elapsed if elapsed > 0 else 0,
                'transactions_per_block': block_size,
                'block_size_bytes': sys.getsizeof(json.dumps([tx.to_dict() for tx in transactions]))
            }
            
            print(f"  Time: {elapsed:.2f}s, Throughput: {block_size/elapsed:.2f} tx/s")
        
        self.results['block_creation'] = results
        return results
    
    def benchmark_consensus_formation(self):
        """Benchmark consensus formation with the BCP."""
        print("\nBenchmarking Consensus Formation...")
        results = {}
        
        # Create multiple coordinator instances to simulate a network
        network_sizes = self.config['network_sizes']
        for size in network_sizes:
            if size > 20:  # Skip very large networks for quick tests
                print(f"Skipping network size {size} (too large for quick test)")
                continue
                
            print(f"Testing network size: {size}")
            
            # Create coordinator instances
            coordinators = [EnhancedBioluminescentCoordinator() for _ in range(size)]
            
            # Generate initial patterns
            patterns = [np.random.rand(64) * 2 - 1 for _ in range(size)]
            patterns = [pattern / np.linalg.norm(pattern) for pattern in patterns]
            
            # Simulation state
            consensus_reached = False
            rounds = 0
            max_rounds = 100
            
            # Record start time
            start_time = time.time()
            
            # Run consensus simulation
            while not consensus_reached and rounds < max_rounds:
                rounds += 1
                
                # Each node emits a pattern
                for i in range(size):
                    # Emit current pattern
                    coordinators[i].emit_pattern(patterns[i].tolist())
                    
                    # Get consensus pattern
                    consensus = coordinators[i].get_consensus_pattern()
                    
                    # Adapt pattern towards consensus
                    patterns[i] = (1 - 0.2) * patterns[i] + 0.2 * consensus
                    patterns[i] = patterns[i] / np.linalg.norm(patterns[i])
                
                # Check if consensus is reached
                consensus_scores = []
                reference = coordinators[0].get_consensus_pattern()
                for i in range(size):
                    consensus_scores.append(np.dot(coordinators[i].get_consensus_pattern(), reference))
                
                avg_consensus = sum(consensus_scores) / len(consensus_scores)
                min_consensus = min(consensus_scores)
                
                # Check if consensus is sufficient
                if min_consensus > 0.85:
                    consensus_reached = True
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            results[size] = {
                'time_seconds': elapsed,
                'rounds': rounds,
                'consensus_reached': consensus_reached,
                'average_consensus': avg_consensus,
                'minimum_consensus': min_consensus,
                'time_per_round': elapsed / rounds if rounds > 0 else 0
            }
            
            print(f"  Time: {elapsed:.2f}s, Rounds: {rounds}, Consensus: {min_consensus:.4f}")
        
        self.results['consensus_formation'] = results
        return results
    
    def benchmark_neural_validation(self):
        """Benchmark neural validation performance."""
        print("\nBenchmarking Neural Validation...")
        results = {}
        
        for batch_size in self.config['transaction_batch_sizes']:
            print(f"Testing validation batch size: {batch_size}")
            
            # Generate transactions
            transactions = self._generate_transaction_batch(batch_size)
            
            # Benchmark pattern generation
            start_time = time.time()
            pattern = self.validator.generate_pattern(transactions)
            end_time = time.time()
            
            pattern_time = end_time - start_time
            
            # Benchmark transaction validation
            start_time = time.time()
            for tx in transactions:
                self.validator.validate_transaction(tx, self.blockchain)
            end_time = time.time()
            
            validation_time = end_time - start_time
            
            # Benchmark anomaly detection
            start_time = time.time()
            anomalies = self.validator.detect_anomalies(transactions)
            end_time = time.time()
            
            anomaly_detection_time = end_time - start_time
            
            results[batch_size] = {
                'pattern_generation_time': pattern_time,
                'transaction_validation_time': validation_time,
                'anomaly_detection_time': anomaly_detection_time,
                'total_time': pattern_time + validation_time + anomaly_detection_time,
                'transactions_per_second': batch_size / (validation_time if validation_time > 0 else 1),
                'pattern_dimension': len(pattern)
            }
            
            print(f"  Validation rate: {batch_size/validation_time:.2f} tx/s")
        
        self.results['neural_validation'] = results
        return results
    
    def benchmark_cryptographic_operations(self):
        """Benchmark cryptographic operations performance."""
        print("\nBenchmarking Cryptographic Operations...")
        results = {}
        
        # Test different security levels
        for level in self.config['security_levels']:
            print(f"Testing security level: {level}")
            
            # Configure crypto with security level
            crypto = HardenedQuantumCrypto({
                'security_level': level
            })
            
            # Test key generation
            start_time = time.time()
            keypair = crypto.generate_keypair()
            key_gen_time = time.time() - start_time
            
            # Test data for operations
            test_data = os.urandom(1024)  # 1 KB
            
            # Test signing
            start_time = time.time()
            signature = crypto.sign(test_data, keypair['private_key'])
            signing_time = time.time() - start_time
            
            # Test verification
            start_time = time.time()
            valid = crypto.verify(test_data, signature, keypair['public_key'])
            verification_time = time.time() - start_time
            
            # Test encryption
            start_time = time.time()
            encrypted = crypto.encrypt(test_data, keypair['public_key'])
            encryption_time = time.time() - start_time
            
            # Test decryption
            start_time = time.time()
            decrypted = crypto.decrypt(encrypted, keypair['private_key'])
            decryption_time = time.time() - start_time
            
            results[level] = {
                'key_generation_time': key_gen_time,
                'signing_time': signing_time,
                'verification_time': verification_time,
                'encryption_time': encryption_time,
                'decryption_time': decryption_time,
                'total_time': key_gen_time + signing_time + verification_time + encryption_time + decryption_time,
                'valid_signature': valid,
                'decryption_correct': decrypted == test_data
            }
            
            print(f"  Key Gen: {key_gen_time:.4f}s, Sign: {signing_time:.4f}s, Verify: {verification_time:.4f}s")
        
        self.results['cryptographic_operations'] = results
        return results
    
    def benchmark_resource_usage(self):
        """Benchmark resource usage (CPU, memory)."""
        print("\nBenchmarking Resource Usage...")
        results = {}
        
        try:
            import psutil
            process = psutil.Process(os.getpid())
            
            def measure_resources():
                return {
                    'cpu_percent': process.cpu_percent(),
                    'memory_usage_mb': process.memory_info().rss / (1024 * 1024),
                    'threads': process.num_threads()
                }
            
            # Baseline usage
            baseline = measure_resources()
            
            # Test transaction processing resource usage
            for batch_size in [10, 100, 1000]:
                print(f"Measuring resource usage with {batch_size} transactions...")
                
                # Generate transactions
                transactions = self._generate_transaction_batch(batch_size)
                
                # Start monitoring
                start_resources = measure_resources()
                start_time = time.time()
                
                # Process transactions
                for tx in transactions:
                    self.blockchain.add_transaction(tx)
                
                # Mine block
                self.blockchain.mine_pending_transactions(self.test_wallets[0]['public_key'])
                
                # End monitoring
                end_resources = measure_resources()
                end_time = time.time()
                
                results[batch_size] = {
                    'cpu_percent': end_resources['cpu_percent'],
                    'memory_usage_mb': end_resources['memory_usage_mb'],
                    'memory_change_mb': end_resources['memory_usage_mb'] - start_resources['memory_usage_mb'],
                    'processing_time': end_time - start_time,
                    'transactions_per_second': batch_size / (end_time - start_time) if (end_time - start_time) > 0 else 0
                }
                
                print(f"  CPU: {end_resources['cpu_percent']}%, Memory: {end_resources['memory_usage_mb']:.2f} MB")
                
        except ImportError:
            print("Warning: psutil module not found. Resource usage benchmarking skipped.")
            results['error'] = "psutil module not found"
        
        self.results['resource_usage'] = results
        return results
    
    def benchmark_scalability(self):
        """Benchmark scalability with increasing load."""
        print("\nBenchmarking Scalability...")
        results = {}
        
        # Test increasing transaction volume
        volumes = [10, 100, 1000, 5000, 10000]
        for volume in volumes:
            if volume > 5000:  # Skip very large volumes for quick tests
                print(f"Skipping volume {volume} (too large for quick test)")
                results[volume] = {'skipped': True}
                continue
                
            print(f"Testing transaction volume: {volume}")
            
            # Generate transactions
            batch_size = min(volume, 100)
            batches = volume // batch_size
            
            start_time = time.time()
            
            # Process transactions in batches
            for _ in range(batches):
                transactions = self._generate_transaction_batch(batch_size)
                for tx in transactions:
                    self.blockchain.add_transaction(tx)
                
                # Mine block after each batch
                self.blockchain.mine_pending_transactions(self.test_wallets[0]['public_key'])
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            results[volume] = {
                'total_time_seconds': elapsed,
                'transactions_per_second': volume / elapsed if elapsed > 0 else 0,
                'batches': batches,
                'batch_size': batch_size
            }
            
            print(f"  Time: {elapsed:.2f}s, Throughput: {volume/elapsed:.2f} tx/s")
        
        self.results['scalability'] = results
        return results
    
    def run_all_benchmarks(self):
        """Run all benchmarks and collect results."""
        print("Running NyxSynth Comprehensive Benchmarks")
        print("========================================")
        
        # Record global start time
        global_start = time.time()
        
        # Run individual benchmarks
        self.benchmark_transaction_throughput()
        self.benchmark_block_creation()
        self.benchmark_consensus_formation()
        self.benchmark_neural_validation()
        self.benchmark_cryptographic_operations()
        self.benchmark_resource_usage()
        self.benchmark_scalability()
        
        # Record global end time
        global_end = time.time()
        global_elapsed = global_end - global_start
        
        print("\nBenchmarks Complete!")
        print(f"Total time: {global_elapsed:.2f} seconds")
        
        # Save results
        self.save_results()
        
        # Generate charts
        self.generate_charts()
        
        return self.results
    
    def save_results(self):
        """Save benchmark results to file."""
        timestamp = int(time.time())
        filename = os.path.join(self.config['output_dir'], f'benchmark_results_{timestamp}.json')
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to {filename}")
    
    def generate_charts(self):
        """Generate charts from benchmark results."""
        print("\nGenerating benchmark charts...")
        
        try:
            # Transaction throughput chart
            self._create_throughput_chart()
            
            # Consensus formation chart
            self._create_consensus_chart()
            
            # Neural validation chart
            self._create_validation_chart()
            
            # Cryptographic operations chart
            self._create_crypto_chart()
            
            # Scalability chart
            self._create_scalability_chart()
            
            print("Charts saved to benchmark_results directory")
            
        except Exception as e:
            print(f"Error generating charts: {e}")
    
    def _create_throughput_chart(self):
        """Create transaction throughput chart."""
        if not self.results.get('transaction_throughput'):
            return
            
        plt.figure(figsize=(10, 6))
        
        sizes = []
        throughputs = []
        
        for size, data in sorted(self.results['transaction_throughput'].items()):
            sizes.append(size)
            throughputs.append(data['transactions_per_second'])
        
        plt.bar(range(len(sizes)), throughputs, color='royalblue')
        plt.xticks(range(len(sizes)), sizes)
        plt.xlabel('Batch Size')
        plt.ylabel('Transactions Per Second')
        plt.title('NyxSynth Transaction Throughput')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save chart
        plt.savefig(os.path.join(self.config['output_dir'], 'transaction_throughput.png'))
    
    def _create_consensus_chart(self):
        """Create consensus formation chart."""
        if not self.results.get('consensus_formation'):
            return
            
        plt.figure(figsize=(10, 6))
        
        sizes = []
        rounds = []
        min_consensus = []
        
        for size, data in sorted(self.results['consensus_formation'].items()):
            sizes.append(size)
            rounds.append(data['rounds'])
            min_consensus.append(data['minimum_consensus'])
        
        # Plot rounds
        ax1 = plt.subplot(111)
        bars = ax1.bar(range(len(sizes)), rounds, color='royalblue', alpha=0.7, label='Rounds')
        ax1.set_xlabel('Network Size')
        ax1.set_ylabel('Rounds to Consensus', color='royalblue')
        ax1.tick_params(axis='y', labelcolor='royalblue')
        ax1.set_xticks(range(len(sizes)))
        ax1.set_xticklabels(sizes)
        
        # Plot consensus level
        ax2 = ax1.twinx()
        line = ax2.plot(range(len(sizes)), min_consensus, 'r-', marker='o', label='Consensus Level')
        ax2.set_ylabel('Minimum Consensus Level', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Add a legend
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')
        
        plt.title('NyxSynth Consensus Formation')
        plt.grid(False)
        
        # Save chart
        plt.savefig(os.path.join(self.config['output_dir'], 'consensus_formation.png'))
    
    def _create_validation_chart(self):
        """Create neural validation chart."""
        if not self.results.get('neural_validation'):
            return
            
        plt.figure(figsize=(10, 6))
        
        sizes = []
        pattern_times = []
        validation_times = []
        anomaly_times = []
        throughputs = []
        
        for size, data in sorted(self.results['neural_validation'].items()):
            sizes.append(size)
            pattern_times.append(data['pattern_generation_time'])
            validation_times.append(data['transaction_validation_time'])
            anomaly_times.append(data['anomaly_detection_time'])
            throughputs.append(data['transactions_per_second'])
        
        # Create stacked bar chart for time components
        bottom_data = np.zeros(len(sizes))
        
        plt.bar(range(len(sizes)), pattern_times, label='Pattern Generation', color='royalblue', bottom=bottom_data)
        bottom_data = np.array(pattern_times)
        
        plt.bar(range(len(sizes)), validation_times, label='Transaction Validation', color='lightgreen', bottom=bottom_data)
        bottom_data = bottom_data + np.array(validation_times)
        
        plt.bar(range(len(sizes)), anomaly_times, label='Anomaly Detection', color='salmon', bottom=bottom_data)
        
        plt.xlabel('Batch Size')
        plt.ylabel('Time (seconds)')
        plt.title('NyxSynth Neural Validation Performance')
        plt.xticks(range(len(sizes)), sizes)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save chart
        plt.savefig(os.path.join(self.config['output_dir'], 'neural_validation.png'))
        
        # Create a second chart for throughput
        plt.figure(figsize=(10, 6))
        plt.plot(sizes, throughputs, 'ro-', linewidth=2)
        plt.xlabel('Batch Size')
        plt.ylabel('Transactions Per Second')
        plt.title('NyxSynth Neural Validation Throughput')
        plt.grid(linestyle='--', alpha=0.7)
        
        # Save chart
        plt.savefig(os.path.join(self.config['output_dir'], 'neural_validation_throughput.png'))
    
    def _create_crypto_chart(self):
        """Create cryptographic operations chart."""
        if not self.results.get('cryptographic_operations'):
            return
            
        plt.figure(figsize=(12, 8))
        
        levels = []
        key_gen_times = []
        signing_times = []
        verification_times = []
        encryption_times = []
        decryption_times = []
        
        for level, data in sorted(self.results['cryptographic_operations'].items()):
            levels.append(f"Level {level}")
            key_gen_times.append(data['key_generation_time'])
            signing_times.append(data['signing_time'])
            verification_times.append(data['verification_time'])
            encryption_times.append(data['encryption_time'])
            decryption_times.append(data['decryption_time'])
        
        # Set width of bars
        bar_width = 0.15
        index = np.arange(len(levels))
        
        # Create grouped bar chart
        plt.bar(index - 2*bar_width, key_gen_times, bar_width, label='Key Generation', color='royalblue')
        plt.bar(index - bar_width, signing_times, bar_width, label='Signing', color='lightgreen')
        plt.bar(index, verification_times, bar_width, label='Verification', color='salmon')
        plt.bar(index + bar_width, encryption_times, bar_width, label='Encryption', color='purple')
        plt.bar(index + 2*bar_width, decryption_times, bar_width, label='Decryption', color='orange')
        
        plt.xlabel('Security Level')
        plt.ylabel('Time (seconds)')
        plt.title('NyxSynth Cryptographic Operations Performance')
        plt.xticks(index, levels)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save chart
        plt.savefig(os.path.join(self.config['output_dir'], 'cryptographic_operations.png'))
    
    def _create_scalability_chart(self):
        """Create scalability chart."""
        if not self.results.get('scalability'):
            return
            
        plt.figure(figsize=(10, 6))
        
        volumes = []
        throughputs = []
        times = []
        
        for volume, data in sorted(self.results['scalability'].items()):
            if data.get('skipped'):
                continue
                
            volumes.append(volume)
            throughputs.append(data['transactions_per_second'])
            times.append(data['total_time_seconds'])
        
        # Plot throughput
        ax1 = plt.subplot(111)
        line1 = ax1.plot(volumes, throughputs, 'b-', marker='o', label='Throughput')
        ax1.set_xlabel('Transaction Volume')
        ax1.set_ylabel('Transactions Per Second', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Plot total time
        ax2 = ax1.twinx()
        line2 = ax2.plot(volumes, times, 'r-', marker='s', label='Total Time')
        ax2.set_ylabel('Total Time (seconds)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Add a legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.title('NyxSynth Scalability Performance')
        plt.grid(False)
        
        # Save chart
        plt.savefig(os.path.join(self.config['output_dir'], 'scalability.png'))


def main():
    """Main function to run benchmarks from command line."""
    parser = argparse.ArgumentParser(description='Run NyxSynth performance benchmarks')
    parser.add_argument('--output', type=str, default='benchmark_results', help='Output directory for results')
    parser.add_argument('--duration', type=int, default=60, help='Test duration in seconds for each benchmark')
    parser.add_argument('--quick', action='store_true', help='Run a quick benchmark with reduced test sizes')
    parser.add_argument('--crypto', action='store_true', help='Run only cryptographic benchmarks')
    parser.add_argument('--consensus', action='store_true', help='Run only consensus benchmarks')
    parser.add_argument('--neural', action='store_true', help='Run only neural validation benchmarks')
    parser.add_argument('--throughput', action='store_true', help='Run only throughput benchmarks')
    parser.add_argument('--scalability', action='store_true', help='Run only scalability benchmarks')
    
    args = parser.parse_args()
    
    # Configure benchmark
    config = {
        'output_dir': args.output,
        'test_duration': args.duration,
        'verbose': True
    }
    
    if args.quick:
        # Reduce test sizes for quick benchmarks
        config.update({
            'transaction_sizes': [1, 10, 100],
            'transaction_batch_sizes': [1, 10, 50],
            'block_sizes': [1, 10, 50],
            'network_sizes': [1, 3, 5],
            'security_levels': [1, 5]
        })
    
    benchmark = NyxSynthBenchmark(config)
    
    # Run selected benchmarks
    if args.crypto:
        benchmark.benchmark_cryptographic_operations()
    elif args.consensus:
        benchmark.benchmark_consensus_formation()
    elif args.neural:
        benchmark.benchmark_neural_validation()
    elif args.throughput:
        benchmark.benchmark_transaction_throughput()
        benchmark.benchmark_block_creation()
    elif args.scalability:
        benchmark.benchmark_scalability()
    else:
        # Run all benchmarks
        benchmark.run_all_benchmarks()
    
    # Save results and generate charts
    benchmark.save_results()
    benchmark.generate_charts()


if __name__ == "__main__":
    main()
