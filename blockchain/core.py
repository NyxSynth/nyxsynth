import hashlib
import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any
import os
import numpy as np
from .neural.validator import NeuralValidator
from .consensus.bcp import BioluminescentCoordinator
from .crypto.quantum import QuantumResistantCrypto

@dataclass
class Transaction:
    """Represents a transaction on the NyxSynth blockchain."""
    sender: str
    recipient: str
    amount: float
    timestamp: float = field(default_factory=time.time)
    signature: str = None
    transaction_id: str = None
    
    def __post_init__(self):
        # Generate transaction ID if not provided
        if not self.transaction_id:
            data = f"{self.sender}{self.recipient}{self.amount}{self.timestamp}"
            self.transaction_id = hashlib.sha256(data.encode()).hexdigest()
    
    def sign(self, private_key):
        """Sign the transaction with the sender's private key."""
        crypto = QuantumResistantCrypto()
        self.signature = crypto.sign(self.to_dict(), private_key)
    
    def verify(self):
        """Verify the transaction signature."""
        if not self.signature:
            return False
        crypto = QuantumResistantCrypto()
        return crypto.verify(self.to_dict(), self.signature, self.sender)
    
    def to_dict(self):
        """Convert transaction to dictionary (excluding signature)."""
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "amount": self.amount,
            "timestamp": self.timestamp,
            "transaction_id": self.transaction_id
        }

@dataclass
class Block:
    """Represents a block in the NyxSynth blockchain."""
    index: int
    timestamp: float
    transactions: List[Transaction]
    previous_hash: str
    nonce: int = 0
    hash: str = None
    neural_pattern: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        # Generate block hash if not provided
        if not self.hash:
            self.hash = self.calculate_hash()
    
    def calculate_hash(self):
        """Calculate the hash of the block."""
        block_data = {
            "index": self.index,
            "timestamp": self.timestamp,
            "transactions": [t.to_dict() for t in self.transactions],
            "previous_hash": self.previous_hash,
            "nonce": self.nonce,
            "neural_pattern": self.neural_pattern
        }
        block_string = json.dumps(block_data, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def mine_block(self, difficulty):
        """Mine the block using Bioluminescent Coordination Protocol."""
        coordinator = BioluminescentCoordinator()
        neural_validator = NeuralValidator()
        
        # Generate initial neural pattern
        self.neural_pattern = neural_validator.generate_pattern(self.transactions)
        
        # Mine block until it meets the difficulty target
        target = "0" * difficulty
        while self.hash[:difficulty] != target:
            self.nonce += 1
            # Update neural pattern based on previous attempts
            self.neural_pattern = neural_validator.adapt_pattern(
                self.neural_pattern, 
                self.hash, 
                self.nonce
            )
            self.hash = self.calculate_hash()
            
            # Coordinate with network using BCP
            coordinator.emit_pattern(self.neural_pattern)
            
        return self.hash

class Blockchain:
    """NyxSynth blockchain implementation."""
    
    def __init__(self):
        self.chain = []
        self.pending_transactions = []
        self.nodes = set()
        self.difficulty = 4
        self.mining_reward = 5.0
        self.neural_validator = NeuralValidator()
        self.bcp_coordinator = BioluminescentCoordinator()
        self.quantum_crypto = QuantumResistantCrypto()
        self.consensus_threshold = 0.65  # Added consensus threshold for validation
        
        # Create genesis block
        self.create_genesis_block()
    
    def create_genesis_block(self):
        """Create the genesis block."""
        genesis_block = Block(
            index=0,
            timestamp=time.time(),
            transactions=[],
            previous_hash="0"
        )
        genesis_block.neural_pattern = self.neural_validator.generate_genesis_pattern()
        genesis_block.hash = genesis_block.calculate_hash()
        self.chain.append(genesis_block)
    
    def get_latest_block(self):
        """Get the latest block in the chain."""
        return self.chain[-1]
    
    def integrate_consensus_validation(self, transaction: Transaction) -> bool:
        """
        Integrate consensus and validation components for complete transaction verification.
        
        Args:
            transaction: Transaction to verify
            
        Returns:
            True if transaction passes all checks
        """
        # 1. Basic verification
        if not transaction.verify():
            return False
        
        # 2. Neural validation
        validation_result = self.neural_validator.validate_transaction(transaction, self)
        if not validation_result:
            return False
        
        # 3. Check transaction pattern against consensus
        tx_pattern = self.neural_validator._transaction_to_pattern(transaction)
        sync_score = self.bcp_coordinator.get_synchronization_score(tx_pattern)
        
        # Transaction must have minimum synchronization with network consensus
        if sync_score < self.consensus_threshold:
            return False
        
        # 4. Check for potential adversarial behavior
        is_anomalous = self.neural_validator._check_pattern_anomaly(tx_pattern, [transaction])
        if is_anomalous:
            return False
        
        return True
    
    def add_transaction(self, transaction):
        """Add a transaction to pending transactions."""
        if not transaction.sender or not transaction.recipient:
            raise ValueError("Transaction must include sender and recipient")
        
        # Use the new integrated consensus validation instead of simple verification
        if not self.integrate_consensus_validation(transaction):
            raise ValueError("Transaction validation failed")
        
        self.pending_transactions.append(transaction)
        return self.get_latest_block().index + 1
    
    def mine_pending_transactions(self, miner_address):
        """Mine pending transactions and add a new block to the chain."""
        # Create reward transaction
        reward_tx = Transaction("SYSTEM", miner_address, self.mining_reward)
        self.pending_transactions.append(reward_tx)
        
        # Create new block
        new_block = Block(
            index=len(self.chain),
            timestamp=time.time(),
            transactions=self.pending_transactions,
            previous_hash=self.get_latest_block().hash
        )
        
        # Mine the block
        new_block.mine_block(self.difficulty)
        
        # Add block to chain and clear pending transactions
        self.chain.append(new_block)
        self.pending_transactions = []
        
        # Adapt neural validator based on new block
        self.neural_validator.adapt(new_block)
        
        return new_block
    
    def is_chain_valid(self):
        """Validate the blockchain."""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]
            
            # Check if block hash is valid
            if current_block.hash != current_block.calculate_hash():
                return False
            
            # Check if block points to correct previous hash
            if current_block.previous_hash != previous_block.hash:
                return False
            
            # Validate all transactions in the block
            for transaction in current_block.transactions:
                if not transaction.verify() and transaction.sender != "SYSTEM":
                    return False
        
        return True
    
    def get_balance(self, address):
        """Get the balance of an address."""
        balance = 0
        
        for block in self.chain:
            for transaction in block.transactions:
                if transaction.sender == address:
                    balance -= transaction.amount
                if transaction.recipient == address:
                    balance += transaction.amount
        
        return balance
    
    def register_node(self, node_url):
        """Add a new node to the network."""
        self.nodes.add(node_url)
    
    def resolve_conflicts(self):
        """Consensus algorithm: resolve conflicts by replacing our chain with the longest valid chain."""
        new_chain = None
        max_length = len(self.chain)
        
        # Find the longest chain among all nodes
        for node in self.nodes:
            # Get the chain from the node
            response = requests.get(f"{node}/chain")
            
            if response.status_code == 200:
                chain_data = response.json()
                length = chain_data['length']
                chain = chain_data['chain']
                
                # Check if the chain is longer and valid
                if length > max_length and self.is_valid_chain(chain):
                    max_length = length
                    new_chain = chain
        
        # Replace our chain if we found a longer valid chain
        if new_chain:
            self.chain = new_chain
            return True
        
        return False
    
    def is_valid_chain(self, chain):
        """Check if a chain is valid."""
        # Implement validation logic for external chains
        return True  # Simplified for this example