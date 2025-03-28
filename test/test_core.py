import sys
import os
import unittest
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import blockchain components
from blockchain.core import Blockchain, Transaction, Block
from blockchain.crypto.quantum import QuantumResistantCrypto
from blockchain.neural.validator import NeuralValidator
from blockchain.consensus.bcp import BioluminescentCoordinator

class TestBlockchainCore(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.blockchain = Blockchain()
        self.crypto = QuantumResistantCrypto()
        self.validator = NeuralValidator()
        self.coordinator = BioluminescentCoordinator()
        
        # Generate test keys
        self.sender_keys = self.crypto.generate_keypair()
        self.recipient_keys = self.crypto.generate_keypair()
    
    def test_genesis_block(self):
        """Test that genesis block is created correctly."""
        genesis = self.blockchain.chain[0]
        
        self.assertEqual(genesis.index, 0)
        self.assertEqual(genesis.previous_hash, "0")
        self.assertIsNotNone(genesis.hash)
        self.assertEqual(len(genesis.transactions), 0)
        self.assertIsNotNone(genesis.neural_pattern)
    
    def test_transaction_creation(self):
        """Test creating a transaction."""
        tx = Transaction(
            sender=self.sender_keys["public_key"],
            recipient=self.recipient_keys["public_key"],
            amount=100.0
        )
        
        self.assertEqual(tx.sender, self.sender_keys["public_key"])
        self.assertEqual(tx.recipient, self.recipient_keys["public_key"])
        self.assertEqual(tx.amount, 100.0)
        self.assertIsNotNone(tx.transaction_id)
        self.assertIsNone(tx.signature)
    
    def test_transaction_signing(self):
        """Test signing a transaction."""
        tx = Transaction(
            sender=self.sender_keys["public_key"],
            recipient=self.recipient_keys["public_key"],
            amount=100.0
        )
        
        tx.sign(self.sender_keys["private_key"])
        self.assertIsNotNone(tx.signature)
        self.assertTrue(tx.verify())
    
    def test_block_creation(self):
        """Test creating a new block."""
        tx = Transaction(
            sender=self.sender_keys["public_key"],
            recipient=self.recipient_keys["public_key"],
            amount=100.0
        )
        tx.sign(self.sender_keys["private_key"])
        
        previous_block = self.blockchain.get_latest_block()
        
        new_block = Block(
            index=previous_block.index + 1,
            timestamp=1234567890,
            transactions=[tx],
            previous_hash=previous_block.hash
        )
        
        self.assertEqual(new_block.index, previous_block.index + 1)
        self.assertEqual(new_block.previous_hash, previous_block.hash)
        self.assertEqual(len(new_block.transactions), 1)
        self.assertIsNotNone(new_block.hash)
    
    def test_blockchain_validation(self):
        """Test blockchain validation."""
        # Add a block to the chain
        tx = Transaction(
            sender=self.sender_keys["public_key"],
            recipient=self.recipient_keys["public_key"],
            amount=100.0
        )
        tx.sign(self.sender_keys["private_key"])
        
        self.blockchain.pending_transactions.append(tx)
        self.blockchain.mine_pending_transactions(self.recipient_keys["public_key"])
        
        # Validate the chain
        self.assertTrue(self.blockchain.is_chain_valid())
        
        # Tamper with a transaction
        self.blockchain.chain[1].transactions[0].amount = 200.0
        
        # Chain should no longer be valid
        self.assertFalse(self.blockchain.is_chain_valid())
    
    def test_neural_validator(self):
        """Test neural validator pattern generation."""
        # Generate a pattern for the genesis block
        genesis_pattern = self.validator.generate_genesis_pattern()
        self.assertEqual(len(genesis_pattern), 64)
        
        # Create some test transactions
        tx1 = Transaction("sender1", "recipient1", 100.0)
        tx2 = Transaction("sender2", "recipient2", 200.0)
        
        # Generate a pattern for these transactions
        pattern = self.validator.generate_pattern([tx1, tx2])
        self.assertEqual(len(pattern), 64)
        
        # Test pattern adaptation
        adapted_pattern = self.validator.adapt_pattern(
            pattern, 
            "a" * 64,  # mock hash
            42  # mock nonce
        )
        self.assertEqual(len(adapted_pattern), 64)
    
    def test_bioluminescent_coordinator(self):
        """Test bioluminescent coordination protocol."""
        # Generate test patterns
        pattern1 = [0.1] * 64
        pattern2 = [0.2] * 64
        
        # Emit patterns
        self.coordinator.emit_pattern(pattern1)
        self.assertEqual(len(self.coordinator.pattern_history), 1)
        
        # Get consensus pattern
        consensus = self.coordinator.get_consensus_pattern()
        self.assertEqual(len(consensus), 64)
        
        # Check synchronization
        sync_score = self.coordinator.get_synchronization_score(pattern1)
        self.assertGreater(sync_score, 0.5)
        
        # Adapt a pattern
        adapted = self.coordinator.adapt_to_network(pattern2)
        self.assertEqual(len(adapted), 64)
    
    def test_quantum_crypto(self):
        """Test quantum-resistant cryptography."""
        # Generate keypair
        keypair = self.crypto.generate_keypair()
        self.assertIn("private_key", keypair)
        self.assertIn("public_key", keypair)
        
        # Hash data
        data = {"test": "data"}
        hash_digest = self.crypto.hash(data)
        self.assertIsInstance(hash_digest, str)
        
        # Sign and verify
        signature = self.crypto.sign(data, keypair["private_key"])
        self.assertTrue(self.crypto.verify(data, signature, keypair["public_key"]))
        
        # Encrypt and decrypt
        encrypted = self.crypto.encrypt("secret message", keypair["public_key"])
        decrypted = self.crypto.decrypt(encrypted, keypair["private_key"])
        self.assertEqual(decrypted, "secret message")

if __name__ == "__main__":
    unittest.main()
