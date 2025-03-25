#!/usr/bin/env python3

import os
import sys
import json
import time
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import blockchain components
from blockchain.core import Blockchain
from blockchain.crypto.quantum import QuantumResistantCrypto
from contracts.symbiotic import TokenContract, StakingContract, SymbioticRegistry

def initialize_blockchain(config_dir, data_dir, admin_user, admin_pass):
    """Initialize the NyxSynth blockchain and contracts."""
    print("Initializing NyxSynth blockchain...")
    
    # Create data directory if it doesn't exist
    Path(data_dir).mkdir(exist_ok=True, parents=True)
    Path(config_dir).mkdir(exist_ok=True, parents=True)
    
    # Initialize blockchain
    blockchain = Blockchain()
    
    # Initialize quantum-resistant cryptography
    crypto = QuantumResistantCrypto()
    
    # Generate genesis keys
    genesis_keys = crypto.generate_keypair()
    genesis_address = genesis_keys["public_key"]
    
    print(f"Genesis address: {genesis_address}")
    
    # Initialize token contract
    token_contract = TokenContract(genesis_address)
    
    # Initialize staking contract
    staking_contract = StakingContract(genesis_address, token_contract.address)
    
    # Initialize registry
    registry = SymbioticRegistry()
    registry.register_contract(token_contract)
    registry.register_contract(staking_contract)
    
    # Create relationship between contracts
    registry.create_relationship(
        token_contract.address,
        staking_contract.address,
        "token-staking"
    )
    
    # Create initial staking pools
    staking_contract.create_pool("pool1", "Standard Neural Pool", 1.0)
    staking_contract.create_pool("pool2", "Advanced Neural Pool", 1.5)
    staking_contract.create_pool("pool3", "Expert Neural Pool", 2.0)
    
    # Distribute initial tokens according to tokenomics
    total_supply = token_contract.state["total_supply"]
    
    # Create addresses for different token allocations
    allocations = {
        "neural_staking": crypto.generate_keypair()["public_key"],
        "ecosystem_dev": crypto.generate_keypair()["public_key"],
        "research": crypto.generate_keypair()["public_key"],
        "community": crypto.generate_keypair()["public_key"]
    }
    
    # Transfer tokens according to distribution plan
    token_contract.transfer(genesis_address, allocations["neural_staking"], total_supply * 0.42)  # 42% to neural staking
    token_contract.transfer(genesis_address, allocations["ecosystem_dev"], total_supply * 0.30)  # 30% to ecosystem development
    token_contract.transfer(genesis_address, allocations["research"], total_supply * 0.15)       # 15% to research
    token_contract.transfer(genesis_address, allocations["community"], total_supply * 0.13)      # 13% to community growth
    
    # Save blockchain state
    save_blockchain_state(data_dir, blockchain, registry, genesis_keys, allocations, admin_user, admin_pass)
    
    print("Blockchain initialization complete!")
    return blockchain, registry, genesis_keys, allocations

def save_blockchain_state(data_dir, blockchain, registry, genesis_keys, allocations, admin_user, admin_pass):
    """Save the blockchain state to disk."""
    data_dir_path = Path(data_dir)
    
    # Save genesis keys (in a real system, this should be kept secure)
    with open(data_dir_path / "genesis_keys.json", "w") as f:
        json.dump(genesis_keys, f, indent=4)
    
    # Save allocation addresses
    with open(data_dir_path / "allocations.json", "w") as f:
        json.dump(allocations, f, indent=4)
    
    # Save admin credentials (in a real system, this would be configured securely)
    admin_credentials = {
        "username": admin_user,
        "password": admin_pass  # In a real system, this would be a secure password
    }
    
    with open(data_dir_path / "admin.json", "w") as f:
        json.dump(admin_credentials, f, indent=4)
    
    print("Blockchain state saved to disk.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Initialize NyxSynth blockchain')
    parser.add_argument('--config-dir', default='/etc/nyxsynth', help='Configuration directory')
    parser.add_argument('--data-dir', default='/var/lib/nyxsynth', help='Data directory')
    parser.add_argument('--admin-user', default='admin', help='Admin username')
    parser.add_argument('--admin-pass', default='nyxadmin123', help='Admin password')
    
    args = parser.parse_args()
    
    initialize_blockchain(args.config_dir, args.data_dir, args.admin_user, args.admin_pass)