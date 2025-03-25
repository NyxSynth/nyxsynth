import os
import json
import time
import uuid
from typing import List, Dict, Any
from flask import Flask, request, jsonify
from blockchain.core import Blockchain, Transaction
from blockchain.crypto.quantum import QuantumResistantCrypto
from contracts.symbiotic import TokenContract, StakingContract, SymbioticRegistry

app = Flask(__name__)

# Initialize blockchain and related components
blockchain = Blockchain()
quantum_crypto = QuantumResistantCrypto()
registry = SymbioticRegistry()

# Initialize token contract (owner is the genesis address)
genesis_keys = quantum_crypto.generate_keypair()
genesis_address = genesis_keys["public_key"]
token_contract = TokenContract(genesis_address)
registry.register_contract(token_contract)

# Initialize staking contract
staking_contract = StakingContract(genesis_address, token_contract.address)
registry.register_contract(staking_contract)

# Create relationship between contracts
registry.create_relationship(
    token_contract.address,
    staking_contract.address,
    "token-staking"
)

# Store contracts in global state
app.config["BLOCKCHAIN"] = blockchain
app.config["TOKEN_CONTRACT"] = token_contract
app.config["STAKING_CONTRACT"] = staking_contract
app.config["REGISTRY"] = registry

# Enable CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

# Wallet API endpoints
@app.route('/api/wallet/create', methods=['POST'])
def create_wallet():
    """Create a new wallet with private/public key pair."""
    try:
        keypair = quantum_crypto.generate_keypair()
        
        # Airdrop some tokens to the new wallet for testing
        token_contract.transfer(genesis_address, keypair["public_key"], 1000)
        
        return jsonify({
            "success": True,
            "wallet": {
                "address": keypair["public_key"],
                "privateKey": keypair["private_key"]
            }
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500

@app.route('/api/wallet/import', methods=['POST'])
def import_wallet():
    """Import a wallet using a private key."""
    try:
        data = request.json
        private_key = data.get('privateKey')
        
        if not private_key:
            return jsonify({
                "success": False,
                "message": "Private key is required"
            }), 400
        
        # Derive public key from private key
        # In a real implementation, this would use proper key derivation
        # For this example, we're using a simplified approach
        public_key = private_key[::-1]  # Reversed private key as placeholder
        
        return jsonify({
            "success": True,
            "wallet": {
                "address": public_key,
                "privateKey": private_key
            }
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500

@app.route('/api/wallet/balance/<address>', methods=['GET'])
def get_balance(address):
    """Get the balance of a wallet address."""
    try:
        # Get token balance from the contract
        balance = token_contract.balance_of(address)
        
        return jsonify({
            "success": True,
            "balance": balance
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500

if __name__ == '__main__':
    # Create initial staking pools if none exist
    if not staking_contract.state["staking_pools"]:
        staking_contract.create_pool("pool1", "Standard Neural Pool", 1.0)
        staking_contract.create_pool("pool2", "Advanced Neural Pool", 1.5)
        staking_contract.create_pool("pool3", "Expert Neural Pool", 2.0)
    
    # Run the server
    app.run(host='0.0.0.0', port=5000, debug=True)