import json
import time
from typing import List, Dict, Any, Callable
import hashlib

class Contract:
    """Base class for all smart contracts."""
    
    def __init__(self, owner, address=None):
        self.owner = owner
        self.created_at = time.time()
        self.address = address or self._generate_address()
        self.state = {}
        self.relationships = []
    
    def _generate_address(self):
        """Generate a unique contract address."""
        data = f"{self.owner}{self.created_at}{hash(self)}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    def get_state(self):
        """Get the current contract state."""
        return self.state
    
    def update_state(self, new_state):
        """Update the contract state."""
        self.state.update(new_state)
    
    def form_relationship(self, other_contract):
        """Form a symbiotic relationship with another contract."""
        if other_contract not in self.relationships:
            self.relationships.append(other_contract)
    
    def call_related(self, method_name, params=None):
        """Call a method on all related contracts."""
        results = []
        for contract in self.relationships:
            if hasattr(contract, method_name):
                method = getattr(contract, method_name)
                results.append(method(**(params or {})))
        return results

class SymbioticRegistry:
    """Registry for managing symbiotic contracts."""
    
    def __init__(self):
        self.contracts = {}
        self.relationships = {}
    
    def register_contract(self, contract):
        """Register a contract in the registry."""
        self.contracts[contract.address] = contract
        self.relationships[contract.address] = []
    
    def create_relationship(self, contract1_address, contract2_address, relationship_type):
        """Create a symbiotic relationship between contracts."""
        if contract1_address not in self.contracts or contract2_address not in self.contracts:
            raise ValueError("One or both contracts are not registered")
        
        contract1 = self.contracts[contract1_address]
        contract2 = self.contracts[contract2_address]
        
        # Form the relationship
        contract1.form_relationship(contract2)
        contract2.form_relationship(contract1)
        
        # Record the relationship
        self.relationships[contract1_address].append({
            "related_to": contract2_address,
            "type": relationship_type
        })
        
        self.relationships[contract2_address].append({
            "related_to": contract1_address,
            "type": relationship_type
        })
    
    def get_contract(self, address):
        """Get a contract by address."""
        return self.contracts.get(address)
    
    def get_relationships(self, contract_address):
        """Get all relationships for a contract."""
        return self.relationships.get(contract_address, [])

class TokenContract(Contract):
    """NYX token implementation as a smart contract."""
    
    def __init__(self, owner, total_supply=808000000, address=None):
        super().__init__(owner, address)
        self.state = {
            "name": "NyxSynth",
            "symbol": "NYX",
            "total_supply": total_supply,
            "decimals": 18,
            "balances": {owner: total_supply},
            "allowances": {},
            "burned": 0
        }
    
    def balance_of(self, address):
        """Get the token balance of an address."""
        return self.state["balances"].get(address, 0)
    
    def transfer(self, from_address, to_address, amount):
        """Transfer tokens from one address to another."""
        # Check if sender has sufficient balance
        if self.state["balances"].get(from_address, 0) < amount:
            raise ValueError("Insufficient balance")
        
        # Update balances
        self.state["balances"][from_address] = self.state["balances"].get(from_address, 0) - amount
        self.state["balances"][to_address] = self.state["balances"].get(to_address, 0) + amount
        
        # Apply automatic burning (0.5% of each transaction)
        burn_amount = amount * 0.005
        self.burn(burn_amount)
        
        return True
    
    def approve(self, owner, spender, amount):
        """Approve a spender to spend tokens on behalf of the owner."""
        if owner not in self.state["allowances"]:
            self.state["allowances"][owner] = {}
        
        self.state["allowances"][owner][spender] = amount
        return True
    
    def allowance(self, owner, spender):
        """Get the amount of tokens approved for a spender."""
        return self.state["allowances"].get(owner, {}).get(spender, 0)
    
    def transfer_from(self, sender, from_address, to_address, amount):
        """Transfer tokens on behalf of another address."""
        # Check if sender has sufficient allowance
        allowed = self.allowance(from_address, sender)
        if allowed < amount:
            raise ValueError("Insufficient allowance")
        
        # Check if from_address has sufficient balance
        if self.state["balances"].get(from_address, 0) < amount:
            raise ValueError("Insufficient balance")
        
        # Update allowance
        self.state["allowances"][from_address][sender] = allowed - amount
        
        # Update balances
        self.state["balances"][from_address] = self.state["balances"].get(from_address, 0) - amount
        self.state["balances"][to_address] = self.state["balances"].get(to_address, 0) + amount
        
        # Apply automatic burning (0.5% of each transaction)
        burn_amount = amount * 0.005
        self.burn(burn_amount)
        
        return True
    
    def burn(self, amount):
        """Burn tokens, removing them from circulation."""
        # Update total supply and burned amount
        self.state["total_supply"] -= amount
        self.state["burned"] += amount
        
        # Check for symbiotic contracts that might react to burning
        self.call_related("on_token_burn", {"amount": amount})
        
        return True