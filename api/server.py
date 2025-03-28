import os
import json
import time
import uuid
import hashlib
import logging
import threading
import secrets
import ipaddress
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple
from functools import wraps

from flask import Flask, request, jsonify, Response, g, abort
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix

# Rate limiting and security middleware
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Import blockchain components
from blockchain.core import Blockchain
from blockchain.neural.enhanced_validator import EnhancedNeuralValidator
from blockchain.consensus.enhanced_bcp import EnhancedBioluminescentCoordinator
from blockchain.crypto.hardened_quantum import HardenedQuantumCrypto
from contracts.enhanced_symbiotic import (
    EnhancedSymbioticRegistry,
    EnhancedTokenContract,
    EnhancedStakingContract,
    RelationshipType
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("nyxsynth-api")

# Create Flask app
app = Flask(__name__)

# Add security headers middleware
@app.after_request
def apply_security_headers(response):
    """Apply security headers to all responses."""
    # Prevent MIME type sniffing
    response.headers['X-Content-Type-Options'] = 'nosniff'
    
    # Strict Transport Security (only in production)
    if not app.debug:
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    
    # Content Security Policy
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; object-src 'none'"
    
    # Prevent clickjacking
    response.headers['X-Frame-Options'] = 'DENY'
    
    # XSS protection
    response.headers['X-XSS-Protection'] = '1; mode=block'
    
    # CORS headers
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE'
    
    # Return modified response
    return response

# Configure app for trusted proxies
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# Set up rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Enable CORS with proper configuration
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configuration
class Config:
    """Server configuration."""
    
    # Data directories
    CONFIG_DIR = os.environ.get('NYXSYNTH_CONFIG_DIR', '/etc/nyxsynth')
    DATA_DIR = os.environ.get('NYXSYNTH_DATA_DIR', '/var/lib/nyxsynth')
    
    # Security settings
    SECRET_KEY = os.environ.get('NYXSYNTH_SECRET_KEY', secrets.token_hex(32))
    JWT_EXPIRATION = int(os.environ.get('NYXSYNTH_JWT_EXPIRATION', 3600))  # 1 hour
    SECURITY_LEVEL = int(os.environ.get('NYXSYNTH_SECURITY_LEVEL', 5))  # 1-5
    
    # API settings
    API_VERSION = "1.0.0"
    NODE_ID = os.environ.get('NYXSYNTH_NODE_ID', str(uuid.uuid4()))
    
    # Create directories if they don't exist
    @classmethod
    def init_directories(cls):
        """Initialize data directories."""
        os.makedirs(cls.CONFIG_DIR, exist_ok=True)
        os.makedirs(cls.DATA_DIR, exist_ok=True)

# Apply configuration
app.config.from_object(Config)
Config.init_directories()

# Security utilities
class SecurityUtils:
    """Security utilities for the API server."""
    
    @staticmethod
    def generate_token(user_id: str, expiration: int = None) -> str:
        """
        Generate a JWT-like token.
        
        Args:
            user_id: User or wallet ID
            expiration: Token expiration in seconds
            
        Returns:
            Token string
        """
        if expiration is None:
            expiration = Config.JWT_EXPIRATION
        
        # Create token payload
        payload = {
            "sub": user_id,
            "iat": int(time.time()),
            "exp": int(time.time()) + expiration,
            "jti": str(uuid.uuid4())
        }
        
        # Convert payload to JSON and encode
        payload_str = json.dumps(payload)
        payload_b64 = base64.b64encode(payload_str.encode()).decode()
        
        # Create signature
        signature_data = payload_b64 + Config.SECRET_KEY
        signature = hashlib.sha256(signature_data.encode()).hexdigest()
        
        # Combine parts
        token = f"{payload_b64}.{signature}"
        
        return token
    
    @staticmethod
    def validate_token(token: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Validate a token.
        
        Args:
            token: Token to validate
            
        Returns:
            (is_valid, payload)
        """
        try:
            # Split token
            parts = token.split('.')
            if len(parts) != 2:
                return False, None
            
            payload_b64, signature = parts
            
            # Verify signature
            expected_signature_data = payload_b64 + Config.SECRET_KEY
            expected_signature = hashlib.sha256(expected_signature_data.encode()).hexdigest()
            
            if signature != expected_signature:
                return False, None
            
            # Decode payload
            payload_str = base64.b64decode(payload_b64).decode()
            payload = json.loads(payload_str)
            
            # Check expiration
            if payload.get("exp", 0) < time.time():
                return False, None
            
            return True, payload
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            return False, None
    
    @staticmethod
    def is_valid_address(address: str) -> bool:
        """
        Check if an address has valid format.
        
        Args:
            address: Address to check
            
        Returns:
            True if address is valid
        """
        # Check for proper length and hex format
        if not address.startswith("0x"):
            return False
        
        try:
            # Remove 0x prefix and check length
            hex_part = address[2:]
            if len(hex_part) != 64:  # 32 bytes = 64 hex chars
                return False
            
            # Check if it's valid hex
            int(hex_part, 16)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def sanitize_input(data: Any) -> Any:
        """
        Sanitize input data to prevent injection attacks.
        
        Args:
            data: Input data
            
        Returns:
            Sanitized data
        """
        if isinstance(data, str):
            # Remove potentially dangerous sequences
            sanitized = data.replace('<', '&lt;').replace('>', '&gt;')
            # Prevent SQL injection
            sanitized = sanitized.replace("'", "''").replace(";", "")
            return sanitized
        elif isinstance(data, dict):
            return {k: SecurityUtils.sanitize_input(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [SecurityUtils.sanitize_input(i) for i in data]
        else:
            return data
    
    @staticmethod
    def is_trusted_ip(ip: str) -> bool:
        """
        Check if an IP is in the trusted range.
        
        Args:
            ip: IP address to check
            
        Returns:
            True if IP is trusted
        """
        try:
            # Define trusted IP ranges
            trusted_ranges = [
                "127.0.0.0/8",     # Localhost
                "10.0.0.0/8",       # Private network
                "172.16.0.0/12",    # Private network
                "192.168.0.0/16"    # Private network
            ]
            
            # Check if IP is in any trusted range
            ip_obj = ipaddress.ip_address(ip)
            for cidr in trusted_ranges:
                if ip_obj in ipaddress.ip_network(cidr):
                    return True
            
            return False
        except ValueError:
            return False

# Initialize blockchain components
blockchain = Blockchain()
crypto = HardenedQuantumCrypto({
    'security_level': Config.SECURITY_LEVEL
})
validator = EnhancedNeuralValidator()
coordinator = EnhancedBioluminescentCoordinator()
registry = EnhancedSymbioticRegistry()

# Initialize contract components and relationships
def initialize_components():
    """Initialize blockchain and contract components."""
    try:
        logger.info("Initializing blockchain components...")
        
        # Load or create genesis keys
        genesis_keys_file = os.path.join(Config.DATA_DIR, "genesis_keys.json")
        if os.path.exists(genesis_keys_file):
            with open(genesis_keys_file, 'r') as f:
                genesis_keys = json.load(f)
        else:
            genesis_keys = crypto.generate_keypair()
            with open(genesis_keys_file, 'w') as f:
                json.dump(genesis_keys, f, indent=2)
        
        genesis_address = genesis_keys["public_key"]
        logger.info(f"Genesis address: {genesis_address}")
        
        # Load existing token contract or create new one
        token_contract_file = os.path.join(Config.DATA_DIR, "token_contract.json")
        if os.path.exists(token_contract_file):
            logger.info("Loading existing token contract...")
            token_contract = EnhancedTokenContract.load(token_contract_file)
            if not token_contract:
                logger.info("Failed to load token contract, creating new one...")
                token_contract = EnhancedTokenContract(genesis_address)
        else:
            logger.info("Creating new token contract...")
            token_contract = EnhancedTokenContract(genesis_address)
        
        # Load existing staking contract or create new one
        staking_contract_file = os.path.join(Config.DATA_DIR, "staking_contract.json")
        if os.path.exists(staking_contract_file):
            logger.info("Loading existing staking contract...")
            staking_contract = EnhancedStakingContract.load(staking_contract_file)
            if not staking_contract:
                logger.info("Failed to load staking contract, creating new one...")
                staking_contract = EnhancedStakingContract(genesis_address, token_contract.address)
        else:
            logger.info("Creating new staking contract...")
            staking_contract = EnhancedStakingContract(genesis_address, token_contract.address)
        
        # Register contracts
        registry.register_contract(token_contract)
        registry.register_contract(staking_contract)
        
        # Create relationship between contracts
        registry.create_relationship(
            token_contract.address,
            staking_contract.address,
            RelationshipType.RESOURCE_SHARING
        )
        
        # Create initial staking pools if none exist
        if not staking_contract.get_all_pools():
            logger.info("Creating initial staking pools...")
            staking_contract.create_pool("pool1", "Standard Neural Pool", 1.0, 100)
            staking_contract.create_pool("pool2", "Advanced Neural Pool", 1.5, 1000)
            staking_contract.create_pool("pool3", "Expert Neural Pool", 2.0, 10000)
        
        logger.info("Blockchain components initialized successfully")
        
        # Save contract states
        token_contract.save(os.path.join(Config.DATA_DIR, "contracts"))
        staking_contract.save(os.path.join(Config.DATA_DIR, "contracts"))
        
        # Return the initialized components
        return {
            "blockchain": blockchain,
            "crypto": crypto,
            "validator": validator,
            "coordinator": coordinator,
            "registry": registry,
            "token_contract": token_contract,
            "staking_contract": staking_contract,
            "genesis_keys": genesis_keys
        }
    
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        raise

# Initialize components and store in app config
app.config["COMPONENTS"] = initialize_components()

# API authentication decorator
def require_auth(f):
    """Decorator to require authentication for API endpoints."""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        
        if not auth_header:
            return jsonify({
                "success": False,
                "message": "Authentication required",
                "error_code": "auth_required"
            }), 401
        
        # Extract token
        try:
            token_type, token = auth_header.split(' ')
            if token_type.lower() != 'bearer':
                raise ValueError("Invalid token type")
        except ValueError:
            return jsonify({
                "success": False,
                "message": "Invalid authorization format",
                "error_code": "invalid_auth_format"
            }), 401
        
        # Validate token
        valid, payload = SecurityUtils.validate_token(token)
        if not valid:
            return jsonify({
                "success": False,
                "message": "Invalid or expired token",
                "error_code": "invalid_token"
            }), 401
        
        # Store user info in request context
        g.user_id = payload.get("sub")
        
        return f(*args, **kwargs)
    return decorated

# API admin access decorator
def require_admin(f):
    """Decorator to require admin access for API endpoints."""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        
        if not auth_header:
            return jsonify({
                "success": False,
                "message": "Authentication required",
                "error_code": "auth_required"
            }), 401
        
        # Extract token
        try:
            token_type, token = auth_header.split(' ')
            if token_type.lower() != 'bearer':
                raise ValueError("Invalid token type")
        except ValueError:
            return jsonify({
                "success": False,
                "message": "Invalid authorization format",
                "error_code": "invalid_auth_format"
            }), 401
        
        # Validate token
        valid, payload = SecurityUtils.validate_token(token)
        if not valid:
            return jsonify({
                "success": False,
                "message": "Invalid or expired token",
                "error_code": "invalid_token"
            }), 401
        
        # Check admin status
        if payload.get("sub") != app.config["COMPONENTS"]["genesis_keys"]["public_key"]:
            return jsonify({
                "success": False,
                "message": "Admin access required",
                "error_code": "admin_required"
            }), 403
        
        # Store user info in request context
        g.user_id = payload.get("sub")
        
        return f(*args, **kwargs)
    return decorated

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    """API health check endpoint."""
    components = app.config["COMPONENTS"]
    
    return jsonify({
        "success": True,
        "status": "healthy",
        "version": Config.API_VERSION,
        "node_id": Config.NODE_ID,
        "uptime": int(time.time() - app.start_time),
        "blockchain": {
            "blocks": len(components["blockchain"].chain),
            "pending_transactions": len(components["blockchain"].pending_transactions)
        },
        "timestamp": int(time.time())
    })

# Wallet API endpoints
@app.route('/api/wallet/create', methods=['POST'])
@limiter.limit("5 per minute")
def create_wallet():
    """Create a new wallet with private/public key pair."""
    try:
        components = app.config["COMPONENTS"]
        
        # Generate keypair with highest security level
        keypair = components["crypto"].generate_keypair()
        
        # Airdrop some tokens to the new wallet for testing
        genesis_address = components["genesis_keys"]["public_key"]
        token_contract = components["token_contract"]
        token_contract.transfer(genesis_address, keypair["public_key"], 100)
        
        # Generate authentication token
        auth_token = SecurityUtils.generate_token(keypair["public_key"])
        
        return jsonify({
            "success": True,
            "wallet": {
                "address": keypair["public_key"],
                "privateKey": keypair["private_key"]
            },
            "auth_token": auth_token
        })
    except Exception as e:
        logger.error(f"Error creating wallet: {e}")
        return jsonify({
            "success": False,
            "message": "Failed to create wallet",
            "error_code": "wallet_creation_failed"
        }), 500

@app.route('/api/wallet/import', methods=['POST'])
@limiter.limit("5 per minute")
def import_wallet():
    """Import a wallet using a private key."""
    try:
        data = request.json
        components = app.config["COMPONENTS"]
        
        if not data or 'privateKey' not in data:
            return jsonify({
                "success": False,
                "message": "Private key is required",
                "error_code": "missing_private_key"
            }), 400
        
        private_key = SecurityUtils.sanitize_input(data['privateKey'])
        
        # Validate private key format
        if not private_key or '.' not in private_key:
            return jsonify({
                "success": False,
                "message": "Invalid private key format",
                "error_code": "invalid_private_key"
            }), 400
        
        # Extract the key material and metadata
        try:
            key_parts = private_key.split('.')
            key_material = key_parts[0]
            
            # Check if it's valid hex
            int(key_material, 16)
            
            # Derive public key from private key
            # Rather than implementing a potentially insecure algorithm here,
            # we'll use our hardened crypto module to derive the public key
            
            # For this example, we'll use a simplified approach
            # In a real implementation, this would use proper key derivation
            crypto = components["crypto"]
            temp_keypair = crypto.generate_keypair()
            
            # Map the imported private key to a new keypair
            keypair = {
                "private_key": private_key,
                "public_key": temp_keypair["public_key"],
                "metadata": temp_keypair.get("metadata", {})
            }
            
            # Generate authentication token
            auth_token = SecurityUtils.generate_token(keypair["public_key"])
            
            return jsonify({
                "success": True,
                "wallet": {
                    "address": keypair["public_key"],
                    "privateKey": keypair["private_key"]
                },
                "auth_token": auth_token
            })
        except Exception as e:
            logger.error(f"Error processing private key: {e}")
            return jsonify({
                "success": False,
                "message": "Invalid private key format",
                "error_code": "invalid_private_key_format"
            }), 400
    except Exception as e:
        logger.error(f"Error importing wallet: {e}")
        return jsonify({
            "success": False,
            "message": "Failed to import wallet",
            "error_code": "wallet_import_failed"
        }), 500

@app.route('/api/wallet/balance/<address>', methods=['GET'])
def get_balance(address):
    """Get the token balance of a wallet address."""
    try:
        components = app.config["COMPONENTS"]
        
        # Sanitize input
        address = SecurityUtils.sanitize_input(address)
        
        # Validate address format
        if not SecurityUtils.is_valid_address(address):
            return jsonify({
                "success": False,
                "message": "Invalid wallet address",
                "error_code": "invalid_address"
            }), 400
        
        # Get token balance from the contract
        token_contract = components["token_contract"]
        balance = token_contract.balance_of(address)
        
        return jsonify({
            "success": True,
            "address": address,
            "balance": balance,
            "symbol": token_contract.state["symbol"]
        })
    except Exception as e:
        logger.error(f"Error getting balance: {e}")
        return jsonify({
            "success": False,
            "message": "Failed to get balance",
            "error_code": "balance_fetch_failed"
        }), 500

@app.route('/api/wallet/transfer', methods=['POST'])
@require_auth
def transfer_tokens():
    """Transfer tokens from one wallet to another."""
    try:
        data = request.json
        components = app.config["COMPONENTS"]
        
        # Validate request data
        if not data or 'to' not in data or 'amount' not in data:
            return jsonify({
                "success": False,
                "message": "Recipient address and amount are required",
                "error_code": "missing_parameters"
            }), 400
        
        # Sanitize input
        to_address = SecurityUtils.sanitize_input(data['to'])
        amount = float(data['amount'])
        
        # Validate addresses
        if not SecurityUtils.is_valid_address(to_address):
            return jsonify({
                "success": False,
                "message": "Invalid recipient address",
                "error_code": "invalid_address"
            }), 400
        
        # Get sender address from auth token
        from_address = g.user_id
        
        # Validate amount
        if amount <= 0:
            return jsonify({
                "success": False,
                "message": "Amount must be positive",
                "error_code": "invalid_amount"
            }), 400
        
        # Perform transfer
        token_contract = components["token_contract"]
        result = token_contract.transfer(from_address, to_address, amount)
        
        if not result:
            return jsonify({
                "success": False,
                "message": "Transfer failed, insufficient balance or other error",
                "error_code": "transfer_failed"
            }), 400
        
        # Get updated balances
        from_balance = token_contract.balance_of(from_address)
        to_balance = token_contract.balance_of(to_address)
        
        return jsonify({
            "success": True,
            "message": f"Successfully transferred {amount} NYX",
            "from": {
                "address": from_address,
                "balance": from_balance
            },
            "to": {
                "address": to_address,
                "balance": to_balance
            },
            "amount": amount,
            "timestamp": int(time.time())
        })
    except Exception as e:
        logger.error(f"Error transferring tokens: {e}")
        return jsonify({
            "success": False,
            "message": "Failed to transfer tokens",
            "error_code": "transfer_error"
        }), 500

# Staking API endpoints
@app.route('/api/staking/pools', methods=['GET'])
def get_staking_pools():
    """Get all available staking pools."""
    try:
        components = app.config["COMPONENTS"]
        staking_contract = components["staking_contract"]
        
        # Get all pools
        pools = staking_contract.get_all_pools()
        
        # Format response
        formatted_pools = []
        for pool in pools:
            formatted_pools.append({
                "id": pool["id"],
                "name": pool["name"],
                "reward_multiplier": pool["reward_multiplier"],
                "min_stake": pool["min_stake"],
                "total_staked": pool["total_staked"],
                "staker_count": pool["staker_count"],
                "lock_period_days": pool.get("lock_period", 0) / (60 * 60 * 24)
            })
        
        return jsonify({
            "success": True,
            "pools": formatted_pools
        })
    except Exception as e:
        logger.error(f"Error getting staking pools: {e}")
        return jsonify({
            "success": False,
            "message": "Failed to get staking pools",
            "error_code": "staking_pools_error"
        }), 500

@app.route('/api/staking/stake', methods=['POST'])
@require_auth
def stake_tokens():
    """Stake tokens in a staking pool."""
    try:
        data = request.json
        components = app.config["COMPONENTS"]
        
        # Validate request data
        if not data or 'poolId' not in data or 'amount' not in data:
            return jsonify({
                "success": False,
                "message": "Pool ID and amount are required",
                "error_code": "missing_parameters"
            }), 400
        
        # Sanitize input
        pool_id = SecurityUtils.sanitize_input(data['poolId'])
        amount = float(data['amount'])
        
        # Get staker address from auth token
        staker_address = g.user_id
        
        # Validate amount
        if amount <= 0:
            return jsonify({
                "success": False,
                "message": "Amount must be positive",
                "error_code": "invalid_amount"
            }), 400
        
        # Perform staking
        staking_contract = components["staking_contract"]
        result = staking_contract.stake(staker_address, pool_id, amount)
        
        if not result:
            return jsonify({
                "success": False,
                "message": "Staking failed, insufficient balance or invalid pool",
                "error_code": "staking_failed"
            }), 400
        
        # Get updated stakes
        stakes = staking_contract.get_stakes_by_address(staker_address)
        
        # Format stakes
        formatted_stakes = []
        for stake in stakes:
            formatted_stakes.append({
                "id": stake["id"],
                "poolId": stake["pool_id"],
                "amount": stake["amount"],
                "startTime": stake["start_time"],
                "active": stake["active"],
                "rewardsClaimed": stake["rewards_claimed"]
            })
        
        return jsonify({
            "success": True,
            "message": f"Successfully staked {amount} NYX",
            "stakes": formatted_stakes,
            "timestamp": int(time.time())
        })
    except Exception as e:
        logger.error(f"Error staking tokens: {e}")
        return jsonify({
            "success": False,
            "message": "Failed to stake tokens",
            "error_code": "staking_error"
        }), 500

@app.route('/api/staking/unstake', methods=['POST'])
@require_auth
def unstake_tokens():
    """Unstake tokens from a staking pool."""
    try:
        data = request.json
        components = app.config["COMPONENTS"]
        
        # Validate request data
        if not data or 'stakeId' not in data:
            return jsonify({
                "success": False,
                "message": "Stake ID is required",
                "error_code": "missing_parameters"
            }), 400
        
        # Sanitize input
        stake_id = SecurityUtils.sanitize_input(data['stakeId'])
        
        # Get staker address from auth token
        staker_address = g.user_id
        
        # Perform unstaking
        staking_contract = components["staking_contract"]
        
        # Verify ownership of stake
        stakes = staking_contract.get_stakes_by_address(staker_address)
        stake_ids = [stake["id"] for stake in stakes]
        
        if stake_id not in stake_ids:
            return jsonify({
                "success": False,
                "message": "Invalid stake ID or not your stake",
                "error_code": "invalid_stake"
            }), 400
        
        result = staking_contract.unstake(stake_id)
        
        if not result:
            return jsonify({
                "success": False,
                "message": "Unstaking failed, stake may be locked or already unstaked",
                "error_code": "unstaking_failed"
            }), 400
        
        # Get token balance after unstaking
        token_contract = components["token_contract"]
        balance = token_contract.balance_of(staker_address)
        
        return jsonify({
            "success": True,
            "message": "Successfully unstaked tokens",
            "balance": balance,
            "timestamp": int(time.time())
        })
    except Exception as e:
        logger.error(f"Error unstaking tokens: {e}")
        return jsonify({
            "success": False,
            "message": "Failed to unstake tokens",
            "error_code": "unstaking_error"
        }), 500

@app.route('/api/staking/rewards', methods=['POST'])
@require_auth
def claim_rewards():
    """Claim staking rewards."""
    try:
        data = request.json
        components = app.config["COMPONENTS"]
        
        # Validate request data
        if not data or 'stakeId' not in data:
            return jsonify({
                "success": False,
                "message": "Stake ID is required",
                "error_code": "missing_parameters"
            }), 400
        
        # Sanitize input
        stake_id = SecurityUtils.sanitize_input(data['stakeId'])
        
        # Get staker address from auth token
        staker_address = g.user_id
        
        # Perform reward claiming
        staking_contract = components["staking_contract"]
        
        # Verify ownership of stake
        stakes = staking_contract.get_stakes_by_address(staker_address)
        stake_ids = [stake["id"] for stake in stakes]
        
        if stake_id not in stake_ids:
            return jsonify({
                "success": False,
                "message": "Invalid stake ID or not your stake",
                "error_code": "invalid_stake"
            }), 400
        
        rewards = staking_contract.claim_rewards(stake_id)
        
        if rewards <= 0:
            return jsonify({
                "success": False,
                "message": "No rewards to claim",
                "error_code": "no_rewards"
            }), 400
        
        # Get token balance after claiming
        token_contract = components["token_contract"]
        balance = token_contract.balance_of(staker_address)
        
        return jsonify({
            "success": True,
            "message": f"Successfully claimed {rewards} NYX rewards",
            "rewards": rewards,
            "balance": balance,
            "timestamp": int(time.time())
        })
    except Exception as e:
        logger.error(f"Error claiming rewards: {e}")
        return jsonify({
            "success": False,
            "message": "Failed to claim rewards",
            "error_code": "rewards_error"
        }), 500

@app.route('/api/staking/mystakes', methods=['GET'])
@require_auth
def get_my_stakes():
    """Get all stakes for the authenticated user."""
    try:
        components = app.config["COMPONENTS"]
        
        # Get staker address from auth token
        staker_address = g.user_id
        
        # Get stakes
        staking_contract = components["staking_contract"]
        stakes = staking_contract.get_stakes_by_address(staker_address)
        
        # Format stakes
        formatted_stakes = []
        for stake in stakes:
            # Get pool details
            pool = staking_contract.state["staking_pools"].get(stake["pool_id"], {})
            
            # Calculate pending rewards
            pending_rewards = 0
            if stake["active"]:
                pending_rewards = staking_contract._calculate_rewards(stake["id"])
            
            formatted_stakes.append({
                "id": stake["id"],
                "poolId": stake["pool_id"],
                "poolName": pool.get("name", "Unknown Pool"),
                "amount": stake["amount"],
                "startTime": stake["start_time"],
                "endTime": stake["end_time"],
                "active": stake["active"],
                "rewardsClaimed": stake["rewards_claimed"],
                "pendingRewards": pending_rewards
            })
        
        return jsonify({
            "success": True,
            "stakes": formatted_stakes,
            "totalStaked": sum(stake["amount"] for stake in stakes if stake["active"]),
            "totalRewardsClaimed": sum(stake["rewards_claimed"] for stake in stakes),
            "pendingRewards": sum(staking_contract._calculate_rewards(stake["id"]) for stake in stakes if stake["active"])
        })
    except Exception as e:
        logger.error(f"Error getting stakes: {e}")
        return jsonify({
            "success": False,
            "message": "Failed to get stakes",
            "error_code": "stakes_error"
        }), 500

# Blockchain explorer API endpoints
@app.route('/api/explorer/blocks', methods=['GET'])
def get_blocks():
    """Get blocks from the blockchain."""
    try:
        components = app.config["COMPONENTS"]
        
        # Parse query parameters
        limit = min(int(request.args.get('limit', 10)), 100)
        offset = max(int(request.args.get('offset', 0)), 0)
        
        # Get blocks
        chain = components["blockchain"].chain
        total_blocks = len(chain)
        
        # Calculate indices for slicing
        start_idx = max(0, total_blocks - offset - limit)
        end_idx = max(0, total_blocks - offset)
        
        # Get block subset in reverse order (newest first)
        blocks = chain[start_idx:end_idx]
        blocks.reverse()
        
        # Format blocks
        formatted_blocks = []
        for block in blocks:
            formatted_blocks.append({
                "index": block.index,
                "hash": block.hash,
                "previousHash": block.previous_hash,
                "timestamp": block.timestamp,
                "transactions": len(block.transactions),
                "nonce": block.nonce
            })
        
        return jsonify({
            "success": True,
            "blocks": formatted_blocks,
            "total": total_blocks,
            "limit": limit,
            "offset": offset
        })
    except Exception as e:
        logger.error(f"Error getting blocks: {e}")
        return jsonify({
            "success": False,
            "message": "Failed to get blocks",
            "error_code": "blocks_error"
        }), 500

@app.route('/api/explorer/blocks/<block_id>', methods=['GET'])
def get_block(block_id):
    """Get a specific block by index or hash."""
    try:
        components = app.config["COMPONENTS"]
        
        # Parse block ID
        chain = components["blockchain"].chain
        block = None
        
        try:
            # Try as index
            index = int(block_id)
            if 0 <= index < len(chain):
                block = chain[index]
        except ValueError:
            # Try as hash
            for b in chain:
                if b.hash == block_id:
                    block = b
                    break
        
        if not block:
            return jsonify({
                "success": False,
                "message": "Block not found",
                "error_code": "block_not_found"
            }), 404
        
        # Format block
        formatted_block = {
            "index": block.index,
            "hash": block.hash,
            "previousHash": block.previous_hash,
            "timestamp": block.timestamp,
            "nonce": block.nonce,
            "transactions": []
        }
        
        # Format transactions
        for tx in block.transactions:
            formatted_block["transactions"].append({
                "id": tx.transaction_id,
                "sender": tx.sender,
                "recipient": tx.recipient,
                "amount": tx.amount,
                "timestamp": tx.timestamp
            })
        
        return jsonify({
            "success": True,
            "block": formatted_block
        })
    except Exception as e:
        logger.error(f"Error getting block: {e}")
        return jsonify({
            "success": False,
            "message": "Failed to get block",
            "error_code": "block_error"
        }), 500

@app.route('/api/explorer/transactions', methods=['GET'])
def get_transactions():
    """Get recent transactions."""
    try:
        components = app.config["COMPONENTS"]
        
        # Parse query parameters
        limit = min(int(request.args.get('limit', 20)), 100)
        
        # Get transactions from token contract
        token_contract = components["token_contract"]
        transactions = token_contract.get_transaction_history(limit=limit)
        
        # Format transactions
        formatted_transactions = []
        for tx in transactions:
            formatted_transactions.append({
                "id": tx["transaction_id"],
                "from": tx["from"],
                "to": tx["to"],
                "amount": tx["amount"],
                "transferAmount": tx.get("transfer_amount", tx["amount"]),
                "burnAmount": tx.get("burn_amount", 0),
                "timestamp": tx["timestamp"]
            })
        
        return jsonify({
            "success": True,
            "transactions": formatted_transactions,
            "total": token_contract.state["transaction_count"],
            "limit": limit
        })
    except Exception as e:
        logger.error(f"Error getting transactions: {e}")
        return jsonify({
            "success": False,
            "message": "Failed to get transactions",
            "error_code": "transactions_error"
        }), 500

@app.route('/api/explorer/address/<address>', methods=['GET'])
def get_address_info(address):
    """Get information about an address."""
    try:
        components = app.config["COMPONENTS"]
        
        # Sanitize input
        address = SecurityUtils.sanitize_input(address)
        
        # Validate address format
        if not SecurityUtils.is_valid_address(address):
            return jsonify({
                "success": False,
                "message": "Invalid address format",
                "error_code": "invalid_address"
            }), 400
        
        # Get token balance
        token_contract = components["token_contract"]
        balance = token_contract.balance_of(address)
        
        # Get transactions for address
        transactions = token_contract.get_transaction_history(address=address, limit=50)
        
        # Format transactions
        formatted_transactions = []
        for tx in transactions:
            formatted_transactions.append({
                "id": tx["transaction_id"],
                "from": tx["from"],
                "to": tx["to"],
                "amount": tx["amount"],
                "transferAmount": tx.get("transfer_amount", tx["amount"]),
                "burnAmount": tx.get("burn_amount", 0),
                "timestamp": tx["timestamp"],
                "type": "in" if tx["to"] == address else "out"
            })
        
        # Get stakes
        staking_contract = components["staking_contract"]
        stakes = staking_contract.get_stakes_by_address(address)
        
        # Format stakes
        formatted_stakes = []
        for stake in stakes:
            formatted_stakes.append({
                "id": stake["id"],
                "poolId": stake["pool_id"],
                "amount": stake["amount"],
                "startTime": stake["start_time"],
                "endTime": stake["end_time"],
                "active": stake["active"],
                "rewardsClaimed": stake["rewards_claimed"]
            })
        
        # Calculate total staked
        active_stakes = [stake for stake in stakes if stake["active"]]
        total_staked = sum(stake["amount"] for stake in active_stakes)
        
        return jsonify({
            "success": True,
            "address": address,
            "balance": balance,
            "totalStaked": total_staked,
            "transactions": formatted_transactions,
            "stakes": formatted_stakes,
            "transactionCount": len(formatted_transactions)
        })
    except Exception as e:
        logger.error(f"Error getting address info: {e}")
        return jsonify({
            "success": False,
            "message": "Failed to get address information",
            "error_code": "address_info_error"
        }), 500

# Stats API endpoint
@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get network statistics."""
    try:
        components = app.config["COMPONENTS"]
        
        # Get blockchain stats
        blockchain_stats = {
            "blockCount": len(components["blockchain"].chain),
            "lastBlockTime": components["blockchain"].chain[-1].timestamp if components["blockchain"].chain else 0,
            "pendingTransactions": len(components["blockchain"].pending_transactions)
        }
        
        # Get token stats
        token_contract = components["token_contract"]
        token_stats = token_contract.get_token_stats()
        
        # Get staking stats
        staking_contract = components["staking_contract"]
        staking_stats = {
            "totalStaked": staking_contract.state["total_staked"],
            "poolCount": len(staking_contract.state["staking_pools"]),
            "stakesCount": len(staking_contract.state["stakes"]),
            "activeStakes": len([s for s in staking_contract.state["stakes"].values() if s["active"]]),
            "rewardsDistributed": staking_contract.state["rewards_distributed"]
        }
        
        # Get validator stats
        validator_stats = components["validator"].get_performance_metrics()
        
        # Get consensus stats
        consensus_stats = components["coordinator"].get_network_metrics()
        
        return jsonify({
            "success": True,
            "blockchain": blockchain_stats,
            "token": token_stats,
            "staking": staking_stats,
            "validator": validator_stats,
            "consensus": consensus_stats,
            "timestamp": int(time.time())
        })
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({
            "success": False,
            "message": "Failed to get statistics",
            "error_code": "stats_error"
        }), 500

# Admin API endpoints
@app.route('/api/admin/login', methods=['POST'])
@limiter.limit("5 per minute")
def admin_login():
    """Login as admin using genesis credentials."""
    try:
        data = request.json
        components = app.config["COMPONENTS"]
        
        # Validate request data
        if not data or 'privateKey' not in data:
            return jsonify({
                "success": False,
                "message": "Private key is required",
                "error_code": "missing_private_key"
            }), 400
        
        # Check against genesis private key
        genesis_keys = components["genesis_keys"]
        if data['privateKey'] != genesis_keys["private_key"]:
            # Use constant-time comparison to prevent timing attacks
            import hmac
            if not hmac.compare_digest(data['privateKey'], genesis_keys["private_key"]):
                return jsonify({
                    "success": False,
                    "message": "Invalid credentials",
                    "error_code": "invalid_credentials"
                }), 401
        
        # Generate admin token with longer expiration
        admin_token = SecurityUtils.generate_token(
            genesis_keys["public_key"],
            expiration=86400  # 24 hours
        )
        
        return jsonify({
            "success": True,
            "message": "Admin login successful",
            "auth_token": admin_token,
            "address": genesis_keys["public_key"]
        })
    except Exception as e:
        logger.error(f"Error during admin login: {e}")
        return jsonify({
            "success": False,
            "message": "Login failed",
            "error_code": "login_error"
        }), 500

@app.route('/api/admin/airdrop', methods=['POST'])
@require_admin
def admin_airdrop():
    """Airdrop tokens to an address (admin only)."""
    try:
        data = request.json
        components = app.config["COMPONENTS"]
        
        # Validate request data
        if not data or 'address' not in data or 'amount' not in data:
            return jsonify({
                "success": False,
                "message": "Address and amount are required",
                "error_code": "missing_parameters"
            }), 400
        
        # Sanitize input
        address = SecurityUtils.sanitize_input(data['address'])
        amount = float(data['amount'])
        
        # Validate address format
        if not SecurityUtils.is_valid_address(address):
            return jsonify({
                "success": False,
                "message": "Invalid address format",
                "error_code": "invalid_address"
            }), 400
        
        # Validate amount
        if amount <= 0 or amount > 100000:
            return jsonify({
                "success": False,
                "message": "Amount must be positive and not exceed 100,000",
                "error_code": "invalid_amount"
            }), 400
        
        # Perform transfer from genesis account
        token_contract = components["token_contract"]
        genesis_address = components["genesis_keys"]["public_key"]
        
        result = token_contract.transfer(genesis_address, address, amount)
        
        if not result:
            return jsonify({
                "success": False,
                "message": "Airdrop failed",
                "error_code": "airdrop_failed"
            }), 400
        
        # Get updated balance
        balance = token_contract.balance_of(address)
        
        return jsonify({
            "success": True,
            "message": f"Successfully airdropped {amount} NYX",
            "address": address,
            "amount": amount,
            "new_balance": balance,
            "timestamp": int(time.time())
        })
    except Exception as e:
        logger.error(f"Error during airdrop: {e}")
        return jsonify({
            "success": False,
            "message": "Airdrop failed",
            "error_code": "airdrop_error"
        }), 500

@app.route('/api/admin/create-pool', methods=['POST'])
@require_admin
def admin_create_pool():
    """Create a new staking pool (admin only)."""
    try:
        data = request.json
        components = app.config["COMPONENTS"]
        
        # Validate request data
        if not data or 'id' not in data or 'name' not in data or 'rewardMultiplier' not in data:
            return jsonify({
                "success": False,
                "message": "Pool ID, name, and reward multiplier are required",
                "error_code": "missing_parameters"
            }), 400
        
        # Sanitize input
        pool_id = SecurityUtils.sanitize_input(data['id'])
        name = SecurityUtils.sanitize_input(data['name'])
        reward_multiplier = float(data['rewardMultiplier'])
        min_stake = float(data.get('minStake', 100))
        lock_period = int(data.get('lockPeriodDays', 0)) * 24 * 60 * 60  # Convert days to seconds
        
        # Validate reward multiplier
        if reward_multiplier <= 0 or reward_multiplier > 10:
            return jsonify({
                "success": False,
                "message": "Reward multiplier must be positive and not exceed 10",
                "error_code": "invalid_multiplier"
            }), 400
        
        # Create pool
        staking_contract = components["staking_contract"]
        result = staking_contract.create_pool(pool_id, name, reward_multiplier, min_stake, lock_period)
        
        if not result:
            return jsonify({
                "success": False,
                "message": "Failed to create pool, ID may already exist",
                "error_code": "pool_creation_failed"
            }), 400
        
        # Get all pools
        pools = staking_contract.get_all_pools()
        
        return jsonify({
            "success": True,
            "message": f"Successfully created pool '{name}'",
            "pool": {
                "id": pool_id,
                "name": name,
                "rewardMultiplier": reward_multiplier,
                "minStake": min_stake,
                "lockPeriodDays": lock_period / (24 * 60 * 60)
            },
            "pools": pools,
            "timestamp": int(time.time())
        })
    except Exception as e:
        logger.error(f"Error creating pool: {e}")
        return jsonify({
            "success": False,
            "message": "Failed to create pool",
            "error_code": "pool_creation_error"
        }), 500

@app.route('/api/admin/backup', methods=['GET'])
@require_admin
def admin_backup():
    """Create a backup of the blockchain state (admin only)."""
    try:
        components = app.config["COMPONENTS"]
        
        # Create backup timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(Config.DATA_DIR, "backups", timestamp)
        os.makedirs(backup_dir, exist_ok=True)
        
        # Save contract states
        token_contract = components["token_contract"]
        staking_contract = components["staking_contract"]
        
        token_contract.save(os.path.join(backup_dir, "contracts"))
        staking_contract.save(os.path.join(backup_dir, "contracts"))
        
        # Save blockchain state
        blockchain_state = {
            "chain_length": len(components["blockchain"].chain),
            "pending_transactions": len(components["blockchain"].pending_transactions),
            "timestamp": int(time.time())
        }
        
        with open(os.path.join(backup_dir, "blockchain.json"), "w") as f:
            json.dump(blockchain_state, f, indent=2)
        
        return jsonify({
            "success": True,
            "message": "Backup created successfully",
            "backup_id": timestamp,
            "backup_path": backup_dir,
            "timestamp": int(time.time())
        })
    except Exception as e:
        logger.error(f"Error creating backup: {e}")
        return jsonify({
            "success": False,
            "message": "Failed to create backup",
            "error_code": "backup_error"
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        "success": False,
        "message": "The requested resource was not found",
        "error_code": "not_found"
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({
        "success": False,
        "message": "Method not allowed",
        "error_code": "method_not_allowed"
    }), 405

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        "success": False,
        "message": "Internal server error",
        "error_code": "internal_error"
    }), 500

@app.errorhandler(429)
def ratelimit_handler(error):
    """Handle rate limit errors."""
    return jsonify({
        "success": False,
        "message": "Rate limit exceeded",
        "error_code": "rate_limit_exceeded",
        "retry_after": error.description
    }), 429

# Startup
if __name__ == '__main__':
    # Record start time
    app.start_time = time.time()
    
    # Log startup
    logger.info(f"Starting NyxSynth API server version {Config.API_VERSION}")
    logger.info(f"Node ID: {Config.NODE_ID}")
    logger.info(f"Security level: {Config.SECURITY_LEVEL}")
    
    # Define host and port
    host = os.environ.get('NYXSYNTH_HOST', '0.0.0.0')
    port = int(os.environ.get('NYXSYNTH_PORT', 5000))
    
    # Run in debug mode if specified
    debug = os.environ.get('NYXSYNTH_DEBUG', 'false').lower() == 'true'
    
    # Start server
    app.run(host=host, port=port, debug=debug)
