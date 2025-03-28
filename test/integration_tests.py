import os
import sys
import unittest
import json
import time
import requests
import threading
import subprocess
from contextlib import contextmanager

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

class IntegrationTestConfig:
    """Configuration for integration tests."""
    TEMP_DATA_DIR = "tests/temp_data"
    API_PORT = 5050
    API_URL = f"http://localhost:{API_PORT}/api"
    
    @classmethod
    def setup(cls):
        """Set up test environment."""
        os.environ["NYXSYNTH_DATA_DIR"] = cls.TEMP_DATA_DIR
        os.environ["NYXSYNTH_PORT"] = str(cls.API_PORT)
        os.environ["NYXSYNTH_DEBUG"] = "false"
        os.makedirs(cls.TEMP_DATA_DIR, exist_ok=True)
    
    @classmethod
    def cleanup(cls):
        """Clean up test environment."""
        import shutil
        
        if os.path.exists(cls.TEMP_DATA_DIR):
            shutil.rmtree(cls.TEMP_DATA_DIR)


@contextmanager
def run_api_server():
    """
    Context manager to start and stop the API server.
    Yields the server process.
    """
    # Start the server in a subprocess
    print("Starting API server...")
    
    # Use the secured server script
    process = subprocess.Popen(
        [sys.executable, "api/secured_server.py"],
        env=os.environ,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    try:
        # Wait for server to start
        time.sleep(3)
        
        # Check if server is running
        try:
            response = requests.get(f"{IntegrationTestConfig.API_URL}/health")
            if response.status_code != 200:
                raise Exception(f"API server failed to start. Status code: {response.status_code}")
            
            print("API server started successfully!")
        except requests.exceptions.ConnectionError:
            stdout, stderr = process.communicate(timeout=1)
            raise Exception(f"API server failed to start. Error: {stderr}")
        
        # Yield server process
        yield process
    finally:
        # Terminate server
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        
        print("API server stopped.")


class ComponentIntegrationTests(unittest.TestCase):
    """Integration tests for core blockchain components."""
    
    def setUp(self):
        """Set up test environment."""
        self.crypto = HardenedQuantumCrypto({
            'security_level': 5
        })
        self.blockchain = Blockchain()
        self.validator = EnhancedNeuralValidator()
        self.coordinator = EnhancedBioluminescentCoordinator()
        self.registry = EnhancedSymbioticRegistry()
    
    def test_crypto_validator_integration(self):
        """Test integration between crypto and validator components."""
        # Generate test keys
        keypair = self.crypto.generate_keypair()
        
        # Create a test transaction
        tx_data = {
            "sender": keypair["public_key"],
            "recipient": "0x" + "1" * 64,
            "amount": 100.0,
            "timestamp": time.time()
        }
        
        # Convert to string and sign
        tx_str = json.dumps(tx_data)
        signature = self.crypto.sign(tx_str, keypair["private_key"])
        
        # Verify signature
        is_valid = self.crypto.verify(tx_str, signature, keypair["public_key"])
        self.assertTrue(is_valid, "Signature verification failed")
        
        # Generate neural pattern
        pattern = self.validator.generate_pattern([tx_data])
        self.assertEqual(len(pattern), 64, "Pattern has incorrect length")
        
        # Emit pattern to coordinator
        result = self.coordinator.emit_pattern(pattern)
        self.assertTrue(result["success"], "Pattern emission failed")
        
        # Verify pattern synchronization
        sync_score = self.coordinator.get_synchronization_score(pattern)
        self.assertGreaterEqual(sync_score, 0.5, "Pattern synchronization score too low")
    
    def test_blockchain_contract_integration(self):
        """Test integration between blockchain and contract components."""
        # Generate test keys
        genesis_keys = self.crypto.generate_keypair()
        user_keys = self.crypto.generate_keypair()
        
        # Create token contract
        token_contract = EnhancedTokenContract(genesis_keys["public_key"], 1000000)
        
        # Register contract
        self.registry.register_contract(token_contract)
        
        # Create staking contract
        staking_contract = EnhancedStakingContract(genesis_keys["public_key"], token_contract.address)
        self.registry.register_contract(staking_contract)
        
        # Create relationship
        self.registry.create_relationship(
            token_contract.address,
            staking_contract.address,
            RelationshipType.RESOURCE_SHARING
        )
        
        # Transfer tokens to user
        result = token_contract.transfer(
            genesis_keys["public_key"],
            user_keys["public_key"],
            1000
        )
        self.assertTrue(result, "Token transfer failed")
        
        # Verify balance
        balance = token_contract.balance_of(user_keys["public_key"])
        self.assertEqual(balance, 995, "Incorrect balance after transfer (5 tokens should be burned)")
        
        # Create staking pool
        result = staking_contract.create_pool("test_pool", "Test Pool", 1.0, 100)
        self.assertTrue(result, "Failed to create staking pool")
        
        # Stake tokens
        result = staking_contract.stake(user_keys["public_key"], "test_pool", 500)
        self.assertTrue(result, "Failed to stake tokens")
        
        # Verify staked amount
        stakes = staking_contract.get_stakes_by_address(user_keys["public_key"])
        self.assertEqual(len(stakes), 1, "Incorrect number of stakes")
        self.assertEqual(stakes[0]["amount"], 500, "Incorrect staked amount")
        
        # Verify remaining balance
        balance = token_contract.balance_of(user_keys["public_key"])
        self.assertEqual(balance, 495, "Incorrect balance after staking")


class APIIntegrationTests(unittest.TestCase):
    """Integration tests for API endpoints."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        IntegrationTestConfig.setup()
        cls.server_process = None
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment after all tests."""
        IntegrationTestConfig.cleanup()
    
    def setUp(self):
        """Set up test environment for each test."""
        self.api_url = IntegrationTestConfig.API_URL
        self.wallet = None
        self.auth_token = None
    
    def _create_wallet(self):
        """Helper to create a test wallet."""
        response = requests.post(f"{self.api_url}/wallet/create")
        self.assertEqual(response.status_code, 200, "Failed to create wallet")
        
        data = response.json()
        self.assertTrue(data["success"], "Wallet creation returned failure")
        
        self.wallet = data["wallet"]
        self.auth_token = data["auth_token"]
    
    def _get_auth_header(self):
        """Helper to get authorization header."""
        if not self.auth_token:
            self._create_wallet()
        
        return {"Authorization": f"Bearer {self.auth_token}"}
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = requests.get(f"{self.api_url}/health")
        self.assertEqual(response.status_code, 200, "Health check failed")
        
        data = response.json()
        self.assertTrue(data["success"], "Health check returned failure")
        self.assertEqual(data["status"], "healthy", "Health status is not healthy")
    
    def test_wallet_endpoints(self):
        """Test wallet-related endpoints."""
        # Create wallet
        self._create_wallet()
        self.assertIsNotNone(self.wallet, "Wallet was not created")
        self.assertIn("address", self.wallet, "Wallet missing address")
        self.assertIn("privateKey", self.wallet, "Wallet missing private key")
        
        # Check balance
        response = requests.get(f"{self.api_url}/wallet/balance/{self.wallet['address']}")
        self.assertEqual(response.status_code, 200, "Failed to get balance")
        
        data = response.json()
        self.assertTrue(data["success"], "Balance query returned failure")
        self.assertEqual(data["address"], self.wallet["address"], "Balance address mismatch")
        self.assertEqual(data["symbol"], "NYX", "Incorrect token symbol")
        
        # Create second wallet
        response = requests.post(f"{self.api_url}/wallet/create")
        self.assertEqual(response.status_code, 200, "Failed to create second wallet")
        
        data = response.json()
        second_wallet = data["wallet"]
        
        # Transfer tokens
        transfer_data = {
            "to": second_wallet["address"],
            "amount": 10
        }
        
        response = requests.post(
            f"{self.api_url}/wallet/transfer",
            json=transfer_data,
            headers=self._get_auth_header()
        )
        
        self.assertEqual(response.status_code, 200, "Transfer failed")
        data = response.json()
        self.assertTrue(data["success"], "Transfer returned failure")
        
        # Check recipient balance
        response = requests.get(f"{self.api_url}/wallet/balance/{second_wallet['address']}")
        data = response.json()
        self.assertGreaterEqual(data["balance"], 9.9, "Recipient didn't receive tokens")
    
    def test_staking_endpoints(self):
        """Test staking-related endpoints."""
        # Create wallet
        self._create_wallet()
        
        # Get staking pools
        response = requests.get(f"{self.api_url}/staking/pools")
        self.assertEqual(response.status_code, 200, "Failed to get staking pools")
        
        data = response.json()
        self.assertTrue(data["success"], "Staking pools query returned failure")
        self.assertGreaterEqual(len(data["pools"]), 1, "No staking pools found")
        
        # Stake tokens
        pool_id = data["pools"][0]["id"]
        stake_data = {
            "poolId": pool_id,
            "amount": 10
        }
        
        response = requests.post(
            f"{self.api_url}/staking/stake",
            json=stake_data,
            headers=self._get_auth_header()
        )
        
        self.assertEqual(response.status_code, 200, "Staking failed")
        data = response.json()
        self.assertTrue(data["success"], "Staking returned failure")
        
        # Get my stakes
        response = requests.get(
            f"{self.api_url}/staking/mystakes",
            headers=self._get_auth_header()
        )
        
        self.assertEqual(response.status_code, 200, "Failed to get stakes")
        data = response.json()
        self.assertTrue(data["success"], "Stakes query returned failure")
        self.assertEqual(len(data["stakes"]), 1, "Incorrect number of stakes")
        
        # Get stake ID
        stake_id = data["stakes"][0]["id"]
        
        # Unstake tokens
        unstake_data = {
            "stakeId": stake_id
        }
        
        response = requests.post(
            f"{self.api_url}/staking/unstake",
            json=unstake_data,
            headers=self._get_auth_header()
        )
        
        self.assertEqual(response.status_code, 200, "Unstaking failed")
        data = response.json()
        self.assertTrue(data["success"], "Unstaking returned failure")
    
    def test_explorer_endpoints(self):
        """Test blockchain explorer endpoints."""
        # Get blocks
        response = requests.get(f"{self.api_url}/explorer/blocks")
        self.assertEqual(response.status_code, 200, "Failed to get blocks")
        
        data = response.json()
        self.assertTrue(data["success"], "Blocks query returned failure")
        
        # Get single block (genesis)
        response = requests.get(f"{self.api_url}/explorer/blocks/0")
        self.assertEqual(response.status_code, 200, "Failed to get genesis block")
        
        data = response.json()
        self.assertTrue(data["success"], "Block query returned failure")
        self.assertEqual(data["block"]["index"], 0, "Incorrect block index")
        
        # Get transactions
        response = requests.get(f"{self.api_url}/explorer/transactions")
        self.assertEqual(response.status_code, 200, "Failed to get transactions")
        
        data = response.json()
        self.assertTrue(data["success"], "Transactions query returned failure")


class End2EndTests(unittest.TestCase):
    """End-to-end tests using running API server."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        IntegrationTestConfig.setup()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment after all tests."""
        IntegrationTestConfig.cleanup()
    
    def test_full_workflow(self):
        """Test complete user workflow from wallet creation to staking and unstaking."""
        with run_api_server() as server:
            api_url = IntegrationTestConfig.API_URL
            
            # 1. Create two wallets
            response = requests.post(f"{api_url}/wallet/create")
            self.assertEqual(response.status_code, 200)
            wallet1 = response.json()["wallet"]
            auth_token1 = response.json()["auth_token"]
            
            response = requests.post(f"{api_url}/wallet/create")
            self.assertEqual(response.status_code, 200)
            wallet2 = response.json()["wallet"]
            
            # 2. Check initial balances
            response = requests.get(f"{api_url}/wallet/balance/{wallet1['address']}")
            self.assertEqual(response.status_code, 200)
            initial_balance1 = response.json()["balance"]
            self.assertGreater(initial_balance1, 0, "Wallet should have initial tokens")
            
            # 3. Transfer some tokens
            transfer_amount = 25
            transfer_data = {
                "to": wallet2["address"],
                "amount": transfer_amount
            }
            
            response = requests.post(
                f"{api_url}/wallet/transfer",
                json=transfer_data,
                headers={"Authorization": f"Bearer {auth_token1}"}
            )
            self.assertEqual(response.status_code, 200)
            
            # 4. Verify balances after transfer
            response = requests.get(f"{api_url}/wallet/balance/{wallet2['address']}")
            self.assertEqual(response.status_code, 200)
            balance2 = response.json()["balance"]
            expected_received = transfer_amount * 0.995  # 0.5% burn rate
            self.assertAlmostEqual(balance2, expected_received, delta=0.1)
            
            response = requests.get(f"{api_url}/wallet/balance/{wallet1['address']}")
            self.assertEqual(response.status_code, 200)
            new_balance1 = response.json()["balance"]
            self.assertAlmostEqual(new_balance1, initial_balance1 - transfer_amount, delta=0.1)
            
            # 5. Get staking pools
            response = requests.get(f"{api_url}/staking/pools")
            self.assertEqual(response.status_code, 200)
            pools = response.json()["pools"]
            self.assertGreaterEqual(len(pools), 1, "No staking pools available")
            
            # Find pool with lowest minimum stake
            pool = min(pools, key=lambda p: p["min_stake"])
            
            # 6. Stake tokens
            stake_amount = max(10, pool["min_stake"])
            stake_data = {
                "poolId": pool["id"],
                "amount": stake_amount
            }
            
            response = requests.post(
                f"{api_url}/staking/stake",
                json=stake_data,
                headers={"Authorization": f"Bearer {auth_token1}"}
            )
            self.assertEqual(response.status_code, 200)
            
            # 7. Verify stakes
            response = requests.get(
                f"{api_url}/staking/mystakes",
                headers={"Authorization": f"Bearer {auth_token1}"}
            )
            self.assertEqual(response.status_code, 200)
            stakes = response.json()["stakes"]
            self.assertEqual(len(stakes), 1, "Stake not created")
            stake_id = stakes[0]["id"]
            
            # 8. Check updated balance
            response = requests.get(f"{api_url}/wallet/balance/{wallet1['address']}")
            self.assertEqual(response.status_code, 200)
            staked_balance = response.json()["balance"]
            self.assertAlmostEqual(staked_balance, new_balance1 - stake_amount, delta=0.1)
            
            # 9. Unstake tokens
            unstake_data = {
                "stakeId": stake_id
            }
            
            response = requests.post(
                f"{api_url}/staking/unstake",
                json=unstake_data,
                headers={"Authorization": f"Bearer {auth_token1}"}
            )
            self.assertEqual(response.status_code, 200)
            
            # 10. Verify final balance
            response = requests.get(f"{api_url}/wallet/balance/{wallet1['address']}")
            self.assertEqual(response.status_code, 200)
            final_balance = response.json()["balance"]
            
            # Balance should be close to what it was before staking
            # It might be slightly higher due to rewards, but for a short test rewards will be minimal
            self.assertGreaterEqual(final_balance, staked_balance, "Balance didn't increase after unstaking")
            
            # 11. Check network stats
            response = requests.get(f"{api_url}/stats")
            self.assertEqual(response.status_code, 200)
            stats = response.json()
            self.assertTrue(stats["success"], "Stats query failed")
            self.assertIn("blockchain", stats, "Missing blockchain stats")
            self.assertIn("token", stats, "Missing token stats")
            self.assertIn("staking", stats, "Missing staking stats")


class LoadTests(unittest.TestCase):
    """Load testing for the API server."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        IntegrationTestConfig.setup()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment after all tests."""
        IntegrationTestConfig.cleanup()
    
    def test_concurrent_requests(self):
        """Test API under concurrent load."""
        with run_api_server() as server:
            api_url = IntegrationTestConfig.API_URL
            
            # Create test wallets
            response = requests.post(f"{api_url}/wallet/create")
            self.assertEqual(response.status_code, 200)
            wallet = response.json()["wallet"]
            auth_token = response.json()["auth_token"]
            
            # Create target wallets for transfers
            target_wallets = []
            for _ in range(5):
                response = requests.post(f"{api_url}/wallet/create")
                self.assertEqual(response.status_code, 200)
                target_wallets.append(response.json()["wallet"]["address"])
            
            # Test parameters
            num_threads = 10
            requests_per_thread = 5
            
            # Track results
            results = {
                "success": 0,
                "failed": 0
            }
            
            def make_requests(thread_id):
                """Make API requests in a thread."""
                for i in range(requests_per_thread):
                    try:
                        # Alternate between different endpoint types
                        if i % 3 == 0:
                            # Balance check
                            response = requests.get(f"{api_url}/wallet/balance/{wallet['address']}")
                        elif i % 3 == 1:
                            # Pool listing
                            response = requests.get(f"{api_url}/staking/pools")
                        else:
                            # Transfer (with rate limiting consideration)
                            transfer_data = {
                                "to": target_wallets[i % len(target_wallets)],
                                "amount": 1  # Small amount to avoid depleting balance
                            }
                            response = requests.post(
                                f"{api_url}/wallet/transfer",
                                json=transfer_data,
                                headers={"Authorization": f"Bearer {auth_token}"}
                            )
                        
                        if response.status_code == 200:
                            results["success"] += 1
                        else:
                            results["failed"] += 1
                            print(f"Request failed: {response.status_code}, {response.text}")
                    except Exception as e:
                        results["failed"] += 1
                        print(f"Request error: {e}")
            
            # Create and start threads
            threads = []
            for i in range(num_threads):
                thread = threading.Thread(target=make_requests, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Check results
            total_requests = num_threads * requests_per_thread
            success_rate = (results["success"] / total_requests) * 100
            
            print(f"Load test results:")
            print(f"  Total requests: {total_requests}")
            print(f"  Successful: {results['success']} ({success_rate:.2f}%)")
            print(f"  Failed: {results['failed']}")
            
            # Assertions
            self.assertGreaterEqual(success_rate, 90, "Success rate below 90%")


def run_tests():
    """Run all integration tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(ComponentIntegrationTests))
    
    # Only run API tests if explicitly requested (they take longer)
    if "--api" in sys.argv:
        suite.addTests(loader.loadTestsFromTestCase(APIIntegrationTests))
    
    # Only run end-to-end tests if explicitly requested (they take much longer)
    if "--e2e" in sys.argv:
        suite.addTests(loader.loadTestsFromTestCase(End2EndTests))
    
    # Only run load tests if explicitly requested
    if "--load" in sys.argv:
        suite.addTests(loader.loadTestsFromTestCase(LoadTests))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result


if __name__ == "__main__":
    # Set up environment
    IntegrationTestConfig.setup()
    
    try:
        # Run tests
        print("Running NyxSynth Integration Tests")
        print("=================================")
        print("Available options:")
        print("  --api    Run API integration tests")
        print("  --e2e    Run end-to-end tests")
        print("  --load   Run load tests")
        print("=================================")
        
        result = run_tests()
        
        # Exit with appropriate code
        sys.exit(not result.wasSuccessful())
    finally:
        # Clean up
        IntegrationTestConfig.cleanup()
