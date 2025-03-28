import json
import time
import uuid
import hashlib
import inspect
from typing import List, Dict, Any, Union, Callable, Optional, Set, Tuple
from threading import Lock
from enum import Enum, auto
from dataclasses import dataclass, field, asdict
import os
import copy

class RelationshipType(Enum):
    """Types of symbiotic relationships between contracts."""
    MUTUALISM = auto()       # Both contracts benefit
    COMMENSALISM = auto()    # One contract benefits, other unaffected
    DEPENDENCY = auto()      # One contract depends on another
    RESOURCE_SHARING = auto() # Contracts share resources
    DATA_EXCHANGE = auto()   # Contracts exchange data
    REGULATORY = auto()      # One contract regulates/validates another
    COMPOSITE = auto()       # Contracts form a composite system


@dataclass
class ContractState:
    """Immutable state snapshot of a contract."""
    contract_id: str
    state: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    version: int = 0
    previous_hash: str = ""
    state_hash: str = ""
    
    def __post_init__(self):
        """Calculate state hash after initialization."""
        if not self.state_hash:
            self.state_hash = self._calculate_hash()
    
    def _calculate_hash(self) -> str:
        """Calculate a deterministic hash of the state."""
        state_data = {
            "contract_id": self.contract_id,
            "state": self.state,
            "timestamp": self.timestamp,
            "version": self.version,
            "previous_hash": self.previous_hash
        }
        serialized = json.dumps(state_data, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return asdict(self)


@dataclass
class Relationship:
    """A relationship between two contracts."""
    source_id: str
    target_id: str
    type: RelationshipType
    permissions: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    creation_time: float = field(default_factory=time.time)
    relationship_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert relationship to dictionary."""
        return {
            "relationship_id": self.relationship_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.type.name,
            "permissions": self.permissions,
            "metadata": self.metadata,
            "creation_time": self.creation_time,
            "active": self.active
        }


class EventType(Enum):
    """Types of events that can be emitted by contracts."""
    STATE_CHANGE = auto()
    RELATIONSHIP_FORMED = auto()
    RELATIONSHIP_TERMINATED = auto()
    FUNCTION_CALL = auto()
    RESOURCE_ALLOCATION = auto()
    ERROR = auto()
    LIFECYCLE = auto()
    REGULATORY = auto()
    CUSTOM = auto()


@dataclass
class ContractEvent:
    """An event emitted by a contract."""
    contract_id: str
    event_type: EventType
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_id": self.event_id,
            "contract_id": self.contract_id,
            "event_type": self.event_type.name,
            "data": self.data,
            "timestamp": self.timestamp
        }


class ResourceType(Enum):
    """Types of resources that can be managed by contracts."""
    COMPUTE = auto()
    MEMORY = auto()
    STORAGE = auto()
    NETWORK = auto()
    TOKEN = auto()
    DATA = auto()
    EXTERNAL_API = auto()
    CUSTOM = auto()


@dataclass
class Resource:
    """A resource that can be allocated and shared between contracts."""
    resource_id: str
    type: ResourceType
    capacity: float
    used: float = 0
    owner_id: str = ""
    shared_with: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def available(self) -> float:
        """Get available resource capacity."""
        return max(0, self.capacity - self.used)
    
    def allocate(self, amount: float, contract_id: str) -> bool:
        """Allocate resource to a contract."""
        if amount <= 0:
            return False
            
        if contract_id != self.owner_id and contract_id not in self.shared_with:
            return False
            
        if amount > self.available():
            return False
            
        self.used += amount
        return True
    
    def deallocate(self, amount: float) -> bool:
        """Deallocate resource."""
        if amount <= 0 or amount > self.used:
            return False
            
        self.used -= amount
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert resource to dictionary."""
        return {
            "resource_id": self.resource_id,
            "type": self.type.name,
            "capacity": self.capacity,
            "used": self.used,
            "owner_id": self.owner_id,
            "shared_with": self.shared_with,
            "metadata": self.metadata
        }


class EnhancedContract:
    """
    Base class for all symbiotic smart contracts with enhanced capabilities.
    
    Features:
    - Immutable state history
    - Fine-grained permissions
    - Event system
    - Resource management
    - Composability
    - Formal verification hooks
    - Regulatory compliance
    """
    
    def __init__(self, owner: str, address: str = None, metadata: Dict[str, Any] = None):
        """Initialize a new enhanced contract."""
        self.owner = owner
        self.address = address or self._generate_address()
        self.contract_id = self.address
        self.created_at = time.time()
        self.metadata = metadata or {}
        
        # Initialize state
        self.state = {}
        self.state_history: List[ContractState] = []
        self.state_version = 0
        self.state_lock = Lock()
        
        # Initialize relationships
        self.relationships: Dict[str, Relationship] = {}
        self.relationship_lock = Lock()
        
        # Initialize events
        self.events: List[ContractEvent] = []
        self.event_handlers: Dict[EventType, List[Callable]] = {
            event_type: [] for event_type in EventType
        }
        
        # Initialize resources
        self.resources: Dict[str, Resource] = {}
        self.resource_lock = Lock()
        
        # Initialize permission system
        self.permissions: Dict[str, Set[str]] = {
            "owner": {"*"}  # Owner has all permissions
        }
        
        # Create initial state
        self._create_state_snapshot()
    
    def _generate_address(self) -> str:
        """Generate a unique contract address."""
        data = f"{self.owner}{time.time()}{uuid.uuid4()}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    def _create_state_snapshot(self) -> ContractState:
        """Create an immutable snapshot of the current state."""
        with self.state_lock:
            # Deep copy the state to ensure immutability
            state_copy = copy.deepcopy(self.state)
            
            # Get previous state hash
            previous_hash = ""
            if self.state_history:
                previous_hash = self.state_history[-1].state_hash
            
            # Create state snapshot
            state_snapshot = ContractState(
                contract_id=self.contract_id,
                state=state_copy,
                version=self.state_version,
                previous_hash=previous_hash
            )
            
            # Add to history
            self.state_history.append(state_snapshot)
            
            # Update version
            self.state_version += 1
            
            return state_snapshot
    
    def update_state(self, new_state: Dict[str, Any], caller: str = None) -> bool:
        """
        Update the contract state.
        
        Args:
            new_state: New state values to merge
            caller: ID of the calling contract/user
            
        Returns:
            True if state was updated successfully
        """
        # Check permissions
        if caller and not self._has_permission(caller, "update_state"):
            self._emit_event(EventType.ERROR, {
                "error": "Permission denied",
                "caller": caller,
                "action": "update_state"
            })
            return False
        
        with self.state_lock:
            # Update state
            old_state = copy.deepcopy(self.state)
            self.state.update(new_state)
            
            # Create state snapshot
            snapshot = self._create_state_snapshot()
            
            # Emit event
            self._emit_event(EventType.STATE_CHANGE, {
                "old_state": old_state,
                "new_state": self.state,
                "changes": new_state,
                "version": snapshot.version
            })
            
            return True
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current contract state."""
        with self.state_lock:
            # Return a copy to prevent direct state modification
            return copy.deepcopy(self.state)
    
    def get_state_history(self) -> List[ContractState]:
        """Get the contract state history."""
        with self.state_lock:
            return self.state_history.copy()
    
    def form_relationship(self, target_contract: 'EnhancedContract', 
                          relationship_type: RelationshipType,
                          permissions: List[str] = None,
                          metadata: Dict[str, Any] = None) -> Optional[Relationship]:
        """
        Form a symbiotic relationship with another contract.
        
        Args:
            target_contract: Contract to form relationship with
            relationship_type: Type of relationship
            permissions: Permissions granted in the relationship
            metadata: Additional relationship metadata
            
        Returns:
            Created relationship or None if failed
        """
        if not target_contract:
            self._emit_event(EventType.ERROR, {
                "error": "Invalid target contract",
                "action": "form_relationship"
            })
            return None
        
        # Default permissions based on relationship type
        if permissions is None:
            permissions = self._default_permissions_for_relationship(relationship_type)
        
        metadata = metadata or {}
        
        with self.relationship_lock:
            # Create relationship
            relationship = Relationship(
                source_id=self.contract_id,
                target_id=target_contract.contract_id,
                type=relationship_type,
                permissions=permissions,
                metadata=metadata
            )
            
            # Store relationship
            self.relationships[relationship.relationship_id] = relationship
            
            # Update permissions
            self._update_permissions(target_contract.contract_id, set(permissions))
            
            # Emit event
            self._emit_event(EventType.RELATIONSHIP_FORMED, {
                "relationship": relationship.to_dict(),
                "target_contract": target_contract.contract_id
            })
            
            # Notify target contract
            target_contract._on_relationship_formed(relationship, self)
            
            return relationship
    
    def _default_permissions_for_relationship(self, relationship_type: RelationshipType) -> List[str]:
        """Get default permissions for a relationship type."""
        defaults = {
            RelationshipType.MUTUALISM: ["call_function", "read_state", "emit_event"],
            RelationshipType.COMMENSALISM: ["read_state"],
            RelationshipType.DEPENDENCY: ["call_function", "read_state"],
            RelationshipType.RESOURCE_SHARING: ["use_resource", "read_state"],
            RelationshipType.DATA_EXCHANGE: ["read_state", "update_partial_state"],
            RelationshipType.REGULATORY: ["read_state", "validate_state"],
            RelationshipType.COMPOSITE: ["call_function", "read_state", "emit_event", "use_resource"]
        }
        
        return defaults.get(relationship_type, [])
    
    def _on_relationship_formed(self, relationship: Relationship, source_contract: 'EnhancedContract') -> None:
        """Handle relationship formation from another contract."""
        # Create reciprocal relationship
        with self.relationship_lock:
            # Store relationship
            self.relationships[relationship.relationship_id] = relationship
            
            # Update permissions
            self._update_permissions(source_contract.contract_id, 
                                    set(relationship.permissions))
            
            # Emit event
            self._emit_event(EventType.RELATIONSHIP_FORMED, {
                "relationship": relationship.to_dict(),
                "source_contract": source_contract.contract_id
            })
    
    def terminate_relationship(self, relationship_id: str) -> bool:
        """
        Terminate a relationship with another contract.
        
        Args:
            relationship_id: ID of the relationship to terminate
            
        Returns:
            True if relationship was terminated successfully
        """
        with self.relationship_lock:
            if relationship_id not in self.relationships:
                return False
            
            relationship = self.relationships[relationship_id]
            
            # Mark as inactive
            relationship.active = False
            
            # Remove permissions
            if relationship.target_id in self.permissions:
                del self.permissions[relationship.target_id]
            
            # Emit event
            self._emit_event(EventType.RELATIONSHIP_TERMINATED, {
                "relationship": relationship.to_dict()
            })
            
            return True
    
    def get_relationships(self) -> List[Relationship]:
        """Get all active relationships."""
        with self.relationship_lock:
            return [r for r in self.relationships.values() if r.active]
    
    def _update_permissions(self, contract_id: str, permissions: Set[str]) -> None:
        """Update permissions for a contract."""
        self.permissions[contract_id] = permissions
    
    def _has_permission(self, caller: str, permission: str) -> bool:
        """Check if a caller has a specific permission."""
        # Owner has all permissions
        if caller == self.owner:
            return True
        
        # Check specific permission
        if caller in self.permissions:
            if "*" in self.permissions[caller]:
                return True
            
            return permission in self.permissions[caller]
        
        return False
    
    def _emit_event(self, event_type: EventType, data: Dict[str, Any]) -> ContractEvent:
        """
        Emit a contract event.
        
        Args:
            event_type: Type of event
            data: Event data
            
        Returns:
            Created event
        """
        event = ContractEvent(
            contract_id=self.contract_id,
            event_type=event_type,
            data=data
        )
        
        # Add to events
        self.events.append(event)
        
        # Call event handlers
        for handler in self.event_handlers[event_type]:
            try:
                handler(event)
            except Exception as e:
                print(f"Error in event handler: {e}")
        
        return event
    
    def add_event_handler(self, event_type: EventType, handler: Callable) -> None:
        """
        Add an event handler.
        
        Args:
            event_type: Type of event to handle
            handler: Function to call when event occurs
        """
        self.event_handlers[event_type].append(handler)
    
    def call_related(self, method_name: str, params: Dict[str, Any] = None,
                    relationship_types: List[RelationshipType] = None) -> List[Any]:
        """
        Call a method on all related contracts.
        
        Args:
            method_name: Method to call
            params: Parameters to pass
            relationship_types: Only call contracts with these relationship types
            
        Returns:
            List of results from related contracts
        """
        params = params or {}
        results = []
        
        for relationship in self.get_relationships():
            # Filter by relationship type
            if relationship_types and relationship.type not in relationship_types:
                continue
            
            # Get target contract
            target_contract = self._get_contract_by_id(relationship.target_id)
            if not target_contract:
                continue
            
            # Check permission
            if not self._has_permission(self.contract_id, "call_function"):
                continue
            
            # Call method
            if hasattr(target_contract, method_name):
                method = getattr(target_contract, method_name)
                
                try:
                    # Emit event
                    self._emit_event(EventType.FUNCTION_CALL, {
                        "target_contract": relationship.target_id,
                        "method": method_name,
                        "params": params
                    })
                    
                    # Call method
                    result = method(caller=self.contract_id, **params)
                    results.append(result)
                except Exception as e:
                    self._emit_event(EventType.ERROR, {
                        "error": str(e),
                        "target_contract": relationship.target_id,
                        "method": method_name,
                        "params": params
                    })
        
        return results
    
    def _get_contract_by_id(self, contract_id: str) -> Optional['EnhancedContract']:
        """
        Get a contract by ID. In a real system, this would use the registry.
        
        This is a placeholder that should be overridden by the registry.
        """
        return None
    
    # Resource management methods
    def create_resource(self, resource_type: ResourceType, capacity: float,
                       metadata: Dict[str, Any] = None) -> Optional[Resource]:
        """
        Create a new resource owned by this contract.
        
        Args:
            resource_type: Type of resource
            capacity: Resource capacity
            metadata: Additional resource metadata
            
        Returns:
            Created resource or None if failed
        """
        with self.resource_lock:
            resource_id = str(uuid.uuid4())
            
            resource = Resource(
                resource_id=resource_id,
                type=resource_type,
                capacity=capacity,
                owner_id=self.contract_id,
                metadata=metadata or {}
            )
            
            self.resources[resource_id] = resource
            
            # Emit event
            self._emit_event(EventType.RESOURCE_ALLOCATION, {
                "action": "create",
                "resource": resource.to_dict()
            })
            
            return resource
    
    def share_resource(self, resource_id: str, target_contract_id: str) -> bool:
        """
        Share a resource with another contract.
        
        Args:
            resource_id: ID of resource to share
            target_contract_id: ID of contract to share with
            
        Returns:
            True if resource was shared successfully
        """
        with self.resource_lock:
            if resource_id not in self.resources:
                return False
            
            resource = self.resources[resource_id]
            
            # Check ownership
            if resource.owner_id != self.contract_id:
                return False
            
            # Add to shared contracts
            if target_contract_id not in resource.shared_with:
                resource.shared_with.append(target_contract_id)
            
            # Emit event
            self._emit_event(EventType.RESOURCE_ALLOCATION, {
                "action": "share",
                "resource_id": resource_id,
                "target_contract_id": target_contract_id
            })
            
            return True
    
    def use_resource(self, resource_id: str, amount: float, 
                    owner_contract_id: str = None) -> bool:
        """
        Use a resource.
        
        Args:
            resource_id: ID of resource to use
            amount: Amount to allocate
            owner_contract_id: ID of the resource owner (if not self)
            
        Returns:
            True if resource was allocated successfully
        """
        # Check local resources first
        if resource_id in self.resources:
            with self.resource_lock:
                resource = self.resources[resource_id]
                return resource.allocate(amount, self.contract_id)
        
        # Check resources shared by other contracts
        if owner_contract_id:
            owner_contract = self._get_contract_by_id(owner_contract_id)
            if owner_contract:
                return owner_contract.allocate_resource(resource_id, amount, self.contract_id)
        
        return False
    
    def allocate_resource(self, resource_id: str, amount: float, requestor_id: str) -> bool:
        """
        Allocate a resource to another contract.
        
        Args:
            resource_id: ID of resource to allocate
            amount: Amount to allocate
            requestor_id: ID of the requesting contract
            
        Returns:
            True if resource was allocated successfully
        """
        with self.resource_lock:
            if resource_id not in self.resources:
                return False
            
            resource = self.resources[resource_id]
            
            # Check permissions
            if requestor_id != resource.owner_id and requestor_id not in resource.shared_with:
                return False
            
            # Allocate resource
            result = resource.allocate(amount, requestor_id)
            
            if result:
                # Emit event
                self._emit_event(EventType.RESOURCE_ALLOCATION, {
                    "action": "allocate",
                    "resource_id": resource_id,
                    "amount": amount,
                    "requestor_id": requestor_id
                })
            
            return result
    
    def release_resource(self, resource_id: str, amount: float) -> bool:
        """
        Release a previously allocated resource.
        
        Args:
            resource_id: ID of resource to release
            amount: Amount to release
            
        Returns:
            True if resource was released successfully
        """
        with self.resource_lock:
            if resource_id not in self.resources:
                return False
            
            resource = self.resources[resource_id]
            
            # Deallocate resource
            result = resource.deallocate(amount)
            
            if result:
                # Emit event
                self._emit_event(EventType.RESOURCE_ALLOCATION, {
                    "action": "release",
                    "resource_id": resource_id,
                    "amount": amount
                })
            
            return result
    
    def get_resources(self) -> List[Resource]:
        """Get all resources owned by this contract."""
        with self.resource_lock:
            return list(self.resources.values())
    
    # Contract serialization and persistence
    def to_dict(self) -> Dict[str, Any]:
        """Convert contract to dictionary."""
        return {
            "contract_id": self.contract_id,
            "address": self.address,
            "owner": self.owner,
            "created_at": self.created_at,
            "state": self.state,
            "state_version": self.state_version,
            "relationships": [r.to_dict() for r in self.get_relationships()],
            "resources": [r.to_dict() for r in self.get_resources()],
            "metadata": self.metadata
        }
    
    def save(self, directory: str = "data/contracts") -> bool:
        """
        Save contract state to disk.
        
        Args:
            directory: Directory to save to
            
        Returns:
            True if saved successfully
        """
        try:
            os.makedirs(directory, exist_ok=True)
            
            filename = os.path.join(directory, f"{self.contract_id}.json")
            
            with open(filename, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving contract: {e}")
            return False
    
    @classmethod
    def load(cls, contract_id: str, directory: str = "data/contracts") -> Optional['EnhancedContract']:
        """
        Load contract from disk.
        
        Args:
            contract_id: ID of contract to load
            directory: Directory to load from
            
        Returns:
            Loaded contract or None if failed
        """
        try:
            filename = os.path.join(directory, f"{contract_id}.json")
            
            if not os.path.exists(filename):
                return None
            
            with open(filename, "r") as f:
                data = json.load(f)
            
            # Create contract instance
            contract = cls(owner=data["owner"], address=data["address"])
            
            # Restore state
            contract.state = data["state"]
            contract.state_version = data["state_version"]
            
            # Create state snapshot
            contract._create_state_snapshot()
            
            return contract
        except Exception as e:
            print(f"Error loading contract: {e}")
            return None


class EnhancedSymbioticRegistry:
    """
    Enhanced registry for managing symbiotic contracts with advanced features.
    
    Features:
    - Contract lifecycle management
    - Relationship brokering
    - Composability
    - Formal verification
    - Resource optimization
    - Event routing
    - Version management
    """
    
    def __init__(self):
        """Initialize the registry."""
        self.contracts: Dict[str, EnhancedContract] = {}
        self.relationships: Dict[str, Relationship] = {}
        self.contract_types: Dict[str, str] = {}
        self.created_at = time.time()
        
        # Locks for thread safety
        self.contracts_lock = Lock()
        self.relationships_lock = Lock()
        
        # Event bus for cross-contract communication
        self.event_listeners: Dict[EventType, List[Callable]] = {
            event_type: [] for event_type in EventType
        }
        
        # Resource registry
        self.global_resources: Dict[str, Resource] = {}
        
        # Verification and regulatory components
        self.verifiers: Dict[str, Callable] = {}
        self.regulatory_rules: Dict[str, Callable] = {}
    
    def register_contract(self, contract: EnhancedContract) -> bool:
        """
        Register a contract in the registry.
        
        Args:
            contract: Contract to register
            
        Returns:
            True if contract was registered successfully
        """
        with self.contracts_lock:
            if contract.contract_id in self.contracts:
                return False
            
            # Store contract
            self.contracts[contract.contract_id] = contract
            
            # Store contract type
            contract_type = contract.__class__.__name__
            self.contract_types[contract.contract_id] = contract_type
            
            # Set up event routing
            contract.add_event_handler(EventType.RELATIONSHIP_FORMED, 
                                      self._on_relationship_formed)
            contract.add_event_handler(EventType.RELATIONSHIP_TERMINATED, 
                                      self._on_relationship_terminated)
            
            # Override _get_contract_by_id method
            contract._get_contract_by_id = self.get_contract
            
            return True
    
    def get_contract(self, contract_id: str) -> Optional[EnhancedContract]:
        """
        Get a contract by ID.
        
        Args:
            contract_id: ID of the contract
            
        Returns:
            Contract instance or None if not found
        """
        return self.contracts.get(contract_id)
    
    def get_contracts_by_type(self, contract_type: str) -> List[EnhancedContract]:
        """
        Get all contracts of a specific type.
        
        Args:
            contract_type: Type of contracts to retrieve
            
        Returns:
            List of matching contracts
        """
        return [
            contract for contract_id, contract in self.contracts.items()
            if self.contract_types.get(contract_id) == contract_type
        ]
    
    def create_relationship(self, source_id: str, target_id: str, 
                           relationship_type: Union[RelationshipType, str],
                           permissions: List[str] = None,
                           metadata: Dict[str, Any] = None) -> Optional[Relationship]:
        """
        Create a relationship between two contracts.
        
        Args:
            source_id: ID of source contract
            target_id: ID of target contract
            relationship_type: Type of relationship
            permissions: Permissions granted in the relationship
            metadata: Additional relationship metadata
            
        Returns:
            Created relationship or None if failed
        """
        # Get contracts
        source_contract = self.get_contract(source_id)
        target_contract = self.get_contract(target_id)
        
        if not source_contract or not target_contract:
            return None
        
        # Convert string relationship type if needed
        if isinstance(relationship_type, str):
            try:
                relationship_type = RelationshipType[relationship_type]
            except KeyError:
                return None
        
        # Create relationship
        return source_contract.form_relationship(
            target_contract,
            relationship_type,
            permissions,
            metadata
        )
    
    def _on_relationship_formed(self, event: ContractEvent) -> None:
        """Handle relationship formation events."""
        with self.relationships_lock:
            relationship_data = event.data.get("relationship")
            if relationship_data:
                relationship_id = relationship_data.get("relationship_id")
                if relationship_id:
                    # Store relationship
                    self.relationships[relationship_id] = relationship_data
                    
                    # Propagate event to listeners
                    self._propagate_event(event)
    
    def _on_relationship_terminated(self, event: ContractEvent) -> None:
        """Handle relationship termination events."""
        with self.relationships_lock:
            relationship_data = event.data.get("relationship")
            if relationship_data:
                relationship_id = relationship_data.get("relationship_id")
                if relationship_id and relationship_id in self.relationships:
                    # Update relationship
                    self.relationships[relationship_id] = relationship_data
                    
                    # Propagate event to listeners
                    self._propagate_event(event)
    
    def get_relationships(self, contract_id: str = None) -> List[Dict[str, Any]]:
        """
        Get relationships in the registry.
        
        Args:
            contract_id: Filter relationships by contract ID
            
        Returns:
            List of relationships
        """
        with self.relationships_lock:
            if contract_id:
                return [
                    r for r in self.relationships.values()
                    if r.get("source_id") == contract_id or r.get("target_id") == contract_id
                ]
            else:
                return list(self.relationships.values())
    
    def add_event_listener(self, event_type: EventType, listener: Callable) -> None:
        """
        Add a listener for registry events.
        
        Args:
            event_type: Type of event to listen for
            listener: Function to call when event occurs
        """
        self.event_listeners[event_type].append(listener)
    
    def _propagate_event(self, event: ContractEvent) -> None:
        """Propagate an event to listeners."""
        for listener in self.event_listeners[event.event_type]:
            try:
                listener(event)
            except Exception as e:
                print(f"Error in event listener: {e}")
    
    def get_contract_dependencies(self, contract_id: str) -> List[str]:
        """
        Get dependencies of a contract.
        
        Args:
            contract_id: ID of the contract
            
        Returns:
            List of contract IDs that this contract depends on
        """
        dependencies = []
        
        for relationship in self.get_relationships(contract_id):
            if relationship.get("source_id") == contract_id:
                if relationship.get("type") == RelationshipType.DEPENDENCY.name:
                    dependencies.append(relationship.get("target_id"))
        
        return dependencies
    
    def get_contract_dependents(self, contract_id: str) -> List[str]:
        """
        Get contracts that depend on this contract.
        
        Args:
            contract_id: ID of the contract
            
        Returns:
            List of contract IDs that depend on this contract
        """
        dependents = []
        
        for relationship in self.get_relationships(contract_id):
            if relationship.get("target_id") == contract_id:
                if relationship.get("type") == RelationshipType.DEPENDENCY.name:
                    dependents.append(relationship.get("source_id"))
        
        return dependents
    
    def register_global_resource(self, resource: Resource) -> bool:
        """
        Register a global resource.
        
        Args:
            resource: Resource to register
            
        Returns:
            True if resource was registered successfully
        """
        self.global_resources[resource.resource_id] = resource
        return True
    
    def allocate_global_resource(self, resource_id: str, amount: float, 
                               contract_id: str) -> bool:
        """
        Allocate a global resource to a contract.
        
        Args:
            resource_id: ID of resource to allocate
            amount: Amount to allocate
            contract_id: ID of the requesting contract
            
        Returns:
            True if resource was allocated successfully
        """
        if resource_id not in self.global_resources:
            return False
        
        resource = self.global_resources[resource_id]
        
        return resource.allocate(amount, contract_id)
    
    def register_verifier(self, name: str, verifier: Callable) -> None:
        """
        Register a formal verifier.
        
        Args:
            name: Name of the verifier
            verifier: Verification function
        """
        self.verifiers[name] = verifier
    
    def verify_contract(self, contract_id: str, verifier_name: str = None) -> Dict[str, Any]:
        """
        Verify a contract.
        
        Args:
            contract_id: ID of contract to verify
            verifier_name: Name of verifier to use (or all if None)
            
        Returns:
            Verification results
        """
        contract = self.get_contract(contract_id)
        if not contract:
            return {"success": False, "error": "Contract not found"}
        
        results = {}
        
        if verifier_name:
            if verifier_name not in self.verifiers:
                return {"success": False, "error": f"Verifier '{verifier_name}' not found"}
            
            verifier = self.verifiers[verifier_name]
            results[verifier_name] = verifier(contract)
        else:
            for name, verifier in self.verifiers.items():
                results[name] = verifier(contract)
        
        return {
            "success": True,
            "contract_id": contract_id,
            "results": results
        }
    
    def register_regulatory_rule(self, name: str, rule: Callable) -> None:
        """
        Register a regulatory rule.
        
        Args:
            name: Name of the rule
            rule: Rule function
        """
        self.regulatory_rules[name] = rule
    
    def check_regulatory_compliance(self, contract_id: str, rule_name: str = None) -> Dict[str, Any]:
        """
        Check regulatory compliance of a contract.
        
        Args:
            contract_id: ID of contract to check
            rule_name: Name of rule to check (or all if None)
            
        Returns:
            Compliance results
        """
        contract = self.get_contract(contract_id)
        if not contract:
            return {"success": False, "error": "Contract not found"}
        
        results = {}
        
        if rule_name:
            if rule_name not in self.regulatory_rules:
                return {"success": False, "error": f"Rule '{rule_name}' not found"}
            
            rule = self.regulatory_rules[rule_name]
            results[rule_name] = rule(contract)
        else:
            for name, rule in self.regulatory_rules.items():
                results[name] = rule(contract)
        
        return {
            "success": True,
            "contract_id": contract_id,
            "results": results
        }
    
    def compose_contracts(self, contract_ids: List[str], 
                         owner: str, 
                         composition_type: str = "aggregate") -> Optional[EnhancedContract]:
        """
        Compose multiple contracts into a composite contract.
        
        Args:
            contract_ids: IDs of contracts to compose
            owner: Owner of the composite contract
            composition_type: Type of composition
            
        Returns:
            Composite contract or None if failed
        """
        # Check if all contracts exist
        contracts = []
        for contract_id in contract_ids:
            contract = self.get_contract(contract_id)
            if not contract:
                return None
            contracts.append(contract)
        
        # Create composite contract
        composite = CompositeContract(owner, contracts, composition_type)
        
        # Register composite
        self.register_contract(composite)
        
        # Create relationships
        for contract in contracts:
            self.create_relationship(
                composite.contract_id,
                contract.contract_id,
                RelationshipType.COMPOSITE,
                ["*"]  # Composite has all permissions
            )
        
        return composite


class CompositeContract(EnhancedContract):
    """
    A contract composed of multiple other contracts.
    
    This demonstrates the composability of contracts in the system.
    """
    
    def __init__(self, owner: str, components: List[EnhancedContract], 
                composition_type: str = "aggregate", address: str = None):
        """
        Initialize a composite contract.
        
        Args:
            owner: Owner of the contract
            components: Component contracts
            composition_type: Type of composition
            address: Contract address
        """
        super().__init__(owner, address)
        
        self.components = components
        self.composition_type = composition_type
        self.component_ids = [c.contract_id for c in components]
        
        # Update metadata
        self.metadata.update({
            "composite": True,
            "composition_type": composition_type,
            "component_ids": self.component_ids,
            "component_count": len(components)
        })
        
        # Initialize composite state
        self._initialize_composite_state()
    
    def _initialize_composite_state(self) -> None:
        """Initialize state based on component contracts."""
        composite_state = {}
        
        if self.composition_type == "aggregate":
            # Aggregate state as namespaced components
            for component in self.components:
                component_id = component.contract_id
                component_type = component.__class__.__name__
                composite_state[component_type] = {
                    "contract_id": component_id,
                    "state": component.get_state()
                }
        elif self.composition_type == "merge":
            # Merge states (with conflict resolution)
            for component in self.components:
                component_state = component.get_state()
                for key, value in component_state.items():
                    # Simple conflict resolution - later components override earlier ones
                    composite_state[key] = value
        
        # Update state
        self.update_state(composite_state)
    
    def delegate_call(self, component_index: int, method_name: str, 
                     params: Dict[str, Any] = None) -> Any:
        """
        Delegate a call to a component contract.
        
        Args:
            component_index: Index of component to call
            method_name: Method to call
            params: Parameters to pass
            
        Returns:
            Result of the call
        """
        if component_index < 0 or component_index >= len(self.components):
            return None
        
        component = self.components[component_index]
        
        if not hasattr(component, method_name):
            return None
        
        method = getattr(component, method_name)
        params = params or {}
        
        try:
            return method(caller=self.contract_id, **params)
        except Exception as e:
            self._emit_event(EventType.ERROR, {
                "error": str(e),
                "component_index": component_index,
                "method": method_name,
                "params": params
            })
            return None
    
    def broadcast_call(self, method_name: str, params: Dict[str, Any] = None) -> List[Any]:
        """
        Call a method on all component contracts.
        
        Args:
            method_name: Method to call
            params: Parameters to pass
            
        Returns:
            List of results from components
        """
        results = []
        params = params or {}
        
        for i, component in enumerate(self.components):
            if hasattr(component, method_name):
                try:
                    method = getattr(component, method_name)
                    result = method(caller=self.contract_id, **params)
                    results.append(result)
                except Exception as e:
                    self._emit_event(EventType.ERROR, {
                        "error": str(e),
                        "component_index": i,
                        "method": method_name,
                        "params": params
                    })
                    results.append(None)
            else:
                results.append(None)
        
        return results
    
    def update_component(self, component_index: int, new_state: Dict[str, Any]) -> bool:
        """
        Update a component's state.
        
        Args:
            component_index: Index of component to update
            new_state: New state for the component
            
        Returns:
            True if state was updated successfully
        """
        if component_index < 0 or component_index >= len(self.components):
            return False
        
        component = self.components[component_index]
        
        # Update component state
        result = component.update_state(new_state, caller=self.contract_id)
        
        if result:
            # Update composite state
            self._initialize_composite_state()
        
        return result
    
    def get_component_state(self, component_index: int) -> Dict[str, Any]:
        """
        Get a component's state.
        
        Args:
            component_index: Index of component
            
        Returns:
            Component state
        """
        if component_index < 0 or component_index >= len(self.components):
            return {}
        
        component = self.components[component_index]
        return component.get_state()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert contract to dictionary."""
        data = super().to_dict()
        
        # Add composite-specific data
        data.update({
            "composition_type": self.composition_type,
            "component_ids": self.component_ids,
            "component_count": len(self.components)
        })
        
        return data


class EnhancedTokenContract(EnhancedContract):
    """
    Enhanced implementation of the NYX token contract.
    
    Improvements:
    - Immutable transaction history
    - Advanced tokenomics
    - Regulatory compliance
    - Staking integration
    - Formal verification
    """
    
    def __init__(self, owner: str, total_supply: float = 808000000, address: str = None):
        """
        Initialize the token contract.
        
        Args:
            owner: Contract owner (genesis address)
            total_supply: Initial token supply
            address: Contract address
        """
        super().__init__(owner, address)
        
        # Initialize token state
        self.state.update({
            "name": "NyxSynth",
            "symbol": "NYX",
            "decimals": 18,
            "total_supply": total_supply,
            "circulating_supply": total_supply,
            "balances": {owner: total_supply},
            "allowances": {},
            "burned": 0,
            "transaction_count": 0,
            "holder_count": 1,
            "genesis_timestamp": time.time(),
            "last_transaction_timestamp": time.time()
        })
        
        # Create token transaction history
        self.transaction_history = []
        
        # Create initial state snapshot
        self._create_state_snapshot()
        
        # Create token resource
        self.token_resource = self.create_resource(
            ResourceType.TOKEN,
            total_supply,
            {"symbol": "NYX", "name": "NyxSynth Token"}
        )
    
    def balance_of(self, address: str, caller: str = None) -> float:
        """
        Get the token balance of an address.
        
        Args:
            address: Address to check
            caller: Calling contract/user
            
        Returns:
            Token balance
        """
        return self.state["balances"].get(address, 0)
    
    def transfer(self, from_address: str, to_address: str, amount: float, 
                caller: str = None) -> bool:
        """
        Transfer tokens from one address to another.
        
        Args:
            from_address: Source address
            to_address: Destination address
            amount: Amount to transfer
            caller: Calling contract/user
            
        Returns:
            True if transfer was successful
        """
        # Validate parameters
        if amount <= 0:
            return False
        
        # Check permissions
        if caller and caller != from_address and caller != self.owner:
            # Check if caller has allowance
            if self.state["allowances"].get(from_address, {}).get(caller, 0) < amount:
                return False
            
            # Deduct from allowance
            self.state["allowances"][from_address][caller] -= amount
        
        # Check balance
        if self.state["balances"].get(from_address, 0) < amount:
            return False
        
        # Check if sender is a burned address
        if from_address == "0x000000000000000000000000000000000000dEaD":
            return False
        
        # Calculate automatic burning (0.5% of each transaction)
        burn_amount = amount * 0.005
        transfer_amount = amount - burn_amount
        
        # Update state
        with self.state_lock:
            # Deduct from sender
            self.state["balances"][from_address] = self.state["balances"].get(from_address, 0) - amount
            
            # Add to recipient
            self.state["balances"][to_address] = self.state["balances"].get(to_address, 0) + transfer_amount
            
            # Update token stats
            self.state["total_supply"] -= burn_amount
            self.state["burned"] += burn_amount
            self.state["transaction_count"] += 1
            self.state["last_transaction_timestamp"] = time.time()
            
            # Update holder count
            if to_address not in self.state["balances"]:
                self.state["holder_count"] += 1
            if self.state["balances"][from_address] == 0:
                self.state["holder_count"] -= 1
            
            # Create transaction record
            transaction = {
                "transaction_id": str(uuid.uuid4()),
                "from": from_address,
                "to": to_address,
                "amount": amount,
                "transfer_amount": transfer_amount,
                "burn_amount": burn_amount,
                "timestamp": time.time(),
                "caller": caller
            }
            
            self.transaction_history.append(transaction)
            
            # Create state snapshot
            self._create_state_snapshot()
            
            # Emit event
            self._emit_event(EventType.STATE_CHANGE, {
                "transaction": transaction,
                "new_balances": {
                    from_address: self.state["balances"].get(from_address, 0),
                    to_address: self.state["balances"].get(to_address, 0)
                },
                "total_supply": self.state["total_supply"],
                "burned": self.state["burned"]
            })
            
            # Notify related contracts
            self.call_related("on_token_transfer", {
                "from": from_address,
                "to": to_address,
                "amount": amount,
                "burn_amount": burn_amount
            })
            
            return True
    
    def approve(self, owner: str, spender: str, amount: float, caller: str = None) -> bool:
        """
        Approve a spender to spend tokens on behalf of the owner.
        
        Args:
            owner: Token owner
            spender: Address to approve
            amount: Approval amount
            caller: Calling contract/user
            
        Returns:
            True if approval was successful
        """
        # Check permissions
        if caller and caller != owner and caller != self.owner:
            return False
        
        with self.state_lock:
            # Initialize owner allowances if needed
            if owner not in self.state["allowances"]:
                self.state["allowances"][owner] = {}
            
            # Set allowance
            self.state["allowances"][owner][spender] = amount
            
            # Create state snapshot
            self._create_state_snapshot()
            
            # Emit event
            self._emit_event(EventType.STATE_CHANGE, {
                "action": "approve",
                "owner": owner,
                "spender": spender,
                "amount": amount
            })
            
            return True
    
    def allowance(self, owner: str, spender: str, caller: str = None) -> float:
        """
        Get the amount of tokens approved for a spender.
        
        Args:
            owner: Token owner
            spender: Address to check allowance for
            caller: Calling contract/user
            
        Returns:
            Allowance amount
        """
        return self.state["allowances"].get(owner, {}).get(spender, 0)
    
    def transfer_from(self, sender: str, from_address: str, to_address: str, 
                     amount: float, caller: str = None) -> bool:
        """
        Transfer tokens on behalf of another address.
        
        Args:
            sender: Address initiating the transfer
            from_address: Source address
            to_address: Destination address
            amount: Amount to transfer
            caller: Calling contract/user
            
        Returns:
            True if transfer was successful
        """
        # Check permissions
        if caller and caller != sender and caller != self.owner:
            return False
        
        # Check allowance
        if self.state["allowances"].get(from_address, {}).get(sender, 0) < amount:
            return False
        
        # Reduce allowance
        self.state["allowances"][from_address][sender] -= amount
        
        # Perform transfer
        return self.transfer(from_address, to_address, amount, sender)
    
    def burn(self, amount: float, from_address: str = None, caller: str = None) -> bool:
        """
        Burn tokens, removing them from circulation.
        
        Args:
            amount: Amount to burn
            from_address: Address to burn from (default: owner)
            caller: Calling contract/user
            
        Returns:
            True if burn was successful
        """
        # Set default from_address to owner if not specified
        if from_address is None:
            from_address = self.owner
        
        # Transfer to burn address
        burn_address = "0x000000000000000000000000000000000000dEaD"
        return self.transfer(from_address, burn_address, amount, caller)
    
    def mint(self, to_address: str, amount: float, caller: str = None) -> bool:
        """
        Mint new tokens.
        
        Args:
            to_address: Address to mint to
            amount: Amount to mint
            caller: Calling contract/user
            
        Returns:
            True if mint was successful
        """
        # Check permissions
        if caller and caller != self.owner:
            return False
        
        with self.state_lock:
            # Update balances
            self.state["balances"][to_address] = self.state["balances"].get(to_address, 0) + amount
            
            # Update token stats
            self.state["total_supply"] += amount
            self.state["circulating_supply"] += amount
            
            # Update holder count
            if to_address not in self.state["balances"]:
                self.state["holder_count"] += 1
            
            # Create transaction record
            transaction = {
                "transaction_id": str(uuid.uuid4()),
                "from": "0x0000000000000000000000000000000000000000",  # Zero address for minting
                "to": to_address,
                "amount": amount,
                "mint": True,
                "timestamp": time.time(),
                "caller": caller
            }
            
            self.transaction_history.append(transaction)
            
            # Create state snapshot
            self._create_state_snapshot()
            
            # Emit event
            self._emit_event(EventType.STATE_CHANGE, {
                "transaction": transaction,
                "new_balance": self.state["balances"].get(to_address, 0),
                "total_supply": self.state["total_supply"]
            })
            
            # Notify related contracts
            self.call_related("on_token_mint", {
                "to": to_address,
                "amount": amount
            })
            
            return True
    
    def get_transaction_history(self, address: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get token transaction history.
        
        Args:
            address: Filter by address
            limit: Maximum number of transactions to return
            
        Returns:
            List of transactions
        """
        if address:
            # Filter by address
            filtered = [
                tx for tx in self.transaction_history
                if tx["from"] == address or tx["to"] == address
            ]
            return filtered[-limit:]
        else:
            # Return all transactions up to limit
            return self.transaction_history[-limit:]
    
    def get_token_stats(self) -> Dict[str, Any]:
        """
        Get token statistics.
        
        Returns:
            Token statistics
        """
        return {
            "name": self.state["name"],
            "symbol": self.state["symbol"],
            "decimals": self.state["decimals"],
            "total_supply": self.state["total_supply"],
            "circulating_supply": self.state["circulating_supply"],
            "burned": self.state["burned"],
            "transaction_count": self.state["transaction_count"],
            "holder_count": self.state["holder_count"],
            "genesis_timestamp": self.state["genesis_timestamp"],
            "age_days": (time.time() - self.state["genesis_timestamp"]) / (60 * 60 * 24)
        }
    
    def on_token_burn(self, amount: float) -> bool:
        """
        Handle token burning events (callback for related contracts).
        
        Args:
            amount: Amount burned
            
        Returns:
            True if handled successfully
        """
        # This method can be called by other contracts when tokens are burned
        # For example, a staking contract might want to adjust rewards based on burns
        
        return True


class EnhancedStakingContract(EnhancedContract):
    """
    Enhanced implementation of a staking contract with neural rewards.
    
    Features:
    - Neural staking pools with adaptive rewards
    - Dynamic reward calculation based on network activity
    - Time-weighted positions
    - Validator incentives
    - Integration with token contract
    """
    
    def __init__(self, owner: str, token_contract_address: str, address: str = None):
        """
        Initialize the staking contract.
        
        Args:
            owner: Contract owner
            token_contract_address: Address of the token contract
            address: Contract address
        """
        super().__init__(owner, address)
        
        # Initialize state
        self.state.update({
            "token_contract": token_contract_address,
            "total_staked": 0,
            "staking_pools": {},
            "stakes": {},
            "rewards_distributed": 0,
            "total_validators": 0,
            "last_distribution": time.time(),
            "created_at": time.time()
        })
        
        # Create initial state snapshot
        self._create_state_snapshot()
    
    def create_pool(self, pool_id: str, name: str, reward_multiplier: float,
                  min_stake: float = 100, lock_period: int = 0,
                  metadata: Dict[str, Any] = None, caller: str = None) -> bool:
        """
        Create a new staking pool.
        
        Args:
            pool_id: Unique ID for the pool
            name: Pool name
            reward_multiplier: Reward multiplier for the pool
            min_stake: Minimum stake amount
            lock_period: Lock period in seconds
            metadata: Additional pool metadata
            caller: Calling contract/user
            
        Returns:
            True if pool was created successfully
        """
        # Check permissions
        if caller and caller != self.owner:
            return False
        
        with self.state_lock:
            # Check if pool already exists
            if pool_id in self.state["staking_pools"]:
                return False
            
            # Create pool
            self.state["staking_pools"][pool_id] = {
                "id": pool_id,
                "name": name,
                "reward_multiplier": reward_multiplier,
                "min_stake": min_stake,
                "lock_period": lock_period,
                "total_staked": 0,
                "staker_count": 0,
                "created_at": time.time(),
                "metadata": metadata or {}
            }
            
            # Create state snapshot
            self._create_state_snapshot()
            
            # Emit event
            self._emit_event(EventType.STATE_CHANGE, {
                "action": "create_pool",
                "pool_id": pool_id,
                "name": name,
                "reward_multiplier": reward_multiplier
            })
            
            return True
    
    def stake(self, staker: str, pool_id: str, amount: float, caller: str = None) -> bool:
        """
        Stake tokens in a pool.
        
        Args:
            staker: Address of the staker
            pool_id: ID of the pool to stake in
            amount: Amount to stake
            caller: Calling contract/user
            
        Returns:
            True if stake was successful
        """
        # Check permissions
        if caller and caller != staker and caller != self.owner:
            return False
        
        with self.state_lock:
            # Check if pool exists
            if pool_id not in self.state["staking_pools"]:
                return False
            
            pool = self.state["staking_pools"][pool_id]
            
            # Check minimum stake
            if amount < pool["min_stake"]:
                return False
            
            # Transfer tokens from staker to this contract
            token_contract = self._get_token_contract()
            if not token_contract:
                return False
            
            # Calculate stake ID
            stake_id = f"{staker}:{pool_id}:{int(time.time())}"
            
            # Check if transfer was successful
            if not token_contract.transfer(staker, self.contract_id, amount, caller=self.contract_id):
                return False
            
            # Update pool stats
            pool["total_staked"] += amount
            
            # Check if this is a new staker for this pool
            new_staker = True
            for existing_stake_id, stake in self.state["stakes"].items():
                if stake["staker"] == staker and stake["pool_id"] == pool_id:
                    new_staker = False
                    break
            
            if new_staker:
                pool["staker_count"] += 1
            
            # Create stake
            self.state["stakes"][stake_id] = {
                "id": stake_id,
                "staker": staker,
                "pool_id": pool_id,
                "amount": amount,
                "start_time": time.time(),
                "end_time": None,
                "rewards_claimed": 0,
                "active": True,
                "neural_pattern": self._generate_stake_pattern(staker, pool_id, amount)
            }
            
            # Update global stats
            self.state["total_staked"] += amount
            
            # Create state snapshot
            self._create_state_snapshot()
            
            # Emit event
            self._emit_event(EventType.STATE_CHANGE, {
                "action": "stake",
                "stake_id": stake_id,
                "staker": staker,
                "pool_id": pool_id,
                "amount": amount
            })
            
            return True
    
    def unstake(self, stake_id: str, caller: str = None) -> bool:
        """
        Unstake tokens from a pool.
        
        Args:
            stake_id: ID of the stake to unstake
            caller: Calling contract/user
            
        Returns:
            True if unstake was successful
        """
        with self.state_lock:
            # Check if stake exists
            if stake_id not in self.state["stakes"]:
                return False
            
            stake = self.state["stakes"][stake_id]
            
            # Check if stake is active
            if not stake["active"]:
                return False
            
            # Check permissions
            if caller and caller != stake["staker"] and caller != self.owner:
                return False
            
            # Check lock period
            pool = self.state["staking_pools"].get(stake["pool_id"])
            if pool and pool["lock_period"] > 0:
                time_staked = time.time() - stake["start_time"]
                if time_staked < pool["lock_period"]:
                    return False
            
            # Calculate pending rewards
            pending_rewards = self._calculate_rewards(stake_id)
            
            # Transfer staked amount back to staker
            token_contract = self._get_token_contract()
            if not token_contract:
                return False
            
            # Transfer stake amount
            if not token_contract.transfer(self.contract_id, stake["staker"], stake["amount"], 
                                         caller=self.contract_id):
                return False
            
            # Transfer rewards if any
            if pending_rewards > 0:
                token_contract.transfer(self.contract_id, stake["staker"], pending_rewards, 
                                       caller=self.contract_id)
                self.state["rewards_distributed"] += pending_rewards
            
            # Update stake
            stake["active"] = False
            stake["end_time"] = time.time()
            stake["rewards_claimed"] += pending_rewards
            
            # Update pool stats
            if pool:
                pool["total_staked"] -= stake["amount"]
                pool["staker_count"] -= 1
            
            # Update global stats
            self.state["total_staked"] -= stake["amount"]
            
            # Create state snapshot
            self._create_state_snapshot()
            
            # Emit event
            self._emit_event(EventType.STATE_CHANGE, {
                "action": "unstake",
                "stake_id": stake_id,
                "staker": stake["staker"],
                "amount": stake["amount"],
                "rewards": pending_rewards
            })
            
            return True
    
    def claim_rewards(self, stake_id: str, caller: str = None) -> float:
        """
        Claim rewards for a stake.
        
        Args:
            stake_id: ID of the stake
            caller: Calling contract/user
            
        Returns:
            Amount of rewards claimed
        """
        with self.state_lock:
            # Check if stake exists
            if stake_id not in self.state["stakes"]:
                return 0
            
            stake = self.state["stakes"][stake_id]
            
            # Check if stake is active
            if not stake["active"]:
                return 0
            
            # Check permissions
            if caller and caller != stake["staker"] and caller != self.owner:
                return 0
            
            # Calculate pending rewards
            pending_rewards = self._calculate_rewards(stake_id)
            
            if pending_rewards <= 0:
                return 0
            
            # Transfer rewards
            token_contract = self._get_token_contract()
            if not token_contract:
                return 0
            
            # Transfer rewards
            if not token_contract.transfer(self.contract_id, stake["staker"], pending_rewards, 
                                         caller=self.contract_id):
                return 0
            
            # Update stake
            stake["rewards_claimed"] += pending_rewards
            
            # Update global stats
            self.state["rewards_distributed"] += pending_rewards
            
            # Create state snapshot
            self._create_state_snapshot()
            
            # Emit event
            self._emit_event(EventType.STATE_CHANGE, {
                "action": "claim_rewards",
                "stake_id": stake_id,
                "staker": stake["staker"],
                "rewards": pending_rewards
            })
            
            return pending_rewards
    
    def _calculate_rewards(self, stake_id: str) -> float:
        """
        Calculate pending rewards for a stake.
        
        Args:
            stake_id: ID of the stake
            
        Returns:
            Pending rewards amount
        """
        stake = self.state["stakes"].get(stake_id)
        if not stake or not stake["active"]:
            return 0
        
        pool = self.state["staking_pools"].get(stake["pool_id"])
        if not pool:
            return 0
        
        # Calculate time staked
        time_staked = time.time() - stake["start_time"]
        days_staked = time_staked / (60 * 60 * 24)
        
        # Base reward rate (5% APY)
        base_rate = 0.05 / 365
        
        # Apply pool multiplier
        rate = base_rate * pool["reward_multiplier"]
        
        # Apply neural pattern bonus
        pattern_bonus = self._calculate_pattern_bonus(stake["neural_pattern"])
        rate *= (1 + pattern_bonus)
        
        # Calculate rewards
        rewards = stake["amount"] * rate * days_staked
        
        # Subtract already claimed rewards
        pending = rewards - stake["rewards_claimed"]
        
        return max(0, pending)
    
    def _calculate_pattern_bonus(self, pattern: List[float]) -> float:
        """
        Calculate bonus based on neural pattern.
        
        Args:
            pattern: Neural pattern
            
        Returns:
            Bonus multiplier (0-0.5)
        """
        # This would use the actual neural validation network
        # For now, we'll use a simplified approach
        
        # Convert to numpy array
        import numpy as np
        pattern_array = np.array(pattern)
        
        # Calculate entropy (diversity of values)
        normalized = np.abs(pattern_array) / np.sum(np.abs(pattern_array) + 1e-10)
        entropy = -np.sum(normalized * np.log2(normalized + 1e-10))
        
        # Calculate bonus based on entropy
        # Higher entropy (more diverse pattern) = higher bonus
        max_entropy = np.log2(len(pattern))
        entropy_ratio = entropy / max_entropy
        
        # Scale to 0-0.5 bonus range
        bonus = entropy_ratio * 0.5
        
        return bonus
    
    def _generate_stake_pattern(self, staker: str, pool_id: str, amount: float) -> List[float]:
        """
        Generate a neural pattern for a stake.
        
        Args:
            staker: Staker address
            pool_id: Pool ID
            amount: Stake amount
            
        Returns:
            Neural pattern
        """
        # This would use the actual neural validator
        # For now, we'll generate a deterministic pattern
        
        import numpy as np
        import hashlib
        
        # Create seed from stake data
        seed_data = f"{staker}:{pool_id}:{amount}:{time.time()}"
        seed = int(hashlib.md5(seed_data.encode()).hexdigest(), 16) % 10000
        
        # Set seed for reproducibility
        np.random.seed(seed)
        
        # Generate pattern
        pattern = np.random.rand(64) * 2 - 1
        
        # Normalize
        pattern = pattern / np.linalg.norm(pattern)
        
        return pattern.tolist()
    
    def get_all_pools(self) -> List[Dict[str, Any]]:
        """
        Get all staking pools.
        
        Returns:
            List of staking pools
        """
        return list(self.state["staking_pools"].values())
    
    def get_stakes_by_address(self, address: str) -> List[Dict[str, Any]]:
        """
        Get all stakes for an address.
        
        Args:
            address: Staker address
            
        Returns:
            List of stakes
        """
        return [
            stake for stake in self.state["stakes"].values()
            if stake["staker"] == address
        ]
    
    def get_pool_stats(self, pool_id: str) -> Dict[str, Any]:
        """
        Get statistics for a staking pool.
        
        Args:
            pool_id: Pool ID
            
        Returns:
            Pool statistics
        """
        pool = self.state["staking_pools"].get(pool_id)
        if not pool:
            return {}
        
        return {
            "id": pool["id"],
            "name": pool["name"],
            "total_staked": pool["total_staked"],
            "staker_count": pool["staker_count"],
            "reward_multiplier": pool["reward_multiplier"],
            "min_stake": pool["min_stake"],
            "lock_period_days": pool["lock_period"] / (60 * 60 * 24) if pool["lock_period"] > 0 else 0,
            "age_days": (time.time() - pool["created_at"]) / (60 * 60 * 24)
        }
    
    def _get_token_contract(self) -> Any:
        """Get the token contract instance."""
        token_contract_address = self.state.get("token_contract")
        if not token_contract_address:
            return None
        
        return self._get_contract_by_id(token_contract_address)
    
    def on_token_transfer(self, from_address: str, to_address: str, 
                         amount: float, burn_amount: float = 0) -> bool:
        """
        Handle token transfer events.
        
        Args:
            from_address: Source address
            to_address: Destination address
            amount: Transfer amount
            burn_amount: Amount burned
            
        Returns:
            True if handled successfully
        """
        # This is called by the token contract when transfers occur
        # We can use this to update staking stats, adjust rewards, etc.
        
        # Example: Adjust rewards based on transfer volume
        if burn_amount > 0:
            # More tokens burned = more rewards for stakers
            # This encourages staking during high-volume periods
            pass
        
        return True
    
    def on_token_burn(self, amount: float) -> bool:
        """
        Handle token burn events.
        
        Args:
            amount: Amount burned
            
        Returns:
            True if handled successfully
        """
        # Tokens being burned increases scarcity, which could increase rewards
        
        return True


class ValidatorVerifier:
    """
    Provides formal verification for neural validation patterns.
    """
    
    @staticmethod
    def verify_pattern(pattern: List[float]) -> Dict[str, Any]:
        """
        Verify a neural pattern.
        
        Args:
            pattern: Pattern to verify
            
        Returns:
            Verification results
        """
        import numpy as np
        
        # Convert to numpy array
        pattern_array = np.array(pattern)
        
        # Check dimension
        if len(pattern_array) != 64:
            return {
                "valid": False,
                "reason": f"Invalid pattern dimension: {len(pattern_array)} (expected 64)"
            }
        
        # Check for NaN or Inf values
        if np.isnan(pattern_array).any() or np.isinf(pattern_array).any():
            return {
                "valid": False,
                "reason": "Pattern contains NaN or Inf values"
            }
        
        # Check norm
        norm = np.linalg.norm(pattern_array)
        if abs(norm - 1.0) > 1e-6:
            return {
                "valid": False,
                "reason": f"Pattern not normalized: norm = {norm}"
            }
        
        # Check entropy (diversity of values)
        normalized = np.abs(pattern_array) / np.sum(np.abs(pattern_array) + 1e-10)
        entropy = -np.sum(normalized * np.log2(normalized + 1e-10))
        
        if entropy < 3.0:  # Minimum required entropy
            return {
                "valid": False,
                "reason": f"Pattern entropy too low: {entropy} < 3.0"
            }
        
        # Check for suspicious patterns
        max_value = np.max(np.abs(pattern_array))
        if max_value > 0.5:
            return {
                "valid": False,
                "reason": f"Pattern contains dominant value: {max_value} > 0.5"
            }
        
        return {
            "valid": True,
            "entropy": entropy,
            "max_value": max_value,
            "norm": norm
        }


class RegulatoryCompliance:
    """
    Provides regulatory compliance checks for contracts.
    """
    
    @staticmethod
    def check_token_compliance(token_contract: EnhancedTokenContract) -> Dict[str, Any]:
        """
        Check token contract compliance.
        
        Args:
            token_contract: Token contract to check
            
        Returns:
            Compliance results
        """
        results = {
            "compliant": True,
            "issues": []
        }
        
        # Check total supply
        if token_contract.state["total_supply"] > 1000000000:
            results["compliant"] = False
            results["issues"].append("Total supply exceeds 1 billion tokens")
        
        # Check burn mechanism
        if "burned" not in token_contract.state or token_contract.state["burned"] == 0:
            results["issues"].append("No token burn mechanism detected (warning)")
        
        # Check for large holders (centralization risk)
        large_holders = 0
        total_supply = token_contract.state["total_supply"]
        
        for address, balance in token_contract.state["balances"].items():
            if balance > total_supply * 0.1:  # Holds more than 10%
                large_holders += 1
        
        if large_holders > 3:
            results["compliant"] = False
            results["issues"].append(f"Too many large holders ({large_holders})")
        
        return results


class EnhancedContractTesting:
    """
    Testing utilities for enhanced contracts.
    """
    
    @staticmethod
    def test_token_contract():
        """Test the enhanced token contract."""
        # Create token contract
        owner = "0xowner"
        token = EnhancedTokenContract(owner, 1000000)
        
        # Test basic functionality
        assert token.balance_of(owner) == 1000000
        
        # Test transfer
        recipient = "0xrecipient"
        assert token.transfer(owner, recipient, 100000)
        assert token.balance_of(recipient) == 99500  # 0.5% burned
        assert token.balance_of(owner) == 900000
        assert token.state["burned"] == 500
        
        # Test allowance
        spender = "0xspender"
        assert token.approve(owner, spender, 50000)
        assert token.allowance(owner, spender) == 50000
        
        # Test transfer_from
        recipient2 = "0xrecipient2"
        assert token.transfer_from(spender, owner, recipient2, 20000, spender)
        assert token.balance_of(recipient2) == 19900  # 0.5% burned
        assert token.balance_of(owner) == 880000
        assert token.allowance(owner, spender) == 30000
        
        print("Token contract tests passed!")
    
    @staticmethod
    def test_staking_contract():
        """Test the enhanced staking contract."""
        # Create contracts
        owner = "0xowner"
        token = EnhancedTokenContract(owner, 1000000)
        staking = EnhancedStakingContract(owner, token.address)
        
        # Set up registry so contracts can find each other
        registry = EnhancedSymbioticRegistry()
        registry.register_contract(token)
        registry.register_contract(staking)
        registry.create_relationship(token.address, staking.address, RelationshipType.RESOURCE_SHARING)
        
        # Create staking pool
        assert staking.create_pool("pool1", "Standard Pool", 1.0, 100)
        
        # Transfer tokens to staker
        staker = "0xstaker"
        token.transfer(owner, staker, 10000)
        
        # Stake tokens
        assert staking.stake(staker, "pool1", 1000)
        
        # Check staking stats
        assert staking.state["total_staked"] == 1000
        assert staking.state["staking_pools"]["pool1"]["total_staked"] == 1000
        
        # Test composite contract
        composite = CompositeContract(owner, [token, staking], "aggregate")
        registry.register_contract(composite)
        
        # Test composite operations
        assert composite.delegate_call(0, "balance_of", {"address": staker}) == token.balance_of(staker)
        
        print("Staking contract tests passed!")


if __name__ == "__main__":
    # Run tests
    EnhancedContractTesting.test_token_contract()
    EnhancedContractTesting.test_staking_contract()
