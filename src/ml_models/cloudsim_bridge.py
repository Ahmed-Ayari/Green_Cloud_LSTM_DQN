"""
CloudSim-Python Bridge
Socket-based API for communication between Python ML models and Java CloudSim simulator
"""

import socket
import json
import threading
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum


class SimulationMode(Enum):
    """Simulation mode enumeration"""
    STANDALONE = "standalone"  # Python-only simulation
    CLOUDSIM = "cloudsim"      # Connected to Java CloudSim


@dataclass
class HostState:
    """Represents the state of a physical host"""
    host_id: int
    cpu_utilization: float      # 0-100%
    ram_utilization: float      # 0-100%
    power_consumption: float    # Watts
    num_vms: int
    is_active: bool
    available_mips: float
    available_ram: float


@dataclass
class VMState:
    """Represents the state of a virtual machine"""
    vm_id: int
    host_id: int
    cpu_requested: float
    ram_requested: float
    cpu_utilization: float


@dataclass
class MigrationCommand:
    """Command to migrate a VM"""
    vm_id: int
    source_host_id: int
    target_host_id: int
    timestamp: float


@dataclass
class SimulationState:
    """Complete simulation state"""
    timestamp: float
    hosts: List[HostState]
    vms: List[VMState]
    total_energy: float
    sla_violations: int
    completed_migrations: int


class CloudSimBridge:
    """
    Bridge for communication between Python ML models and CloudSim (Java)
    
    Supports two modes:
    1. STANDALONE: Python-only simulation (default)
    2. CLOUDSIM: Connected to Java CloudSim via socket
    """
    
    def __init__(self, host: str = "localhost", port: int = 9999, 
                 mode: SimulationMode = SimulationMode.STANDALONE,
                 timeout: float = 30.0):
        """
        Initialize CloudSim Bridge
        
        Args:
            host: CloudSim server host
            port: CloudSim server port
            mode: Simulation mode (STANDALONE or CLOUDSIM)
            timeout: Socket timeout in seconds
        """
        self.host = host
        self.port = port
        self.mode = mode
        self.timeout = timeout
        
        self.socket: Optional[socket.socket] = None
        self.connected = False
        self.lock = threading.Lock()
        
        # Standalone simulation state
        self._standalone_state: Optional[SimulationState] = None
        self._hosts: Dict[int, HostState] = {}
        self._vms: Dict[int, VMState] = {}
        
        # Metrics tracking
        self.total_messages_sent = 0
        self.total_messages_received = 0
        self.avg_latency_ms = 0.0
        
    def connect(self) -> bool:
        """
        Establish connection to CloudSim server
        
        Returns:
            True if connected successfully
        """
        if self.mode == SimulationMode.STANDALONE:
            print("Running in STANDALONE mode (Python-only simulation)")
            self.connected = True
            return True
            
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.host, self.port))
            self.connected = True
            print(f"Connected to CloudSim at {self.host}:{self.port}")
            
            # Send handshake
            self._send_message({"type": "HANDSHAKE", "client": "LSTM-DQN-Controller"})
            response = self._receive_message()
            
            if response.get("status") == "OK":
                print(f"Handshake successful. CloudSim version: {response.get('version', 'unknown')}")
                return True
            else:
                print(f"Handshake failed: {response}")
                self.disconnect()
                return False
                
        except socket.error as e:
            print(f"Failed to connect to CloudSim: {e}")
            print("Falling back to STANDALONE mode")
            self.mode = SimulationMode.STANDALONE
            self.connected = True
            return True
    
    def disconnect(self):
        """Close connection to CloudSim"""
        if self.socket:
            try:
                self._send_message({"type": "DISCONNECT"})
                self.socket.close()
            except:
                pass
        self.socket = None
        self.connected = False
        print("Disconnected from CloudSim")
    
    def _send_message(self, message: dict):
        """Send JSON message to CloudSim"""
        if self.mode == SimulationMode.STANDALONE:
            return
            
        with self.lock:
            try:
                data = json.dumps(message).encode('utf-8')
                # Send length prefix (4 bytes) + data
                length = len(data)
                self.socket.sendall(length.to_bytes(4, 'big') + data)
                self.total_messages_sent += 1
            except socket.error as e:
                print(f"Error sending message: {e}")
                raise
    
    def _receive_message(self) -> dict:
        """Receive JSON message from CloudSim"""
        if self.mode == SimulationMode.STANDALONE:
            return {"status": "OK"}
            
        with self.lock:
            try:
                # Receive length prefix
                length_data = self.socket.recv(4)
                if not length_data:
                    return {}
                length = int.from_bytes(length_data, 'big')
                
                # Receive data
                data = b''
                while len(data) < length:
                    chunk = self.socket.recv(min(4096, length - len(data)))
                    if not chunk:
                        break
                    data += chunk
                
                self.total_messages_received += 1
                return json.loads(data.decode('utf-8'))
            except socket.error as e:
                print(f"Error receiving message: {e}")
                return {}
    
    def initialize_datacenter(self, num_hosts: int, num_vms: int,
                              host_config: dict = None, vm_config: dict = None) -> bool:
        """
        Initialize the data center in CloudSim
        
        Args:
            num_hosts: Number of physical hosts
            num_vms: Number of virtual machines
            host_config: Host configuration (MIPS, RAM, etc.)
            vm_config: VM configuration
            
        Returns:
            True if initialization successful
        """
        host_config = host_config or {
            "mips": 10000,
            "ram": 16384,  # MB
            "storage": 1000000,  # MB
            "bandwidth": 10000,  # Mbps
            "power_model": {
                "idle_power": 70,
                "max_power": 250
            }
        }
        
        vm_config = vm_config or {
            "mips": 1000,
            "ram": 1024,
            "bandwidth": 1000
        }
        
        if self.mode == SimulationMode.CLOUDSIM:
            message = {
                "type": "INIT_DATACENTER",
                "num_hosts": num_hosts,
                "num_vms": num_vms,
                "host_config": host_config,
                "vm_config": vm_config
            }
            self._send_message(message)
            response = self._receive_message()
            return response.get("status") == "OK"
        else:
            # Standalone initialization
            self._hosts = {}
            self._vms = {}
            
            for i in range(num_hosts):
                self._hosts[i] = HostState(
                    host_id=i,
                    cpu_utilization=0.0,
                    ram_utilization=0.0,
                    power_consumption=host_config["power_model"]["idle_power"],
                    num_vms=0,
                    is_active=True,
                    available_mips=host_config["mips"],
                    available_ram=host_config["ram"]
                )
            
            vms_per_host = num_vms // num_hosts
            for i in range(num_vms):
                host_id = i // vms_per_host if vms_per_host > 0 else i % num_hosts
                self._vms[i] = VMState(
                    vm_id=i,
                    host_id=min(host_id, num_hosts - 1),
                    cpu_requested=vm_config["mips"],
                    ram_requested=vm_config["ram"],
                    cpu_utilization=0.0
                )
                self._hosts[min(host_id, num_hosts - 1)].num_vms += 1
            
            return True
    
    def get_simulation_state(self) -> SimulationState:
        """
        Get current simulation state from CloudSim
        
        Returns:
            SimulationState object
        """
        if self.mode == SimulationMode.CLOUDSIM:
            start_time = time.time()
            self._send_message({"type": "GET_STATE"})
            response = self._receive_message()
            latency = (time.time() - start_time) * 1000
            self.avg_latency_ms = (self.avg_latency_ms * 0.9) + (latency * 0.1)
            
            hosts = [HostState(**h) for h in response.get("hosts", [])]
            vms = [VMState(**v) for v in response.get("vms", [])]
            
            return SimulationState(
                timestamp=response.get("timestamp", 0),
                hosts=hosts,
                vms=vms,
                total_energy=response.get("total_energy", 0),
                sla_violations=response.get("sla_violations", 0),
                completed_migrations=response.get("completed_migrations", 0)
            )
        else:
            # Standalone mode - return current state
            return SimulationState(
                timestamp=time.time(),
                hosts=list(self._hosts.values()),
                vms=list(self._vms.values()),
                total_energy=sum(h.power_consumption for h in self._hosts.values()),
                sla_violations=sum(1 for h in self._hosts.values() if h.cpu_utilization > 80),
                completed_migrations=0
            )
    
    def execute_migration(self, vm_id: int, source_host: int, target_host: int) -> Tuple[bool, dict]:
        """
        Execute VM migration in CloudSim
        
        Args:
            vm_id: VM to migrate
            source_host: Source host ID
            target_host: Target host ID
            
        Returns:
            Tuple of (success, metrics)
        """
        if self.mode == SimulationMode.CLOUDSIM:
            message = {
                "type": "MIGRATE_VM",
                "vm_id": vm_id,
                "source_host": source_host,
                "target_host": target_host,
                "timestamp": time.time()
            }
            self._send_message(message)
            response = self._receive_message()
            
            return (
                response.get("status") == "OK",
                {
                    "migration_time": response.get("migration_time", 0),
                    "energy_cost": response.get("energy_cost", 0),
                    "downtime": response.get("downtime", 0)
                }
            )
        else:
            # Standalone migration simulation
            if vm_id in self._vms and target_host in self._hosts:
                vm = self._vms[vm_id]
                if vm.host_id == source_host:
                    # Update source host
                    self._hosts[source_host].num_vms -= 1
                    # Update target host
                    self._hosts[target_host].num_vms += 1
                    # Update VM
                    vm.host_id = target_host
                    
                    return True, {
                        "migration_time": 0.5,  # Simulated 500ms
                        "energy_cost": 5.0,     # Wh
                        "downtime": 0.1         # 100ms
                    }
            return False, {}
    
    def set_host_utilization(self, host_id: int, cpu_util: float, ram_util: float = None):
        """
        Set host utilization (for standalone simulation or testing)
        
        Args:
            host_id: Host ID
            cpu_util: CPU utilization (0-100)
            ram_util: RAM utilization (0-100), defaults to cpu_util
        """
        if host_id in self._hosts:
            ram_util = ram_util if ram_util is not None else cpu_util
            host = self._hosts[host_id]
            host.cpu_utilization = cpu_util
            host.ram_utilization = ram_util
            
            # Update power consumption using linear model
            idle_power = 70
            max_power = 250
            if cpu_util > 0:
                host.power_consumption = idle_power + (max_power - idle_power) * (cpu_util / 100)
            else:
                host.power_consumption = 10  # Sleep power
            
            host.is_active = cpu_util > 0
    
    def consolidate_host(self, host_id: int) -> Tuple[bool, List[MigrationCommand]]:
        """
        Request CloudSim to consolidate (turn off) a host
        
        Args:
            host_id: Host to consolidate
            
        Returns:
            Tuple of (success, list of migrations performed)
        """
        if self.mode == SimulationMode.CLOUDSIM:
            message = {
                "type": "CONSOLIDATE_HOST",
                "host_id": host_id
            }
            self._send_message(message)
            response = self._receive_message()
            
            migrations = [
                MigrationCommand(**m) for m in response.get("migrations", [])
            ]
            return response.get("status") == "OK", migrations
        else:
            # Standalone: migrate all VMs from host
            migrations = []
            vms_to_migrate = [v for v in self._vms.values() if v.host_id == host_id]
            
            for vm in vms_to_migrate:
                # Find target host with lowest utilization
                target = min(
                    [h for h in self._hosts.values() if h.host_id != host_id and h.is_active],
                    key=lambda h: h.cpu_utilization,
                    default=None
                )
                if target:
                    success, _ = self.execute_migration(vm.vm_id, host_id, target.host_id)
                    if success:
                        migrations.append(MigrationCommand(
                            vm_id=vm.vm_id,
                            source_host_id=host_id,
                            target_host_id=target.host_id,
                            timestamp=time.time()
                        ))
            
            # Turn off host if empty
            if self._hosts[host_id].num_vms == 0:
                self._hosts[host_id].is_active = False
                self._hosts[host_id].power_consumption = 10  # Sleep power
            
            return True, migrations
    
    def step_simulation(self, time_step: float = 1.0) -> SimulationState:
        """
        Advance simulation by one time step
        
        Args:
            time_step: Time to advance (seconds)
            
        Returns:
            New simulation state
        """
        if self.mode == SimulationMode.CLOUDSIM:
            message = {
                "type": "STEP",
                "time_step": time_step
            }
            self._send_message(message)
            response = self._receive_message()
            
            # Parse and return new state
            return self.get_simulation_state()
        else:
            # Standalone: just return current state
            return self.get_simulation_state()
    
    def get_host_utilizations(self) -> np.ndarray:
        """Get array of host CPU utilizations"""
        state = self.get_simulation_state()
        return np.array([h.cpu_utilization for h in state.hosts])
    
    def get_active_host_count(self) -> int:
        """Get number of active hosts"""
        state = self.get_simulation_state()
        return sum(1 for h in state.hosts if h.is_active)
    
    def get_statistics(self) -> dict:
        """Get bridge statistics"""
        return {
            "mode": self.mode.value,
            "connected": self.connected,
            "messages_sent": self.total_messages_sent,
            "messages_received": self.total_messages_received,
            "avg_latency_ms": round(self.avg_latency_ms, 2)
        }


class FFDHostSelector:
    """
    First Fit Decreasing (FFD) heuristic for candidate host selection
    Filters and ranks target hosts for VM migrations
    """
    
    def __init__(self, num_candidates: int = 5):
        """
        Initialize FFD selector
        
        Args:
            num_candidates: Number of top candidate hosts to return
        """
        self.num_candidates = num_candidates
    
    def select_underutilized_hosts(self, hosts: List[HostState], 
                                    threshold: float = 30.0) -> List[int]:
        """
        Select underutilized hosts that should be consolidated
        
        Args:
            hosts: List of host states
            threshold: Utilization threshold (%)
            
        Returns:
            List of host IDs to consolidate
        """
        underutilized = [
            h.host_id for h in hosts 
            if h.is_active and h.cpu_utilization < threshold and h.num_vms > 0
        ]
        # Sort by utilization (lowest first)
        underutilized.sort(key=lambda hid: next(h.cpu_utilization for h in hosts if h.host_id == hid))
        return underutilized
    
    def select_target_hosts_ffd(self, hosts: List[HostState], 
                                 vm_cpu_requirement: float,
                                 exclude_host: int = None) -> List[int]:
        """
        Select top candidate target hosts using FFD heuristic
        
        FFD sorts hosts by decreasing available capacity and selects
        the first hosts that can fit the VM.
        
        Args:
            hosts: List of host states
            vm_cpu_requirement: CPU requirement of VM to migrate
            exclude_host: Host ID to exclude (source host)
            
        Returns:
            List of top candidate host IDs
        """
        # Filter active hosts with enough capacity
        candidates = [
            h for h in hosts 
            if h.is_active 
            and h.host_id != exclude_host
            and h.cpu_utilization + vm_cpu_requirement <= 80  # Don't overload
        ]
        
        # Sort by decreasing available capacity (FFD)
        candidates.sort(key=lambda h: (100 - h.cpu_utilization), reverse=True)
        
        # Return top candidates
        return [h.host_id for h in candidates[:self.num_candidates]]
    
    def select_overloaded_hosts(self, hosts: List[HostState],
                                 threshold: float = 80.0) -> List[int]:
        """
        Select overloaded hosts that need load balancing
        
        Args:
            hosts: List of host states
            threshold: Overload threshold (%)
            
        Returns:
            List of overloaded host IDs
        """
        overloaded = [
            h.host_id for h in hosts
            if h.is_active and h.cpu_utilization > threshold
        ]
        # Sort by utilization (highest first - most critical)
        overloaded.sort(
            key=lambda hid: next(h.cpu_utilization for h in hosts if h.host_id == hid),
            reverse=True
        )
        return overloaded
    
    def get_migration_candidates(self, hosts: List[HostState], vms: List[VMState],
                                  source_host_id: int) -> List[Tuple[int, List[int]]]:
        """
        Get VM migration candidates with target hosts for a source host
        
        Args:
            hosts: List of host states
            vms: List of VM states
            source_host_id: Source host ID
            
        Returns:
            List of (vm_id, [target_host_ids]) tuples
        """
        source_vms = [v for v in vms if v.host_id == source_host_id]
        
        candidates = []
        for vm in source_vms:
            targets = self.select_target_hosts_ffd(
                hosts, 
                vm.cpu_utilization,
                exclude_host=source_host_id
            )
            if targets:
                candidates.append((vm.vm_id, targets))
        
        return candidates


# Example usage and testing
if __name__ == "__main__":
    print("Testing CloudSim Bridge...")
    
    # Test standalone mode
    bridge = CloudSimBridge(mode=SimulationMode.STANDALONE)
    bridge.connect()
    
    # Initialize datacenter
    bridge.initialize_datacenter(num_hosts=5, num_vms=20)
    
    # Set some utilizations
    bridge.set_host_utilization(0, 75.0)
    bridge.set_host_utilization(1, 45.0)
    bridge.set_host_utilization(2, 20.0)
    bridge.set_host_utilization(3, 85.0)  # Overloaded
    bridge.set_host_utilization(4, 15.0)  # Underutilized
    
    # Get state
    state = bridge.get_simulation_state()
    print(f"\nSimulation State:")
    print(f"  Total Energy: {state.total_energy:.2f} W")
    print(f"  SLA Violations: {state.sla_violations}")
    print(f"  Active Hosts: {bridge.get_active_host_count()}")
    
    # Test FFD selector
    print("\n\nTesting FFD Selector...")
    ffd = FFDHostSelector(num_candidates=3)
    
    underutilized = ffd.select_underutilized_hosts(state.hosts)
    print(f"Underutilized hosts: {underutilized}")
    
    overloaded = ffd.select_overloaded_hosts(state.hosts)
    print(f"Overloaded hosts: {overloaded}")
    
    targets = ffd.select_target_hosts_ffd(state.hosts, vm_cpu_requirement=10, exclude_host=3)
    print(f"Target hosts for migration (FFD): {targets}")
    
    # Print statistics
    print(f"\nBridge Statistics: {bridge.get_statistics()}")
    
    bridge.disconnect()
    print("\n✅ CloudSim Bridge test complete!")
