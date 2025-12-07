"""
Hybrid LSTM-DQN Controller for VM Consolidation
Integrates LSTM workload prediction with DQN decision-making
Supports CloudSim integration via socket-based API
"""

import numpy as np
import pandas as pd
try:
    from .lstm_predictor import LSTMPredictor, MultiResourceLSTM
    from .dqn_agent import DQNAgent, ConsolidationAction
    from .cloudsim_bridge import CloudSimBridge, FFDHostSelector, SimulationMode
except ImportError:
    from lstm_predictor import LSTMPredictor, MultiResourceLSTM
    from dqn_agent import DQNAgent, ConsolidationAction
    from cloudsim_bridge import CloudSimBridge, FFDHostSelector, SimulationMode
import json
import time
from datetime import datetime


class HybridController:
    """
    Main controller integrating LSTM prediction and DQN decision-making
    Supports both standalone Python simulation and CloudSim (Java) integration
    """
    
    def __init__(self, num_hosts, num_vms, lstm_sequence_length=10,
                 target_update_frequency=10, config=None,
                 use_cloudsim=False, cloudsim_host="localhost", cloudsim_port=9999):
        """
        Initialize Hybrid Controller
        
        Args:
            num_hosts: Number of physical hosts
            num_vms: Number of virtual machines
            lstm_sequence_length: Lookback window for LSTM
            target_update_frequency: How often to update DQN target network
            config: Configuration dictionary
            use_cloudsim: Whether to connect to CloudSim (Java)
            cloudsim_host: CloudSim server host
            cloudsim_port: CloudSim server port
        """
        self.num_hosts = num_hosts
        self.num_vms = num_vms
        self.lstm_sequence_length = lstm_sequence_length
        self.target_update_frequency = target_update_frequency
        
        # Configuration
        self.config = config or self._default_config()
        
        # CloudSim Bridge
        self.use_cloudsim = use_cloudsim
        sim_mode = SimulationMode.CLOUDSIM if use_cloudsim else SimulationMode.STANDALONE
        self.cloudsim_bridge = CloudSimBridge(
            host=cloudsim_host,
            port=cloudsim_port,
            mode=sim_mode
        )
        
        # FFD Heuristic for candidate filtering
        self.ffd_selector = FFDHostSelector(num_candidates=5)
        
        # LSTM Predictor for each host
        self.predictors = {
            f'host_{i}': LSTMPredictor(
                sequence_length=lstm_sequence_length,
                prediction_horizon=1
            ) for i in range(num_hosts)
        }
        
        # DQN Agent with updated state size
        # State: [U1...Un, Û1...Ûn, active_hosts_ratio]
        state_size = num_hosts * 2 + 1  # current_util + predicted_util + active_hosts
        
        # Action space: filtered by FFD (top 5 targets per source) + do_nothing
        # Maximum actions = num_hosts * 5 + 1
        action_size = num_hosts * 5 + 1
        
        self.dqn_agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=self.config['learning_rate'],
            gamma=self.config['gamma'],
            epsilon_start=self.config['epsilon_start'],
            epsilon_end=self.config['epsilon_end'],
            epsilon_decay=self.config['epsilon_decay']
        )
        
        # Action mapping (updated dynamically based on FFD filtering)
        self.action_mapping = {}
        self._update_action_mapping()
        
        # History tracking
        self.utilization_history = {f'host_{i}': [] for i in range(num_hosts)}
        self.prediction_history = []
        self.action_history = []
        self.reward_history = []
        
        # Metrics
        self.total_energy_consumed = 0
        self.total_sla_violations = 0
        self.total_migrations = 0
        self.episode_count = 0
        
        # Initialize CloudSim bridge
        self._initialize_simulation()
    
    def _initialize_simulation(self):
        """Initialize the simulation environment"""
        self.cloudsim_bridge.connect()
        self.cloudsim_bridge.initialize_datacenter(
            num_hosts=self.num_hosts,
            num_vms=self.num_vms,
            host_config={
                "mips": 10000,
                "ram": 16384,
                "storage": 1000000,
                "bandwidth": 10000,
                "power_model": self.config['energy_model']
            }
        )
    
    def _update_action_mapping(self):
        """
        Update action mapping based on FFD-filtered candidates
        This is called each step to dynamically filter valid actions
        """
        self.action_mapping = {0: ('do_nothing', None, None)}
        action_idx = 1
        
        # For each source host, map to top 5 FFD targets
        for source in range(self.num_hosts):
            for target_rank in range(5):  # Top 5 candidates
                if action_idx < self.dqn_agent.action_size:
                    self.action_mapping[action_idx] = ('migrate', source, target_rank)
                    action_idx += 1
    
    def _default_config(self):
        """Default configuration"""
        return {
            'learning_rate': 0.001,
            'gamma': 0.95,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'reward_weights': {
                'energy': 0.5,
                'sla': 0.3,
                'migration': 0.2
            },
            'energy_model': {
                'idle_power': 70,  # Watts
                'max_power': 250,  # Watts
                'sleep_power': 10  # Watts
            },
            'sla_threshold': 0.8,  # 80% utilization threshold
            'consolidation_threshold': 0.3  # Below 30% -> consider consolidation
        }
    
    def update_utilization(self, host_id, utilization_value):
        """
        Update utilization history for a host
        
        Args:
            host_id: Host identifier
            utilization_value: Current CPU utilization (0-100)
        """
        self.utilization_history[f'host_{host_id}'].append(utilization_value)
    
    def predict_workload(self):
        """
        Predict future workload for all hosts using LSTM
        
        Returns:
            predictions: Dictionary of predictions for each host
            trends: Dictionary of trends for each host
        """
        predictions = {}
        trends = {}
        
        for host_id in range(self.num_hosts):
            host_key = f'host_{host_id}'
            history = self.utilization_history[host_key]
            
            # Need enough history for prediction
            if len(history) >= self.lstm_sequence_length:
                sequence = np.array(history[-self.lstm_sequence_length:])
                
                # Predict if model is trained
                if self.predictors[host_key].is_trained:
                    pred = self.predictors[host_key].predict(sequence)
                    trend = self.predictors[host_key].predict_trend(sequence)
                else:
                    # Use simple moving average as fallback
                    pred = np.array([np.mean(sequence)])
                    trend = 'stable'
                
                predictions[host_id] = pred[0]
                trends[host_id] = trend
            else:
                # Not enough data
                predictions[host_id] = history[-1] if history else 0
                trends[host_id] = 'stable'
        
        return predictions, trends
    
    def get_current_state(self, hosts_utilization, predictions):
        """
        Create state representation for DQN
        
        Args:
            hosts_utilization: Current utilization of all hosts
            predictions: LSTM predictions for all hosts
            
        Returns:
            state: State vector for DQN
        """
        # State vector: S = [U1...Un, Û1...Ûn, active_hosts_ratio]
        # As per report: S = [U1, ..., Un, Û1, ..., Ûn, active_hosts]
        
        # Current utilizations (normalized to 0-1)
        current_util = np.array(hosts_utilization) / 100.0
        
        # Predicted utilizations from LSTM (normalized to 0-1)
        pred_util = np.array([predictions.get(i, 0) for i in range(self.num_hosts)]) / 100.0
        
        # Active hosts ratio (normalized)
        active_hosts = sum(1 for util in hosts_utilization if util > 0) / self.num_hosts
        
        # Concatenate: [U1...Un, Û1...Ûn, active_hosts]
        state = np.concatenate([
            current_util,      # Current utilizations
            pred_util,         # LSTM predictions
            [active_hosts]     # Active hosts ratio
        ])
        
        return state
    
    def get_ffd_filtered_targets(self, hosts_utilization, source_host):
        """
        Get FFD-filtered target hosts for a migration
        
        Args:
            hosts_utilization: Current host utilizations
            source_host: Source host ID
            
        Returns:
            List of top 5 candidate target host IDs
        """
        # Get simulation state
        state = self.cloudsim_bridge.get_simulation_state()
        
        # Use FFD to select top 5 target candidates
        vm_cpu_requirement = 10.0  # Estimated VM CPU requirement
        targets = self.ffd_selector.select_target_hosts_ffd(
            state.hosts,
            vm_cpu_requirement,
            exclude_host=source_host
        )
        
        return targets
    
    def decode_action(self, action_idx, hosts_utilization=None):
        """
        Decode action index to consolidation action using FFD filtering
        
        Args:
            action_idx: Action index from DQN
            hosts_utilization: Current host utilizations (for FFD filtering)
            
        Returns:
            ConsolidationAction object
        """
        # Action 0 is "do nothing"
        if action_idx == 0:
            return ConsolidationAction('do_nothing')
        
        # Decode using action mapping
        if action_idx in self.action_mapping:
            action_type, source_host, target_rank = self.action_mapping[action_idx]
            
            if action_type == 'do_nothing':
                return ConsolidationAction('do_nothing')
            
            # Get FFD-filtered targets for source host
            if hosts_utilization is not None:
                targets = self.get_ffd_filtered_targets(hosts_utilization, source_host)
                if target_rank < len(targets):
                    target_host = targets[target_rank]
                else:
                    return ConsolidationAction('do_nothing')
            else:
                # Fallback: use target_rank directly
                target_host = target_rank
            
            return ConsolidationAction(
                'migrate',
                vm_id=None,
                source_host=source_host,
                target_host=target_host
            )
        
        return ConsolidationAction('do_nothing')
    
    def decode_action_legacy(self, action_idx):
        """
        Legacy decode action (without FFD filtering)
        
        Args:
            action_idx: Action index from DQN
            
        Returns:
            ConsolidationAction object
        """
        # Last action is "do nothing"
        if action_idx == self.num_hosts * self.num_hosts:
            return ConsolidationAction('do_nothing')
        
        # Decode migration: action = source * num_hosts + target
        source_host = action_idx // self.num_hosts
        target_host = action_idx % self.num_hosts
        
        # Cannot migrate to same host
        if source_host == target_host:
            return ConsolidationAction('do_nothing')
        
        return ConsolidationAction(
            'migrate',
            vm_id=None,  # Will be determined by VM selection policy
            source_host=source_host,
            target_host=target_host
        )
    
    def calculate_energy(self, hosts_utilization):
        """
        Calculate energy consumption based on utilization
        
        Args:
            hosts_utilization: List of host utilizations (0-100)
            
        Returns:
            energy: Energy consumed in Watts
        """
        idle_power = self.config['energy_model']['idle_power']
        max_power = self.config['energy_model']['max_power']
        
        total_energy = 0
        for util in hosts_utilization:
            if util == 0:
                # Host is off/sleep
                total_energy += self.config['energy_model']['sleep_power']
            else:
                # Linear power model: P = P_idle + (P_max - P_idle) * utilization
                total_energy += idle_power + (max_power - idle_power) * (util / 100.0)
        
        return total_energy
    
    def calculate_sla_violations(self, hosts_utilization):
        """
        Calculate SLA violations (hosts above threshold)
        
        Args:
            hosts_utilization: List of host utilizations
            
        Returns:
            sla_violations: Number of violations
        """
        threshold = self.config['sla_threshold'] * 100
        violations = sum(1 for util in hosts_utilization if util > threshold)
        return violations
    
    def step(self, hosts_utilization):
        """
        Perform one consolidation step
        
        Args:
            hosts_utilization: Current utilization of all hosts
            
        Returns:
            action: Selected consolidation action
            reward: Received reward
            metrics: Dictionary of metrics
        """
        # Update history
        for host_id, util in enumerate(hosts_utilization):
            self.update_utilization(host_id, util)
        
        # Predict future workload
        predictions, trends = self.predict_workload()
        
        # Get current state
        state = self.get_current_state(hosts_utilization, predictions)
        
        # Select action
        action_idx = self.dqn_agent.select_action(state)
        action = self.decode_action(action_idx)
        
        # Execute action and calculate metrics
        # Update CloudSim bridge with current utilizations
        for host_id, util in enumerate(hosts_utilization):
            self.cloudsim_bridge.set_host_utilization(host_id, util)
        
        energy_before = self.calculate_energy(hosts_utilization)
        sla_violations_before = self.calculate_sla_violations(hosts_utilization)
        
        # Execute action via CloudSim bridge or simulation
        migration_metrics = {}
        if action.action_type == 'migrate' and action.source_host is not None:
            success, migration_metrics = self.cloudsim_bridge.execute_migration(
                vm_id=action.vm_id or 0,  # Default VM ID if not specified
                source_host=action.source_host,
                target_host=action.target_host
            )
            if not success:
                action = ConsolidationAction('do_nothing')
        
        # Get updated state from CloudSim
        sim_state = self.cloudsim_bridge.get_simulation_state()
        hosts_utilization_after = [h.cpu_utilization for h in sim_state.hosts]
        
        # If standalone mode, simulate the action effect
        if not hosts_utilization_after or all(u == 0 for u in hosts_utilization_after):
            hosts_utilization_after = self._simulate_action(hosts_utilization, action)
        
        energy_after = self.calculate_energy(hosts_utilization_after)
        sla_violations_after = self.calculate_sla_violations(hosts_utilization_after)
        migrations = 1 if action.action_type == 'migrate' else 0
        
        # Calculate reward
        energy_delta = (energy_before - energy_after) / energy_before if energy_before > 0 else 0
        sla_delta = (sla_violations_before - sla_violations_after)
        
        # Get reward weights from config
        weights = self.config['reward_weights']
        reward = self.dqn_agent.compute_reward(
            energy_consumed=-energy_delta,  # Negative because we want reduction
            sla_violations=sla_violations_after / self.num_hosts,
            num_migrations=migrations,
            w_energy=weights.get('energy', 0.5),
            w_sla=weights.get('sla', 0.3),
            w_migration=weights.get('migration', 0.2)
        )
        
        # Get next state
        next_predictions, _ = self.predict_workload()
        next_state = self.get_current_state(hosts_utilization_after, next_predictions)
        
        # Store experience
        self.dqn_agent.store_experience(state, action_idx, reward, next_state, False)
        
        # Train agent
        loss = self.dqn_agent.train()
        
        # Update target network periodically
        if self.dqn_agent.training_step % self.target_update_frequency == 0:
            self.dqn_agent.update_target_network()
        
        # Update metrics
        self.total_energy_consumed += energy_after
        self.total_sla_violations += sla_violations_after
        self.total_migrations += migrations
        
        # Store history
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.prediction_history.append(predictions)
        
        metrics = {
            'energy': energy_after,
            'energy_delta': energy_delta,
            'sla_violations': sla_violations_after,
            'migrations': migrations,
            'reward': reward,
            'loss': loss,
            'epsilon': self.dqn_agent.epsilon,
            'predictions': predictions,
            'trends': trends,
            'migration_time': migration_metrics.get('migration_time', 0),
            'migration_energy_cost': migration_metrics.get('energy_cost', 0)
        }
        
        return action, reward, metrics
    
    def _simulate_action(self, hosts_utilization, action):
        """
        Simulate action execution (fallback when CloudSim is not connected)
        
        Args:
            hosts_utilization: Current utilization
            action: ConsolidationAction
            
        Returns:
            new_utilization: Updated utilization after action
        """
        new_utilization = hosts_utilization.copy()
        
        if action.action_type == 'migrate':
            # Simple simulation: reduce source, increase target
            source = action.source_host
            target = action.target_host
            
            if source is not None and target is not None:
                # Transfer 10% load as example
                transfer = min(new_utilization[source], 10)
                new_utilization[source] -= transfer
                new_utilization[target] = min(100, new_utilization[target] + transfer)
        
        return new_utilization
    
    def train_lstm_predictors(self, historical_data, epochs=50):
        """
        Train LSTM predictors with historical data
        
        Args:
            historical_data: Dictionary mapping host_id to historical utilization
            epochs: Number of training epochs
        """
        print("Training LSTM predictors...")
        for host_id in range(self.num_hosts):
            host_key = f'host_{host_id}'
            if host_key in historical_data and len(historical_data[host_key]) > 100:
                data = np.array(historical_data[host_key])
                print(f"Training predictor for {host_key}...")
                self.predictors[host_key].train(data, epochs=epochs, verbose=0)
        print("LSTM training completed!")
    
    def save(self, filepath='models/hybrid_controller'):
        """Save controller state"""
        # Save DQN agent
        self.dqn_agent.save(f'{filepath}_dqn.pth')
        
        # Save LSTM predictors
        for host_id in range(self.num_hosts):
            host_key = f'host_{host_id}'
            if self.predictors[host_key].is_trained:
                self.predictors[host_key].save_model(
                    f'{filepath}_lstm_{host_id}.h5',
                    f'{filepath}_scaler_{host_id}.pkl'
                )
        
        # Save metrics
        metrics = {
            'total_energy': self.total_energy_consumed,
            'total_sla_violations': self.total_sla_violations,
            'total_migrations': self.total_migrations,
            'episode_count': self.episode_count,
            'reward_history': self.reward_history
        }
        
        with open(f'{filepath}_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Controller saved to {filepath}")
    
    def load(self, filepath='models/hybrid_controller'):
        """Load controller state"""
        # Load DQN agent
        self.dqn_agent.load(f'{filepath}_dqn.pth')
        
        # Load LSTM predictors
        for host_id in range(self.num_hosts):
            try:
                self.predictors[f'host_{host_id}'].load_model(
                    f'{filepath}_lstm_{host_id}.h5',
                    f'{filepath}_scaler_{host_id}.pkl'
                )
            except:
                print(f"Warning: Could not load LSTM for host_{host_id}")
        
        print(f"Controller loaded from {filepath}")
    
    def get_metrics_summary(self):
        """Get summary of performance metrics"""
        return {
            'total_energy_kwh': self.total_energy_consumed / 1000.0,
            'avg_energy_per_step': np.mean([m for m in self.total_energy_consumed]) if self.episode_count > 0 else 0,
            'total_sla_violations': self.total_sla_violations,
            'total_migrations': self.total_migrations,
            'avg_reward': np.mean(self.reward_history) if self.reward_history else 0,
            'episodes': self.episode_count
        }


if __name__ == "__main__":
    # Example usage
    print("Testing Hybrid LSTM-DQN Controller...\n")
    
    # Configuration
    num_hosts = 5
    num_vms = 20
    
    # Initialize controller
    controller = HybridController(num_hosts=num_hosts, num_vms=num_vms)
    
    # Simulate workload
    print("Simulating consolidation episodes...")
    for episode in range(10):
        # Random initial utilization
        hosts_util = np.random.rand(num_hosts) * 100
        
        # Perform consolidation step
        action, reward, metrics = controller.step(hosts_util)
        
        print(f"\nEpisode {episode + 1}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward:.4f}")
        print(f"  Energy: {metrics['energy']:.2f}W")
        print(f"  SLA Violations: {metrics['sla_violations']}")
        print(f"  Epsilon: {metrics['epsilon']:.4f}")
    
    # Summary
    print("\n" + "="*50)
    print("Performance Summary:")
    summary = controller.get_metrics_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Save controller
    controller.save()
    print("\nController saved successfully!")
