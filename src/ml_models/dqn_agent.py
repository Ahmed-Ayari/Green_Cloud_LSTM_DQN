"""
Deep Q-Network (DQN) Agent for VM Consolidation Decisions
Makes intelligent migration and placement decisions to minimize energy consumption
while maintaining QoS
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple
import pickle


# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class DQNNetwork(nn.Module):
    """
    Deep Q-Network architecture
    """
    
    def __init__(self, state_size, action_size, hidden_size=128):
        """
        Initialize DQN
        
        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            hidden_size: Size of hidden layers
        """
        super(DQNNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            
            nn.Linear(hidden_size // 2, action_size)
        )
    
    def forward(self, state):
        """Forward pass"""
        return self.network(state)


class ReplayBuffer:
    """
    Experience replay buffer for stable training
    """
    
    def __init__(self, capacity=10000):
        """
        Initialize replay buffer
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample random batch of experiences"""
        experiences = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor([e.state for e in experiences])
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor([e.next_state for e in experiences])
        dones = torch.FloatTensor([e.done for e in experiences])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return current buffer size"""
        return len(self.buffer)


class DQNAgent:
    """
    DQN Agent for VM Consolidation
    """
    
    def __init__(self, state_size, action_size, 
                 learning_rate=0.001, gamma=0.95, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 buffer_capacity=10000, batch_size=64):
        """
        Initialize DQN Agent
        
        Args:
            state_size: Dimension of state (host utilizations + predictions)
            action_size: Number of possible actions (migration decisions)
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Epsilon decay rate
            buffer_capacity: Replay buffer capacity
            batch_size: Training batch size
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Q-Networks (online and target)
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Training statistics
        self.training_step = 0
        self.loss_history = []
    
    def get_state_representation(self, hosts_utilization, predicted_utilization, 
                                  vm_resources, num_vms):
        """
        Create state representation for DQN
        
        Args:
            hosts_utilization: Current CPU/RAM utilization of each host
            predicted_utilization: LSTM predictions for each host
            vm_resources: Resource requirements of VMs
            num_vms: Number of active VMs
            
        Returns:
            state: Flattened state vector
        """
        # Combine current and predicted states
        state = np.concatenate([
            hosts_utilization.flatten(),
            predicted_utilization.flatten(),
            [num_vms / 100.0]  # Normalized VM count
        ])
        
        return state
    
    def select_action(self, state, valid_actions=None):
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            valid_actions: List of valid action indices (optional)
            
        Returns:
            action: Selected action index
        """
        # Exploration
        if random.random() < self.epsilon:
            if valid_actions is not None:
                return random.choice(valid_actions)
            return random.randint(0, self.action_size - 1)
        
        # Exploitation
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            
            if valid_actions is not None:
                # Mask invalid actions
                mask = torch.full(q_values.shape, float('-inf')).to(self.device)
                mask[0, valid_actions] = 0
                q_values = q_values + mask
            
            action = q_values.argmax(dim=1).item()
        
        return action
    
    def compute_reward(self, energy_consumed, sla_violations, num_migrations,
                       w_energy=0.5, w_sla=0.3, w_migration=0.2):
        """
        Compute multi-objective reward
        
        Args:
            energy_consumed: Energy consumed in this step (normalized)
            sla_violations: Number/severity of SLA violations (normalized)
            num_migrations: Number of VM migrations performed (normalized)
            w_energy: Weight for energy objective
            w_sla: Weight for SLA objective
            w_migration: Weight for migration cost
            
        Returns:
            reward: Scalar reward value
        """
        # Negative rewards for costs (we want to minimize them)
        reward = (
            - w_energy * energy_consumed
            - w_sla * sla_violations
            - w_migration * num_migrations
        )
        
        return reward
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train(self):
        """
        Train the DQN using experience replay
        
        Returns:
            loss: Training loss
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Compute current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q values (Double DQN)
        with torch.no_grad():
            # Select actions using online network
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            # Evaluate using target network
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            target_q_values = rewards.unsqueeze(1) + self.gamma * next_q_values * (1 - dones.unsqueeze(1))
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update statistics
        self.training_step += 1
        self.loss_history.append(loss.item())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with weights from online network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, filepath='models/dqn_agent.pth'):
        """Save agent state"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'loss_history': self.loss_history
        }, filepath)
        print(f"Agent saved to {filepath}")
    
    def load(self, filepath='models/dqn_agent.pth'):
        """Load agent state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        self.loss_history = checkpoint['loss_history']
        print(f"Agent loaded from {filepath}")


class ConsolidationAction:
    """
    Helper class to represent consolidation actions
    """
    
    def __init__(self, action_type, vm_id=None, source_host=None, target_host=None):
        """
        Args:
            action_type: 'migrate', 'do_nothing', 'shutdown_host', 'wakeup_host'
            vm_id: ID of VM to migrate
            source_host: Source host ID
            target_host: Target host ID
        """
        self.action_type = action_type
        self.vm_id = vm_id
        self.source_host = source_host
        self.target_host = target_host
    
    def __repr__(self):
        if self.action_type == 'migrate':
            return f"Migrate VM {self.vm_id} from Host {self.source_host} to Host {self.target_host}"
        return f"Action: {self.action_type}"


if __name__ == "__main__":
    # Example usage
    print("Testing DQN Agent...")
    
    # Configuration
    num_hosts = 10
    state_size = num_hosts * 2 + 1  # current + predicted utilization + vm_count
    action_size = num_hosts * num_hosts  # All possible migration pairs
    
    # Initialize agent
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    
    # Simulate training
    print("\nSimulating training episodes...")
    for episode in range(5):
        # Random state
        hosts_util = np.random.rand(num_hosts)
        predicted_util = np.random.rand(num_hosts)
        state = agent.get_state_representation(hosts_util, predicted_util, None, 50)
        
        # Select action
        action = agent.select_action(state)
        
        # Simulate reward
        reward = agent.compute_reward(
            energy_consumed=0.5,
            sla_violations=0.1,
            num_migrations=0.05
        )
        
        # Next state
        next_state = agent.get_state_representation(
            np.random.rand(num_hosts),
            np.random.rand(num_hosts),
            None, 48
        )
        
        # Store and train
        agent.store_experience(state, action, reward, next_state, False)
        loss = agent.train()
        
        if loss is not None:
            print(f"Episode {episode + 1}: Loss = {loss:.4f}, Epsilon = {agent.epsilon:.4f}")
    
    # Save agent
    agent.save()
    print("\nAgent saved successfully!")
