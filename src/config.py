# Green Cloud LSTM-DQN Project Configuration

## Project Metadata
PROJECT_NAME = "Green_Cloud_LSTM_DQN"
VERSION = "1.0.0"
DESCRIPTION = "Energy Optimization in Data Centers using Hybrid LSTM-DQN Approach"

## Environment Configuration
NUM_HOSTS = 10
NUM_VMS = 50
SEQUENCE_LENGTH = 10

## LSTM Configuration
LSTM_CONFIG = {
    'units': 64,
    'dropout_rate': 0.2,
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 0.001
}

## DQN Configuration
DQN_CONFIG = {
    'learning_rate': 0.001,
    'gamma': 0.99,
    'epsilon': 1.0,
    'epsilon_decay': 0.995,
    'epsilon_min': 0.01,
    'replay_buffer_size': 10000,
    'batch_size': 64,
    'target_update_frequency': 10
}

## Training Configuration
TRAINING_CONFIG = {
    'num_episodes': 100,
    'steps_per_episode': 100,
    'validation_split': 0.2,
    'test_split': 0.2
}

## Energy Model Configuration
ENERGY_CONFIG = {
    'p_idle': 50,      # Watts
    'p_max': 250,      # Watts
    'p_sleep': 5,      # Watts
    'migration_cost': 10  # Watts per migration
}

## Reward Weights (Multi-Objective Optimization)
REWARD_WEIGHTS = {
    'energy': 0.4,
    'sla': 0.4,
    'migration': 0.2
}

## SLA Thresholds
SLA_CONFIG = {
    'cpu_threshold_upper': 80,  # %
    'cpu_threshold_lower': 20,  # %
    'ram_threshold_upper': 80,  # %
    'ram_threshold_lower': 20   # %
}

## Paths
PATHS = {
    'data': '../data',
    'models': '../models',
    'results': '../results',
    'graphs': '../results/graphs',
    'metrics': '../results/metrics'
}

## Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': '../logs/experiment.log'
}
