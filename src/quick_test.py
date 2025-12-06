"""
Quick Test Script - Runs a shortened experiment to verify pipeline
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

# Set TF logging to minimum
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from ml_models.data_preprocessing import WorkloadDataLoader, prepare_dataset_for_training
from ml_models.lstm_predictor import LSTMPredictor
from ml_models.dqn_agent import DQNAgent
from ml_models.hybrid_controller import HybridController
from ml_models.metrics_evaluation import MetricsCalculator, PerformanceVisualizer


def main():
    print("=" * 70)
    print("LSTM-DQN VM CONSOLIDATION - QUICK TEST")
    print("=" * 70)
    print(f"Start time: {datetime.now()}")
    
    # Configuration for quick test
    config = {
        'num_hosts': 5,  # Reduced
        'num_vms': 20,   # Reduced
        'num_training_episodes': 5,  # Reduced significantly
        'num_evaluation_episodes': 2,  # Reduced
        'lstm_sequence_length': 10,
        'lstm_epochs': 10,  # Reduced epochs for LSTM
        'data_dir': '../data'
    }
    
    # Create output directories
    os.makedirs('results/graphs', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # =====================================================================
    # PHASE 1: Load Data
    # =====================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: Loading PlanetLab Data")
    print("=" * 60)
    
    data_loader = WorkloadDataLoader(config['data_dir'])
    
    try:
        vm_data = data_loader.load_planetlab_from_directory(max_traces=config['num_vms'])
        print(f"✅ Loaded PlanetLab data: {vm_data.shape}")
        data_source = "PlanetLab"
    except Exception as e:
        print(f"⚠️ Could not load PlanetLab: {e}")
        print("Using synthetic data...")
        vm_data = data_loader.generate_synthetic_workload(
            num_vms=config['num_vms'],
            timesteps=500,
            pattern='mixed'
        )
        data_source = "Synthetic"
    
    # Aggregate VMs into hosts
    vms_per_host = config['num_vms'] // config['num_hosts']
    host_data = np.zeros((vm_data.shape[0], config['num_hosts']))
    for h in range(config['num_hosts']):
        start_vm = h * vms_per_host
        end_vm = start_vm + vms_per_host
        host_data[:, h] = np.mean(vm_data[:, start_vm:end_vm], axis=1)
    
    print(f"Aggregated to {config['num_hosts']} hosts: {host_data.shape}")
    
    # =====================================================================
    # PHASE 2: Initialize Controller
    # =====================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Initializing Hybrid Controller")
    print("=" * 60)
    
    controller = HybridController(
        num_hosts=config['num_hosts'],
        num_vms=config['num_vms'],
        lstm_sequence_length=config['lstm_sequence_length']
    )
    
    # =====================================================================
    # PHASE 3: Quick LSTM Training
    # =====================================================================
    print("\n" + "=" * 60)
    print("PHASE 3: Training LSTM Predictors (Quick)")
    print("=" * 60)
    
    # Train LSTM for each host with reduced epochs
    for h in range(config['num_hosts']):
        host_key = f'host_{h}'
        host_workload = host_data[:, h]
        
        # Create and train predictor
        predictor = LSTMPredictor(
            sequence_length=config['lstm_sequence_length'],
            lstm_units=32,    # Reduced for speed
            dropout_rate=0.2
        )
        
        # Train with few epochs
        print(f"Training LSTM for {host_key}...", end=' ')
        predictor.train(
            host_workload,
            epochs=config['lstm_epochs'],
            batch_size=16,
            verbose=0
        )
        print("Done")
        
        controller.predictors[host_key] = predictor
    
    print("✅ LSTM training complete!")
    
    # =====================================================================
    # PHASE 4: Quick DQN Training
    # =====================================================================
    print("\n" + "=" * 60)
    print("PHASE 4: Training DQN Agent (Quick)")
    print("=" * 60)
    
    training_metrics = []
    timesteps = host_data.shape[0]
    
    for episode in range(config['num_training_episodes']):
        episode_reward = 0
        episode_energy = 0
        episode_sla = 0
        
        # Run through a subset of timesteps
        num_steps = min(50, timesteps - config['lstm_sequence_length'] - 1)
        
        for t in range(num_steps):
            step_idx = config['lstm_sequence_length'] + t
            
            # Get current host utilizations
            hosts_util = host_data[step_idx, :]
            
            # Update utilization for each host
            for h in range(config['num_hosts']):
                controller.update_utilization(h, hosts_util[h])
            
            # Take a step
            action, reward, metrics = controller.step(hosts_util)
            
            episode_reward += reward
            episode_energy += metrics.get('energy', 0)
            episode_sla += metrics.get('sla_violations', 0)
        
        # Record metrics
        training_metrics.append({
            'episode': episode + 1,
            'reward': episode_reward,
            'energy': episode_energy / num_steps,
            'sla_violations': episode_sla / num_steps,
            'epsilon': controller.dqn_agent.epsilon
        })
        
        print(f"Episode {episode+1}/{config['num_training_episodes']}: "
              f"Reward={episode_reward:.2f}, Energy={episode_energy/num_steps:.2f}, "
              f"Epsilon={controller.dqn_agent.epsilon:.3f}")
    
    print("✅ DQN training complete!")
    
    # =====================================================================
    # PHASE 5: Quick Evaluation
    # =====================================================================
    print("\n" + "=" * 60)
    print("PHASE 5: Evaluation")
    print("=" * 60)
    
    # Simple evaluation
    controller.dqn_agent.epsilon = 0  # Greedy
    
    eval_energy = []
    eval_sla = []
    
    for step_idx in range(config['lstm_sequence_length'], 
                          min(100, timesteps - 1)):
        hosts_util = host_data[step_idx, :]
        
        # Update utilization for each host
        for h in range(config['num_hosts']):
            controller.update_utilization(h, hosts_util[h])
        
        action, reward, metrics = controller.step(hosts_util)
        
        eval_energy.append(metrics.get('energy', 0))
        eval_sla.append(metrics.get('sla_violations', 0))
    
    print(f"\nEvaluation Results ({data_source} data):")
    print(f"  Average Energy: {np.mean(eval_energy):.2f} Watts")
    print(f"  Average SLA Violations: {np.mean(eval_sla):.2%}")
    print(f"  Total Steps: {len(eval_energy)}")
    
    # =====================================================================
    # PHASE 6: Save Results
    # =====================================================================
    print("\n" + "=" * 60)
    print("PHASE 6: Saving Results")
    print("=" * 60)
    
    # Save metrics
    results = {
        'data_source': data_source,
        'config': config,
        'training_metrics': training_metrics,
        'evaluation': {
            'avg_energy': float(np.mean(eval_energy)),
            'avg_sla': float(np.mean(eval_sla)),
            'total_steps': len(eval_energy)
        }
    }
    
    with open('results/metrics/quick_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✅ Results saved to results/metrics/quick_test_results.json")
    
    # Generate simple visualization
    visualizer = PerformanceVisualizer()
    
    if len(training_metrics) > 1:
        # Training progress plot
        rewards = [m['reward'] for m in training_metrics]
        energies = [m['energy'] for m in training_metrics]
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].plot(rewards, 'b-o')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Total Reward')
        axes[0].set_title('Training Progress - Rewards')
        axes[0].grid(True)
        
        axes[1].plot(energies, 'r-o')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Avg Energy (Watts)')
        axes[1].set_title('Training Progress - Energy')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('results/graphs/quick_test_training.png', dpi=150)
        plt.close()
        
        print("✅ Plot saved to results/graphs/quick_test_training.png")
    
    # =====================================================================
    # SUMMARY
    # =====================================================================
    print("\n" + "=" * 70)
    print("QUICK TEST COMPLETE!")
    print("=" * 70)
    print(f"End time: {datetime.now()}")
    print(f"\nData Source: {data_source}")
    print(f"Hosts: {config['num_hosts']}, VMs: {config['num_vms']}")
    print(f"Training Episodes: {config['num_training_episodes']}")
    print(f"\nFinal Evaluation Metrics:")
    print(f"  - Energy Consumption: {np.mean(eval_energy):.2f} Watts")
    print(f"  - SLA Violations: {np.mean(eval_sla)*100:.1f}%")
    print("\n✅ Pipeline verification successful!")
    print("   You can now run the full experiment with main_experiment.py")


if __name__ == "__main__":
    main()
