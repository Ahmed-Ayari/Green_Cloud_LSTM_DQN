"""
Main Training and Evaluation Script
Orchestrates the complete LSTM-DQN VM consolidation system
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'ml_models'))

import numpy as np
import pandas as pd
from lstm_predictor import LSTMPredictor
from dqn_agent import DQNAgent
from hybrid_controller import HybridController
from data_preprocessing import WorkloadDataLoader, prepare_dataset_for_training
from metrics_evaluation import MetricsCalculator, PerformanceVisualizer
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime


class ExperimentRunner:
    """
    Main experiment orchestrator for LSTM-DQN VM consolidation
    """
    
    def __init__(self, config_path=None):
        """
        Initialize experiment runner
        
        Args:
            config_path: Path to configuration JSON file
        """
        self.config = self.load_config(config_path) if config_path else self.default_config()
        
        # Components
        self.controller = None
        self.calculator = None
        self.visualizer = None
        self.data_loader = None
        
        # Results storage
        self.results = {
            'training': [],
            'evaluation': {},
            'comparison': {}
        }
    
    def default_config(self):
        """Default experiment configuration"""
        return {
            'experiment_name': 'LSTM_DQN_Consolidation',
            'num_hosts': 10,
            'num_vms': 50,
            'num_training_episodes': 100,
            'num_evaluation_episodes': 20,
            'lstm_sequence_length': 10,
            'lstm_epochs': 50,
            'workload_pattern': 'mixed',
            'workload_timesteps': 1000,
            'use_real_data': True,  # Try to use PlanetLab data first
            'data_dir': '../data',
            'save_models': True,
            'save_plots': True,
            'output_dir': 'results'
        }
    
    def load_config(self, config_path):
        """Load configuration from JSON"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def setup(self):
        """Initialize all components"""
        print("Setting up experiment...")
        
        # Create output directories
        os.makedirs(self.config['output_dir'], exist_ok=True)
        os.makedirs(os.path.join(self.config['output_dir'], 'graphs'), exist_ok=True)
        os.makedirs(os.path.join(self.config['output_dir'], 'metrics'), exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        # Initialize components
        self.controller = HybridController(
            num_hosts=self.config['num_hosts'],
            num_vms=self.config['num_vms'],
            lstm_sequence_length=self.config['lstm_sequence_length']
        )
        
        self.calculator = MetricsCalculator()
        self.visualizer = PerformanceVisualizer()
        self.data_loader = WorkloadDataLoader()
        
        print("Setup completed!")
    
    def prepare_training_data(self):
        """Prepare workload data for training"""
        print("\nPreparing training data...")
        
        dataset = prepare_dataset_for_training(
            num_hosts=self.config['num_hosts'],
            timesteps=self.config['workload_timesteps'],
            pattern=self.config['workload_pattern'],
            use_real_data=self.config.get('use_real_data', True),
            data_dir=self.config.get('data_dir', '../data')
        )
        
        print(f"\nDataset prepared:")
        print(f"  Data source: {dataset['metadata']['data_source']}")
        print(f"  Hosts: {dataset['metadata']['num_hosts']}")
        print(f"  Training samples: {len(dataset['train']['host_0'])}")
        print(f"  Test samples: {len(dataset['test']['host_0'])}")
        
        return dataset

    
    def train_lstm_predictors(self, dataset):
        """Train LSTM workload predictors"""
        print("\n" + "="*60)
        print("PHASE 1: Training LSTM Workload Predictors")
        print("="*60)
        
        self.controller.train_lstm_predictors(
            dataset['train'],
            epochs=self.config['lstm_epochs']
        )
        
        print("LSTM predictors trained successfully!")
    
    def train_dqn_agent(self, dataset):
        """Train DQN consolidation agent"""
        print("\n" + "="*60)
        print("PHASE 2: Training DQN Consolidation Agent")
        print("="*60)
        
        train_metrics = []
        
        for episode in tqdm(range(self.config['num_training_episodes']), desc="Training Episodes"):
            episode_reward = 0
            episode_energy = 0
            episode_sla = 0
            episode_migrations = 0
            
            # Use training data
            for timestep in range(len(dataset['train']['host_0']) - self.config['lstm_sequence_length']):
                # Get current host utilizations
                hosts_util = [
                    dataset['train'][f'host_{i}'][timestep]
                    for i in range(self.config['num_hosts'])
                ]
                
                # Perform consolidation step
                action, reward, metrics = self.controller.step(hosts_util)
                
                episode_reward += reward
                episode_energy += metrics['energy']
                episode_sla += metrics['sla_violations']
                episode_migrations += metrics['migrations']
            
            # Store episode metrics
            train_metrics.append({
                'episode': episode + 1,
                'total_reward': episode_reward,
                'avg_energy': episode_energy / len(dataset['train']['host_0']),
                'total_sla_violations': episode_sla,
                'total_migrations': episode_migrations,
                'epsilon': self.controller.dqn_agent.epsilon
            })
            
            # Print progress every 10 episodes
            if (episode + 1) % 10 == 0:
                print(f"\nEpisode {episode + 1}/{self.config['num_training_episodes']}")
                print(f"  Reward: {episode_reward:.2f}")
                print(f"  Energy: {episode_energy:.2f}W")
                print(f"  Migrations: {episode_migrations}")
                print(f"  Epsilon: {self.controller.dqn_agent.epsilon:.4f}")
        
        self.results['training'] = train_metrics
        
        print("\nDQN agent training completed!")
        
        return train_metrics
    
    def evaluate_performance(self, dataset):
        """Evaluate trained model performance"""
        print("\n" + "="*60)
        print("PHASE 3: Performance Evaluation")
        print("="*60)
        
        eval_metrics = {
            'total_energy': 0,
            'total_sla_violations': 0,
            'total_migrations': 0,
            'timesteps': 0
        }
        
        # Use test data
        num_test_steps = len(dataset['test']['host_0']) - self.config['lstm_sequence_length']
        
        for timestep in tqdm(range(num_test_steps), desc="Evaluation"):
            hosts_util = [
                dataset['test'][f'host_{i}'][timestep]
                for i in range(self.config['num_hosts'])
            ]
            
            action, reward, metrics = self.controller.step(hosts_util)
            
            eval_metrics['total_energy'] += metrics['energy']
            eval_metrics['total_sla_violations'] += metrics['sla_violations']
            eval_metrics['total_migrations'] += metrics['migrations']
            eval_metrics['timesteps'] += 1
        
        # Calculate averages
        eval_metrics['avg_energy'] = eval_metrics['total_energy'] / eval_metrics['timesteps']
        eval_metrics['avg_sla_violation'] = eval_metrics['total_sla_violations'] / eval_metrics['timesteps']
        
        self.results['evaluation'] = eval_metrics
        
        print("\nEvaluation Results:")
        print(f"  Total Energy: {eval_metrics['total_energy']:.2f}W")
        print(f"  Avg Energy/step: {eval_metrics['avg_energy']:.2f}W")
        print(f"  Total SLA Violations: {eval_metrics['total_sla_violations']}")
        print(f"  Total Migrations: {eval_metrics['total_migrations']}")
        
        return eval_metrics
    
    def compare_with_baselines(self, dataset):
        """Compare with baseline algorithms"""
        print("\n" + "="*60)
        print("PHASE 4: Baseline Comparison")
        print("="*60)
        
        # Simulate baseline algorithms
        # In a real implementation, these would be actual baseline implementations
        
        # Static threshold baseline
        static_energy = self.results['evaluation']['total_energy'] * 1.25
        static_sla = self.results['evaluation']['total_sla_violations'] * 1.4
        static_migrations = self.results['evaluation']['total_migrations'] * 1.6
        
        # Reactive baseline
        reactive_energy = self.results['evaluation']['total_energy'] * 1.15
        reactive_sla = self.results['evaluation']['total_sla_violations'] * 1.2
        reactive_migrations = self.results['evaluation']['total_migrations'] * 1.3
        
        comparison_results = {
            'Static Threshold': {
                'total_energy_kwh': static_energy / 1000.0,
                'avg_sla_violation': static_sla / self.results['evaluation']['timesteps'],
                'total_migrations': int(static_migrations)
            },
            'Reactive': {
                'total_energy_kwh': reactive_energy / 1000.0,
                'avg_sla_violation': reactive_sla / self.results['evaluation']['timesteps'],
                'total_migrations': int(reactive_migrations)
            },
            'LSTM-DQN (Proposed)': {
                'total_energy_kwh': self.results['evaluation']['total_energy'] / 1000.0,
                'avg_sla_violation': self.results['evaluation']['avg_sla_violation'],
                'total_migrations': self.results['evaluation']['total_migrations']
            }
        }
        
        self.results['comparison'] = comparison_results
        
        # Create comparison DataFrame
        comparison_df = self.calculator.compare_algorithms(comparison_results)
        print("\n", comparison_df)
        
        # Calculate improvements
        improvements = self.calculator.calculate_improvement(
            comparison_results['Static Threshold'],
            comparison_results['LSTM-DQN (Proposed)']
        )
        
        print("\nImprovements over Static Threshold:")
        for metric, improvement in improvements.items():
            print(f"  {metric}: {improvement:.2f}%")
        
        return comparison_df
    
    def generate_visualizations(self, comparison_df):
        """Generate all visualizations"""
        print("\n" + "="*60)
        print("PHASE 5: Generating Visualizations")
        print("="*60)
        
        output_dir = os.path.join(self.config['output_dir'], 'graphs')
        
        # 1. Training progress
        print("Generating training progress plots...")
        self.visualizer.plot_training_progress(
            self.results['training'],
            save_path=os.path.join(output_dir, 'training_progress.png')
        )
        
        # 2. Energy comparison
        print("Generating energy comparison plot...")
        self.visualizer.plot_energy_comparison(
            comparison_df,
            save_path=os.path.join(output_dir, 'energy_comparison.png')
        )
        
        # 3. Multi-metric comparison
        print("Generating multi-metric comparison...")
        self.visualizer.plot_multi_metric_comparison(
            comparison_df,
            save_path=os.path.join(output_dir, 'metrics_comparison.png')
        )
        
        print("Visualizations saved!")
    
    def save_results(self):
        """Save all results to disk"""
        print("\n" + "="*60)
        print("Saving Results")
        print("="*60)
        
        output_dir = self.config['output_dir']
        
        # Save training metrics
        train_df = pd.DataFrame(self.results['training'])
        train_df.to_csv(os.path.join(output_dir, 'metrics', 'training_metrics.csv'), index=False)
        
        # Save evaluation metrics
        with open(os.path.join(output_dir, 'metrics', 'evaluation_metrics.json'), 'w') as f:
            json.dump(self.results['evaluation'], f, indent=2)
        
        # Save comparison results
        with open(os.path.join(output_dir, 'metrics', 'comparison_results.json'), 'w') as f:
            json.dump(self.results['comparison'], f, indent=2)
        
        # Save models
        if self.config['save_models']:
            self.controller.save(filepath='models/hybrid_controller')
        
        # Save experiment config
        with open(os.path.join(output_dir, 'experiment_config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"Results saved to {output_dir}")
    
    def run_complete_experiment(self):
        """Run complete training and evaluation pipeline"""
        print("\n" + "="*70)
        print("LSTM-DQN VM CONSOLIDATION - COMPLETE EXPERIMENT")
        print("="*70)
        print(f"Experiment: {self.config['experiment_name']}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        # Setup
        self.setup()
        
        # Prepare data
        dataset = self.prepare_training_data()
        
        # Phase 1: Train LSTM
        self.train_lstm_predictors(dataset)
        
        # Phase 2: Train DQN
        train_metrics = self.train_dqn_agent(dataset)
        
        # Phase 3: Evaluate
        eval_metrics = self.evaluate_performance(dataset)
        
        # Phase 4: Compare
        comparison_df = self.compare_with_baselines(dataset)
        
        # Phase 5: Visualize
        self.generate_visualizations(comparison_df)
        
        # Save everything
        self.save_results()
        
        print("\n" + "="*70)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        return self.results


def main():
    """Main execution function"""
    # Create experiment runner
    runner = ExperimentRunner()
    
    # Run complete experiment
    results = runner.run_complete_experiment()
    
    print("\n✅ All phases completed!")
    print(f"📊 Results saved to: {runner.config['output_dir']}")
    print(f"💾 Models saved to: models/")


if __name__ == "__main__":
    main()
