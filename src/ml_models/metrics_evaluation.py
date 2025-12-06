"""
Evaluation Metrics and Performance Analysis
Calculates energy consumption, SLA violations, and migration costs
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json


class MetricsCalculator:
    """
    Calculate performance metrics for VM consolidation
    """
    
    def __init__(self, energy_model=None):
        """
        Initialize metrics calculator
        
        Args:
            energy_model: Dictionary with power consumption parameters
        """
        self.energy_model = energy_model or {
            'idle_power': 70,      # Watts at 0% utilization
            'max_power': 250,      # Watts at 100% utilization
            'sleep_power': 10,     # Watts in sleep mode
            'migration_cost': 5    # Wh per migration
        }
        
        self.metrics_history = []
    
    def calculate_power_consumption(self, utilization):
        """
        Calculate power consumption using linear model
        
        P(u) = P_idle + (P_max - P_idle) * u
        
        Args:
            utilization: CPU utilization (0-1 or 0-100)
            
        Returns:
            power: Power consumption in Watts
        """
        # Normalize to 0-1 range
        if utilization > 1:
            utilization = utilization / 100.0
        
        if utilization == 0:
            return self.energy_model['sleep_power']
        
        idle_power = self.energy_model['idle_power']
        max_power = self.energy_model['max_power']
        
        power = idle_power + (max_power - idle_power) * utilization
        
        return power
    
    def calculate_total_energy(self, host_utilizations, time_interval=1.0):
        """
        Calculate total energy consumption for all hosts
        
        Args:
            host_utilizations: Array of host utilizations
            time_interval: Time interval in hours (default 1 hour)
            
        Returns:
            energy: Total energy in kWh
        """
        total_power = sum(self.calculate_power_consumption(u) for u in host_utilizations)
        
        # Convert to kWh: (Watts * hours) / 1000
        energy_kwh = (total_power * time_interval) / 1000.0
        
        return energy_kwh
    
    def calculate_pue(self, it_energy, total_facility_energy):
        """
        Calculate Power Usage Effectiveness (PUE)
        
        PUE = Total Facility Energy / IT Equipment Energy
        
        Args:
            it_energy: Energy consumed by IT equipment
            total_facility_energy: Total facility energy (IT + cooling + overhead)
            
        Returns:
            pue: Power Usage Effectiveness
        """
        if it_energy == 0:
            return 0
        
        pue = total_facility_energy / it_energy
        return pue
    
    def calculate_sla_violations(self, host_utilizations, threshold=80):
        """
        Calculate SLA violations
        
        SLAV consists of two components:
        1. SLATAH: SLA violation time per active host
        2. PDM: Performance degradation due to migrations
        
        Args:
            host_utilizations: List of utilization values
            threshold: Overload threshold (default 80%)
            
        Returns:
            sla_metrics: Dictionary with SLAV components
        """
        # Count hosts above threshold
        overloaded_hosts = sum(1 for u in host_utilizations if u > threshold)
        active_hosts = sum(1 for u in host_utilizations if u > 0)
        
        # SLATAH: Percentage of time hosts are overloaded
        slatah = overloaded_hosts / active_hosts if active_hosts > 0 else 0
        
        return {
            'slatah': slatah,
            'overloaded_hosts': overloaded_hosts,
            'active_hosts': active_hosts
        }
    
    def calculate_migration_cost(self, num_migrations):
        """
        Calculate energy cost of migrations
        
        Args:
            num_migrations: Number of VM migrations
            
        Returns:
            cost: Energy cost in kWh
        """
        cost_per_migration = self.energy_model['migration_cost']
        total_cost = num_migrations * cost_per_migration / 1000.0  # Convert to kWh
        
        return total_cost
    
    def calculate_combined_slav(self, slatah, pdm):
        """
        Calculate combined SLA violation metric
        
        SLAV = SLATAH × PDM
        
        Args:
            slatah: SLA violation time per active host
            pdm: Performance degradation due to migration
            
        Returns:
            slav: Combined SLA violation
        """
        return slatah * pdm
    
    def calculate_resource_utilization(self, host_utilizations):
        """
        Calculate overall resource utilization statistics
        
        Args:
            host_utilizations: Array of host utilizations
            
        Returns:
            stats: Dictionary with utilization statistics
        """
        active_hosts = [u for u in host_utilizations if u > 0]
        
        if not active_hosts:
            return {
                'mean_utilization': 0,
                'std_utilization': 0,
                'active_hosts': 0,
                'idle_hosts': len(host_utilizations)
            }
        
        return {
            'mean_utilization': np.mean(active_hosts),
            'std_utilization': np.std(active_hosts),
            'min_utilization': np.min(active_hosts),
            'max_utilization': np.max(active_hosts),
            'active_hosts': len(active_hosts),
            'idle_hosts': len(host_utilizations) - len(active_hosts),
            'utilization_balance': np.std(active_hosts)  # Lower is better
        }
    
    def evaluate_episode(self, episode_data):
        """
        Evaluate complete episode
        
        Args:
            episode_data: Dictionary with episode information
                - host_utilizations: List of utilization arrays over time
                - num_migrations: Total migrations
                - time_steps: Number of time steps
            
        Returns:
            metrics: Complete metrics dictionary
        """
        total_energy = 0
        total_sla_violations = 0
        timesteps = len(episode_data['host_utilizations'])
        
        for host_utils in episode_data['host_utilizations']:
            energy = self.calculate_total_energy(host_utils, time_interval=1.0/timesteps)
            total_energy += energy
            
            sla = self.calculate_sla_violations(host_utils)
            total_sla_violations += sla['slatah']
        
        # Migration cost
        migration_energy = self.calculate_migration_cost(episode_data['num_migrations'])
        
        # Total energy includes migration overhead
        total_energy += migration_energy
        
        # Average metrics
        avg_sla_violation = total_sla_violations / timesteps if timesteps > 0 else 0
        
        metrics = {
            'total_energy_kwh': total_energy,
            'avg_energy_per_step': total_energy / timesteps if timesteps > 0 else 0,
            'total_migrations': episode_data['num_migrations'],
            'migration_energy_kwh': migration_energy,
            'avg_sla_violation': avg_sla_violation,
            'timesteps': timesteps
        }
        
        self.metrics_history.append(metrics)
        
        return metrics
    
    def compare_algorithms(self, results_dict):
        """
        Compare multiple algorithms
        
        Args:
            results_dict: Dictionary mapping algorithm names to their metrics
            
        Returns:
            comparison_df: DataFrame with comparison results
        """
        comparison_data = []
        
        for algo_name, metrics in results_dict.items():
            comparison_data.append({
                'Algorithm': algo_name,
                'Energy (kWh)': metrics['total_energy_kwh'],
                'SLA Violations (%)': metrics['avg_sla_violation'] * 100,
                'Migrations': metrics['total_migrations'],
                'Migration Energy (kWh)': metrics.get('migration_energy_kwh', 0)
            })
        
        df = pd.DataFrame(comparison_data)
        
        return df
    
    def calculate_improvement(self, baseline_metrics, proposed_metrics):
        """
        Calculate improvement over baseline
        
        Args:
            baseline_metrics: Metrics from baseline algorithm
            proposed_metrics: Metrics from proposed algorithm
            
        Returns:
            improvements: Dictionary with improvement percentages
        """
        improvements = {}
        
        for key in ['total_energy_kwh', 'avg_sla_violation', 'total_migrations']:
            if key in baseline_metrics and key in proposed_metrics:
                baseline_val = baseline_metrics[key]
                proposed_val = proposed_metrics[key]
                
                if baseline_val > 0:
                    improvement = ((baseline_val - proposed_val) / baseline_val) * 100
                    improvements[key] = improvement
                else:
                    improvements[key] = 0
        
        return improvements
    
    def save_metrics(self, filepath='results/metrics/metrics.json'):
        """Save metrics to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        print(f"Metrics saved to {filepath}")
    
    def load_metrics(self, filepath='results/metrics/metrics.json'):
        """Load metrics from JSON file"""
        with open(filepath, 'r') as f:
            self.metrics_history = json.load(f)
        
        print(f"Metrics loaded from {filepath}")


class PerformanceVisualizer:
    """
    Visualize performance metrics and comparisons
    """
    
    def __init__(self, style='seaborn-v0_8-darkgrid'):
        """Initialize visualizer"""
        plt.style.use('default')  # Use default style as fallback
        sns.set_palette("husl")
        self.fig_count = 0
    
    def plot_energy_comparison(self, comparison_df, save_path=None):
        """
        Plot energy consumption comparison
        
        Args:
            comparison_df: DataFrame from compare_algorithms
            save_path: Path to save figure
        """
        plt.figure(figsize=(10, 6))
        
        x = np.arange(len(comparison_df))
        width = 0.6
        
        bars = plt.bar(x, comparison_df['Energy (kWh)'], width, alpha=0.8)
        
        # Color code bars
        colors = ['#2ecc71' if i == comparison_df['Energy (kWh)'].idxmin() 
                  else '#3498db' for i in range(len(bars))]
        
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.xlabel('Algorithm', fontsize=12)
        plt.ylabel('Total Energy (kWh)', fontsize=12)
        plt.title('Energy Consumption Comparison', fontsize=14, fontweight='bold')
        plt.xticks(x, comparison_df['Algorithm'], rotation=15, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_multi_metric_comparison(self, comparison_df, save_path=None):
        """
        Plot multiple metrics side by side
        
        Args:
            comparison_df: DataFrame with multiple metrics
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics = ['Energy (kWh)', 'SLA Violations (%)', 'Migrations']
        colors = ['#3498db', '#e74c3c', '#f39c12']
        
        for ax, metric, color in zip(axes, metrics, colors):
            x = np.arange(len(comparison_df))
            bars = ax.bar(x, comparison_df[metric], alpha=0.8, color=color)
            
            ax.set_xlabel('Algorithm', fontsize=11)
            ax.set_ylabel(metric, fontsize=11)
            ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(comparison_df['Algorithm'], rotation=15, ha='right')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_training_progress(self, metrics_history, save_path=None):
        """
        Plot training progress over episodes
        
        Args:
            metrics_history: List of metrics dictionaries
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        episodes = range(1, len(metrics_history) + 1)
        
        # Energy over time
        energy = [m['total_energy_kwh'] for m in metrics_history]
        axes[0, 0].plot(episodes, energy, linewidth=2, marker='o', markersize=4)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Energy (kWh)')
        axes[0, 0].set_title('Energy Consumption Over Time')
        axes[0, 0].grid(alpha=0.3)
        
        # SLA violations over time
        sla = [m['avg_sla_violation'] * 100 for m in metrics_history]
        axes[0, 1].plot(episodes, sla, linewidth=2, marker='s', markersize=4, color='red')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('SLA Violation (%)')
        axes[0, 1].set_title('SLA Violations Over Time')
        axes[0, 1].grid(alpha=0.3)
        
        # Migrations over time
        migrations = [m['total_migrations'] for m in metrics_history]
        axes[1, 0].plot(episodes, migrations, linewidth=2, marker='^', markersize=4, color='orange')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Number of Migrations')
        axes[1, 0].set_title('VM Migrations Over Time')
        axes[1, 0].grid(alpha=0.3)
        
        # Combined metric (weighted sum)
        combined = [0.5 * m['total_energy_kwh'] + 0.3 * m['avg_sla_violation'] * 100 + 0.2 * m['total_migrations']
                   for m in metrics_history]
        axes[1, 1].plot(episodes, combined, linewidth=2, marker='D', markersize=4, color='green')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Combined Score')
        axes[1, 1].set_title('Combined Performance Metric')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


if __name__ == "__main__":
    print("Testing Metrics Calculator...\n")
    
    # Initialize calculator
    calculator = MetricsCalculator()
    
    # Test energy calculation
    host_utils = [30, 45, 60, 0, 80]
    energy = calculator.calculate_total_energy(host_utils)
    print(f"Total energy for utilizations {host_utils}: {energy:.4f} kWh")
    
    # Test SLA violations
    sla_metrics = calculator.calculate_sla_violations(host_utils, threshold=70)
    print(f"\nSLA Metrics: {sla_metrics}")
    
    # Simulate algorithm comparison
    print("\n" + "="*50)
    print("Algorithm Comparison")
    print("="*50)
    
    results = {
        'Static Threshold': {
            'total_energy_kwh': 5.2,
            'avg_sla_violation': 0.15,
            'total_migrations': 45
        },
        'Reactive': {
            'total_energy_kwh': 4.8,
            'avg_sla_violation': 0.12,
            'total_migrations': 38
        },
        'LSTM-DQN (Proposed)': {
            'total_energy_kwh': 4.1,
            'avg_sla_violation': 0.08,
            'total_migrations': 25
        }
    }
    
    comparison_df = calculator.compare_algorithms(results)
    print("\n", comparison_df)
    
    # Calculate improvements
    improvements = calculator.calculate_improvement(
        results['Static Threshold'],
        results['LSTM-DQN (Proposed)']
    )
    
    print("\nImprovements over Static Threshold:")
    for metric, improvement in improvements.items():
        print(f"  {metric}: {improvement:.2f}%")
    
    # Visualize
    print("\nGenerating visualizations...")
    visualizer = PerformanceVisualizer()
    visualizer.plot_multi_metric_comparison(comparison_df)
    
    print("\nMetrics evaluation completed!")
