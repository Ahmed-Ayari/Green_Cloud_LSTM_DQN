"""
Data Preprocessing Pipeline for PlanetLab and Google Cluster Traces
Loads, cleans, and prepares workload traces for training and evaluation
"""

import numpy as np
import pandas as pd
import os
import glob
from datetime import datetime
import matplotlib.pyplot as plt


class WorkloadDataLoader:
    """
    Load and preprocess cloud workload traces
    """
    
    def __init__(self, data_dir='data'):
        """
        Initialize data loader
        
        Args:
            data_dir: Root directory containing workload data
        """
        self.data_dir = data_dir
        self.planetlab_dir = os.path.join(data_dir, 'planetlab')
        self.google_dir = os.path.join(data_dir, 'google_cluster')
    
    def load_planetlab_trace(self, filename=None, normalize=True):
        """
        Load PlanetLab workload trace
        
        PlanetLab format: Each line contains CPU utilization value (simple format)
        or timestamp and CPU utilization (extended format)
        
        Args:
            filename: Specific trace file (if None, loads first available)
            normalize: Whether to normalize utilization to 0-100 range
            
        Returns:
            df: DataFrame with timestamp and utilization columns
        """
        if filename is None:
            # Find first file in planetlab directory
            files = glob.glob(os.path.join(self.planetlab_dir, '*'))
            if not files:
                raise FileNotFoundError(f"No PlanetLab traces found in {self.planetlab_dir}")
            filename = files[0]
        
        # Use the filename directly if it's an absolute path
        if os.path.isabs(filename):
            filepath = filename
        else:
            filepath = os.path.join(self.planetlab_dir, filename)
        
        # Load data
        data = []
        with open(filepath, 'r') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                try:
                    if len(parts) >= 2:
                        # Extended format: timestamp utilization
                        timestamp = int(parts[0])
                        utilization = float(parts[1])
                    else:
                        # Simple format: just utilization value
                        timestamp = idx * 300  # 5-minute intervals
                        utilization = float(parts[0])
                    data.append([timestamp, utilization])
                except ValueError:
                    continue
        
        df = pd.DataFrame(data, columns=['timestamp', 'cpu_utilization'])
        
        # Normalize if requested
        if normalize:
            # Check if values are already in percentage (0-100)
            max_util = df['cpu_utilization'].max()
            if max_util > 100:
                df['cpu_utilization'] = (df['cpu_utilization'] / max_util) * 100
            
            # Clip to 0-100 range
            df['cpu_utilization'] = df['cpu_utilization'].clip(0, 100)
        
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        return df
    
    def load_planetlab_from_directory(self, date_folder=None, max_traces=100):
        """
        Load PlanetLab traces from the standard directory structure
        (e.g., planetlab/20110303/...)
        
        Args:
            date_folder: Specific date folder (e.g., '20110303'), if None uses first available
            max_traces: Maximum number of traces to load
            
        Returns:
            vm_data: numpy array (timesteps, num_vms)
        """
        # Find date folders
        if date_folder:
            date_path = os.path.join(self.planetlab_dir, date_folder)
        else:
            date_folders = sorted([d for d in os.listdir(self.planetlab_dir) 
                                   if os.path.isdir(os.path.join(self.planetlab_dir, d))])
            if not date_folders:
                raise FileNotFoundError(f"No date folders found in {self.planetlab_dir}")
            date_path = os.path.join(self.planetlab_dir, date_folders[0])
            print(f"Using PlanetLab traces from: {date_folders[0]}")
        
        # Get all trace files in the folder
        trace_files = sorted([f for f in os.listdir(date_path) 
                             if os.path.isfile(os.path.join(date_path, f))])[:max_traces]
        
        print(f"Found {len(trace_files)} trace files")
        
        vm_data = []
        for trace_file in trace_files:
            # Use the full absolute path directly
            filepath = os.path.join(date_path, trace_file)
            try:
                # Read the file directly instead of using load_planetlab_trace
                data = []
                with open(filepath, 'r') as f:
                    for idx, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        try:
                            if len(parts) >= 2:
                                utilization = float(parts[1])
                            else:
                                utilization = float(parts[0])
                            data.append(utilization)
                        except ValueError:
                            continue
                
                if data:
                    # Normalize to 0-100 range
                    data = np.array(data)
                    max_val = data.max()
                    if max_val > 100:
                        data = (data / max_val) * 100
                    data = np.clip(data, 0, 100)
                    vm_data.append(data)
                    
            except Exception as e:
                print(f"Warning: Could not load {trace_file}: {e}")
                continue
        
        if not vm_data:
            raise ValueError("No traces could be loaded successfully")
        
        # Find minimum length and truncate all traces
        min_length = min(len(trace) for trace in vm_data)
        vm_data = np.array([trace[:min_length] for trace in vm_data]).T
        
        print(f"Loaded {vm_data.shape[1]} VMs with {vm_data.shape[0]} timesteps each")
        
        return vm_data
    
    def load_all_planetlab_traces(self, max_traces=None):
        """
        Load all PlanetLab traces
        
        Args:
            max_traces: Maximum number of traces to load
            
        Returns:
            traces: Dictionary mapping VM ID to utilization DataFrame
        """
        # Check if we have date-based folder structure
        subdirs = [d for d in os.listdir(self.planetlab_dir) 
                   if os.path.isdir(os.path.join(self.planetlab_dir, d))]
        
        if subdirs:
            # Use new folder structure loader
            vm_data = self.load_planetlab_from_directory(max_traces=max_traces or 100)
            traces = {}
            for i in range(vm_data.shape[1]):
                traces[f'vm_{i}'] = pd.DataFrame({
                    'timestamp': range(len(vm_data)),
                    'cpu_utilization': vm_data[:, i]
                })
            return traces
        
        # Fallback to old .txt file structure
        files = glob.glob(os.path.join(self.planetlab_dir, '*.txt'))
        
        if max_traces:
            files = files[:max_traces]
        
        traces = {}
        for i, filepath in enumerate(files):
            try:
                df = self.load_planetlab_trace(filepath)
                vm_id = f'vm_{i}'
                traces[vm_id] = df
                print(f"Loaded {vm_id}: {len(df)} timesteps")
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
        
        return traces
    
    def load_google_cluster_trace(self, filename=None):
        """
        Load Google Cluster trace
        
        Google Cluster format: CSV with multiple columns including CPU/RAM usage
        
        Args:
            filename: Specific trace file
            
        Returns:
            df: DataFrame with resource utilization data
        """
        if filename is None:
            files = glob.glob(os.path.join(self.google_dir, '*.csv'))
            if not files:
                raise FileNotFoundError(f"No Google traces found in {self.google_dir}")
            filename = files[0]
        
        filepath = os.path.join(self.google_dir, filename) if not os.path.isabs(filename) else filename
        
        # Load CSV
        df = pd.read_csv(filepath)
        
        # Google Cluster typically has: timestamp, cpu_usage, memory_usage, etc.
        # Normalize to percentage
        if 'cpu_usage' in df.columns:
            df['cpu_utilization'] = df['cpu_usage'] * 100
        
        if 'memory_usage' in df.columns:
            df['ram_utilization'] = df['memory_usage'] * 100
        
        return df
    
    def aggregate_to_host_level(self, vm_traces, num_hosts=10):
        """
        Aggregate multiple VM traces to host-level utilization
        
        Args:
            vm_traces: Dictionary of VM traces
            num_hosts: Number of physical hosts to simulate
            
        Returns:
            host_traces: Dictionary of host-level utilization
        """
        # Assign VMs to hosts (simple round-robin)
        host_assignment = {}
        vm_ids = list(vm_traces.keys())
        
        for i, vm_id in enumerate(vm_ids):
            host_id = i % num_hosts
            if host_id not in host_assignment:
                host_assignment[host_id] = []
            host_assignment[host_id].append(vm_id)
        
        # Aggregate utilization per host
        host_traces = {}
        
        for host_id, assigned_vms in host_assignment.items():
            # Find common timestamps
            min_length = min(len(vm_traces[vm_id]) for vm_id in assigned_vms)
            
            # Sum utilizations (with capacity limit of 100%)
            combined_util = np.zeros(min_length)
            
            for vm_id in assigned_vms:
                vm_util = vm_traces[vm_id]['cpu_utilization'].values[:min_length]
                combined_util += vm_util
            
            # Normalize to host capacity (assume each VM contributes proportionally)
            host_util = combined_util / len(assigned_vms)
            host_util = np.clip(host_util, 0, 100)
            
            host_traces[f'host_{host_id}'] = host_util
        
        return host_traces
    
    def create_train_test_split(self, data, train_ratio=0.8):
        """
        Split data into training and testing sets
        
        Args:
            data: Time series data
            train_ratio: Fraction of data for training
            
        Returns:
            train_data, test_data
        """
        split_idx = int(len(data) * train_ratio)
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        
        return train_data, test_data
    
    def generate_synthetic_workload(self, num_timesteps=1000, pattern='mixed'):
        """
        Generate synthetic workload for testing
        
        Args:
            num_timesteps: Number of time steps
            pattern: 'sine', 'random', 'spike', 'mixed'
            
        Returns:
            utilization: Array of utilization values
        """
        t = np.linspace(0, 10, num_timesteps)
        
        if pattern == 'sine':
            # Sinusoidal pattern
            utilization = 50 + 30 * np.sin(t)
        
        elif pattern == 'random':
            # Random walk
            utilization = 50 + np.cumsum(np.random.randn(num_timesteps) * 2)
            utilization = np.clip(utilization, 10, 90)
        
        elif pattern == 'spike':
            # Periodic spikes
            base = 30 + 10 * np.sin(t)
            spikes = np.zeros(num_timesteps)
            spike_indices = np.random.choice(num_timesteps, size=10, replace=False)
            spikes[spike_indices] = 40
            utilization = base + spikes
        
        elif pattern == 'mixed':
            # Complex pattern
            trend = 40 + 20 * np.sin(t)
            seasonal = 15 * np.sin(t * 5)
            noise = np.random.normal(0, 5, num_timesteps)
            utilization = trend + seasonal + noise
        
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
        
        # Ensure valid range
        utilization = np.clip(utilization, 0, 100)
        
        return utilization
    
    def visualize_trace(self, trace_data, title='Workload Trace', save_path=None):
        """
        Visualize workload trace
        
        Args:
            trace_data: Time series data
            title: Plot title
            save_path: Path to save figure
        """
        plt.figure(figsize=(12, 4))
        plt.plot(trace_data, linewidth=0.8)
        plt.xlabel('Time Step')
        plt.ylabel('CPU Utilization (%)')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_statistics(self, trace_data):
        """
        Calculate statistics for workload trace
        
        Args:
            trace_data: Time series data
            
        Returns:
            stats: Dictionary of statistics
        """
        stats = {
            'mean': np.mean(trace_data),
            'std': np.std(trace_data),
            'min': np.min(trace_data),
            'max': np.max(trace_data),
            'median': np.median(trace_data),
            'p25': np.percentile(trace_data, 25),
            'p75': np.percentile(trace_data, 75),
            'p95': np.percentile(trace_data, 95),
            'length': len(trace_data)
        }
        
        return stats


def prepare_dataset_for_training(num_hosts=10, timesteps=1000, pattern='mixed', use_real_data=True, data_dir='../data'):
    """
    Prepare complete dataset for LSTM-DQN training
    
    Args:
        num_hosts: Number of hosts
        timesteps: Number of timesteps (used for synthetic data)
        pattern: Workload pattern (used for synthetic data)
        use_real_data: If True, try to load PlanetLab data first
        data_dir: Directory containing real data
        
    Returns:
        dataset: Dictionary with train and test data
    """
    loader = WorkloadDataLoader(data_dir=data_dir)
    
    dataset = {
        'train': {},
        'test': {},
        'metadata': {
            'num_hosts': num_hosts,
            'timesteps': timesteps,
            'pattern': pattern,
            'data_source': 'synthetic'
        }
    }
    
    # Try to load real PlanetLab data
    if use_real_data:
        try:
            print("Attempting to load PlanetLab data...")
            vm_data = loader.load_planetlab_from_directory(max_traces=num_hosts * 5)
            
            # Aggregate VMs to hosts
            print(f"Aggregating {vm_data.shape[1]} VMs into {num_hosts} hosts...")
            
            # Simple aggregation: average VMs per host
            vms_per_host = vm_data.shape[1] // num_hosts
            host_data = []
            
            for host_id in range(num_hosts):
                start_vm = host_id * vms_per_host
                end_vm = start_vm + vms_per_host if host_id < num_hosts - 1 else vm_data.shape[1]
                
                # Average utilization of assigned VMs
                host_util = np.mean(vm_data[:, start_vm:end_vm], axis=1)
                host_util = np.clip(host_util, 0, 100)
                host_data.append(host_util)
            
            # Convert to numpy array
            host_data = np.array(host_data).T  # Shape: (timesteps, num_hosts)
            
            # Split into train/test
            train_size = int(0.8 * len(host_data))
            
            for host_id in range(num_hosts):
                dataset['train'][f'host_{host_id}'] = host_data[:train_size, host_id]
                dataset['test'][f'host_{host_id}'] = host_data[train_size:, host_id]
            
            dataset['metadata']['data_source'] = 'planetlab'
            dataset['metadata']['timesteps'] = len(host_data)
            
            print(f"✅ Successfully loaded PlanetLab data!")
            print(f"   Timesteps: {len(host_data)}, Hosts: {num_hosts}")
            
            return dataset
            
        except Exception as e:
            print(f"⚠️ Could not load PlanetLab data: {e}")
            print("   Falling back to synthetic data...")
    
    # Generate synthetic data for each host
    print("Generating synthetic workload data...")
    
    for host_id in range(num_hosts):
        # Generate workload with slight variation per host
        workload = loader.generate_synthetic_workload(timesteps, pattern)
        
        # Add some host-specific variation
        variation = np.random.uniform(-5, 5, timesteps)
        workload = np.clip(workload + variation, 0, 100)
        
        # Split into train/test
        train_data, test_data = loader.create_train_test_split(workload, train_ratio=0.8)
        
        dataset['train'][f'host_{host_id}'] = train_data
        dataset['test'][f'host_{host_id}'] = test_data
    
    print(f"✅ Generated synthetic data: {timesteps} timesteps, {num_hosts} hosts")
    
    return dataset


if __name__ == "__main__":
    print("Testing Data Preprocessing Pipeline...\n")
    
    # Initialize loader
    loader = WorkloadDataLoader()
    
    # Generate synthetic workload
    print("Generating synthetic workload...")
    workload = loader.generate_synthetic_workload(num_timesteps=500, pattern='mixed')
    
    # Statistics
    stats = loader.get_statistics(workload)
    print("\nWorkload Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}")
    
    # Visualize
    print("\nVisualizing workload...")
    loader.visualize_trace(workload, title='Synthetic Workload - Mixed Pattern')
    
    # Prepare full dataset
    print("\nPreparing training dataset...")
    dataset = prepare_dataset_for_training(num_hosts=5, timesteps=1000)
    
    print(f"\nDataset prepared:")
    print(f"  Hosts: {dataset['metadata']['num_hosts']}")
    print(f"  Training samples per host: {len(dataset['train']['host_0'])}")
    print(f"  Testing samples per host: {len(dataset['test']['host_0'])}")
    
    print("\nData preprocessing completed!")
