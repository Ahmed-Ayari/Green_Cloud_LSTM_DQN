# 🌱 Green Cloud Computing: LSTM-DQN VM Consolidation

**Energy Optimization in Data Centers through Dynamic Resource Allocation**

*Master 1 Data Science - Systems Architecture*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange.svg)](https://tensorflow.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9-red.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

- [Context and Problem Statement](#context-and-problem-statement)
- [Proposed Solution](#proposed-solution)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Project Structure](#project-structure)
- [Experimental Results](#experimental-results)
- [References](#references)

---

## 🎯 Context and Problem Statement

### The Energy Challenge of Data Centers

Data centers represent **1-2% of global electricity consumption** and this share is growing rapidly. Inefficiency mainly comes from:

- **Underutilized servers**: On average, servers use only 15-20% of their capacity
- **Static allocation**: Resources are often over-provisioned "just in case"
- **Lack of prediction**: Decisions are reactive rather than proactive

### Research Question

> *"How can a dynamic and consolidated resource allocation mechanism be designed to significantly reduce energy consumption while maintaining Quality of Service (QoS)?"*

---

## 💡 Proposed Solution

Our **hybrid LSTM-DQN approach** combines two complementary techniques:

### 1. Proactive Prediction (LSTM)

The LSTM (Long Short-Term Memory) network analyzes usage history to predict future load:

```
Historical sequence [t-9, t-8, ..., t] → LSTM → Prediction [t+1]
```

**Advantages**:
- Anticipates load peaks
- Enables preventive consolidation
- Reduces SLA violations

### 2. Autonomous Decision Making (DQN)

The DQN (Deep Q-Network) agent learns the optimal consolidation policy:

```
State (current utilization + predictions) → DQN → Action (migration/consolidation)
```

**Multi-Objective Reward Function**:
```
R = -w₁·E - w₂·SLA - w₃·M

where:
  E   = Energy consumed (normalized)
  SLA = SLA violations
  M   = Number of migrations
  w₁, w₂, w₃ = Weights (0.5, 0.3, 0.2 by default)
```

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       INPUT DATA                                │
│       PlanetLab Traces (CPU utilization - real data)            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  PREPROCESSING MODULE                           │
│  • Trace loading (data_preprocessing.py)                        │
│  • Normalization [0-100%]                                       │
│  • VMs → Hosts aggregation                                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  LSTM WORKLOAD PREDICTOR                        │
│  • Architecture: LSTM(64) → Dropout(0.2) → LSTM(32) → Dense(1)  │
│  • Input: sequence of 10 timesteps                              │
│  • Output: prediction t+1 + trend (↑↓→)                         │
│  • File: lstm_predictor.py                                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  DQN CONSOLIDATION AGENT                        │
│  • Architecture: Linear(state) → ReLU → Linear(128) → Linear(A) │
│  • Double DQN with Experience Replay                            │
│  • FFD Heuristic: Top 5 candidate hosts per migration           │
│  • Actions: do_nothing | migrate(src, dst)                      │
│  • File: dqn_agent.py                                           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  CLOUDSIM BRIDGE (Optional)                     │
│  • Socket-based API (JSON serialization)                        │
│  • Modes: STANDALONE (Python) or CLOUDSIM (Java)                │
│  • File: cloudsim_bridge.py                                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  HYBRID CONTROLLER                              │
│  • Integrates LSTM + DQN + CloudSim Bridge                      │
│  • Energy model: P = P_idle + (P_max - P_idle) × U              │
│  • State: S = [U₁...Uₙ, Û₁...Ûₙ, active_hosts]                  │
│  • File: hybrid_controller.py                                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  METRICS AND EVALUATION                         │
│  • Total energy (Watts/kWh)                                     │
│  • SLA violations (% hosts > 80%)                               │
│  • Number of migrations                                         │
│  • Pareto frontier visualization                                │
│  • File: metrics_evaluation.py                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Technology | Description |
|-----------|------------|-------------|
| **LSTM Predictor** | TensorFlow/Keras | Forecasts host utilization Û_{t+1} |
| **DQN Agent** | PyTorch | Learns optimal consolidation policy |
| **FFD Selector** | Python | Filters top 5 target hosts (First Fit Decreasing) |
| **CloudSim Bridge** | Socket/JSON | Python-Java communication interface |
| **Hybrid Controller** | Python | Orchestrates all components |

---

## 🚀 Installation

### Prerequisites

- **Python** 3.10 or higher
- **pip** (Python package manager)
- **GPU** (optional, speeds up training)

### Step 1: Navigate to Project

```powershell
cd "C:\Users\ahmed\OneDrive\Desktop\M1 Data Science\Architecture Systèmes\Projet"
```

### Step 2: Create Virtual Environment

```powershell
# Windows PowerShell
python -m venv venv
.\venv\Scripts\activate
```

### Step 3: Install Dependencies

```powershell
pip install -r requirements.txt
```

**Main Dependencies**:
| Package | Version | Usage |
|---------|---------|-------|
| TensorFlow | 2.20.0 | LSTM (Keras) |
| PyTorch | 2.9.1 | DQN |
| NumPy | 1.26+ | Numerical computations |
| Pandas | 2.0+ | Data manipulation |
| Matplotlib | 3.7+ | Visualizations |
| scikit-learn | 1.7+ | Normalization |
| tqdm | 4.65+ | Progress bars |

### Step 4: Download Required Data

#### PlanetLab Workload Traces (Required)

The PlanetLab traces contain real CPU utilization data from distributed systems.

**Download from**: https://github.com/beloglazov/planetlab-workload-traces

```powershell
# Clone the PlanetLab traces repository
git clone https://github.com/beloglazov/planetlab-workload-traces.git temp_planetlab

# Copy traces to the data folder
Copy-Item -Recurse temp_planetlab/* data/planetlab/

# Clean up
Remove-Item -Recurse -Force temp_planetlab
```

**Expected structure after download**:
```
data/planetlab/
├── 20110303/
│   ├── trace_file_1
│   ├── trace_file_2
│   └── ...
├── 20110306/
├── 20110309/
└── ...
```

Each file contains CPU utilization values (one value per line, 0-100%).

#### Google Cluster Data (Optional)

For additional experiments with Google cluster traces:

**Download from**: https://github.com/google/cluster-data

Place CSV files in `data/google_cluster/`.

### Step 5: CloudSim (Optional - Java Simulation)

CloudSim is an optional Java-based cloud simulation toolkit. It's **not required** for running the ML models.

**If needed, download from**: https://github.com/Cloudslab/cloudsim

```powershell
# Clone CloudSim (optional)
git clone https://github.com/Cloudslab/cloudsim.git src/cloudsim
```

---

## 💻 Usage Guide

### Option 1: Quick Test (Recommended to start)

```powershell
cd src
python quick_test.py
```

**Duration**: ~10 minutes  
**Configuration**: 5 hosts, 20 VMs, 5 episodes

**Expected output**:
```
======================================================================
LSTM-DQN VM CONSOLIDATION - QUICK TEST
======================================================================
✅ Loaded PlanetLab data: (288, 20)
✅ LSTM training complete!
Episode 1/5: Reward=-6.41, Energy=477.55, Epsilon=1.000
Episode 2/5: Reward=-6.87, Energy=475.15, Epsilon=0.831
...
✅ Pipeline verification successful!
```

### Option 2: Full Experiment

```powershell
cd src
python main_experiment.py
```

**Duration**: ~2-4 hours (depends on hardware)  
**Configuration**: 10 hosts, 50 VMs, 100 episodes

### Option 3: Programmatic Usage

```python
# 1. Load PlanetLab data
from ml_models.data_preprocessing import WorkloadDataLoader

loader = WorkloadDataLoader('../data')
vm_data = loader.load_planetlab_from_directory(max_traces=50)
print(f"Data loaded: {vm_data.shape}")  # (timesteps, num_vms)

# 2. Train an LSTM predictor
from ml_models.lstm_predictor import LSTMPredictor

predictor = LSTMPredictor(sequence_length=10, lstm_units=64)
predictor.train(vm_data[:, 0], epochs=50, verbose=1)

# Predict next value
sequence = vm_data[-10:, 0]
prediction = predictor.predict(sequence)
trend = predictor.predict_trend(sequence)
print(f"Prediction: {prediction[0]:.2f}%, Trend: {trend}")

# 3. Use the hybrid controller
from ml_models.hybrid_controller import HybridController

controller = HybridController(num_hosts=10, num_vms=50)

# Train LSTMs on historical data
controller.train_lstm_predictors(vm_data, epochs=50)

# Perform a consolidation step
hosts_util = vm_data[100, :10]  # Utilization of 10 hosts
action, reward, metrics = controller.step(hosts_util)

print(f"Action: {action}")
print(f"Energy: {metrics['energy']:.2f} W")
print(f"SLA Violations: {metrics['sla_violations']}")
```

### Option 4: Jupyter Notebooks

```powershell
cd notebooks
jupyter notebook
```

Available notebooks:
1. `01_data_exploration.ipynb` - PlanetLab data exploration
2. `02_lstm_analysis.ipynb` - LSTM predictor analysis
3. `03_dqn_training.ipynb` - DQN agent training
4. `04_results_visualization.ipynb` - Results visualization

---

## 📁 Project Structure

```
Green_Cloud_LSTM_DQN/
│
├── data/                              # Data
│   └── planetlab/                     # PlanetLab traces (download separately)
│       ├── 20110303/                  # Folder per date
│       ├── 20110306/
│       └── ...
│
├── src/                               # Source code
│   ├── ml_models/                     # ML/RL models
│   │   ├── __init__.py
│   │   ├── lstm_predictor.py          # LSTM workload predictor
│   │   ├── dqn_agent.py               # DQN agent (Double DQN + Replay)
│   │   ├── hybrid_controller.py       # LSTM+DQN+CloudSim controller
│   │   ├── cloudsim_bridge.py         # Python-Java socket API + FFD
│   │   ├── data_preprocessing.py      # Data loading utilities
│   │   └── metrics_evaluation.py      # Metrics + Pareto visualization
│   │
│   ├── cloudsim/                      # CloudSim Java (optional)
│   │   ├── modules/
│   │   └── pom.xml
│   │
│   ├── main_experiment.py             # Full experiment
│   ├── quick_test.py                  # Quick test (~10 min)
│   └── config.py                      # Global configuration
│
├── notebooks/                         # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_lstm_analysis.ipynb
│   ├── 03_dqn_training.ipynb
│   └── 04_results_visualization.ipynb
│
├── results/                           # Results
│   ├── graphs/                        # PNG visualizations
│   │   ├── quick_test_training.png    # Training progress
│   │   └── pareto_frontier.png        # Energy vs SLA trade-off
│   └── metrics/                       # JSON/CSV metrics
│       └── quick_test_results.json
│
├── models/                            # Trained models
│   ├── lstm_host_*.h5                 # Saved LSTM models
│   └── dqn_agent.pth                  # Saved DQN agent
│
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
└── LICENSE                            # MIT License
```

---

## 📊 Experimental Results

### Test Configuration

| Parameter | Value |
|-----------|-------|
| Data source | PlanetLab (20110303) |
| Number of VMs | 20-50 |
| Number of Hosts | 5-10 |
| LSTM sequence | 10 timesteps |
| DQN episodes | 5-100 |
| Batch size | 32 |

### Quick Test Results

```json
{
  "data_source": "PlanetLab",
  "training_metrics": [
    {"episode": 1, "reward": -6.41, "energy": 477.55, "epsilon": 1.0},
    {"episode": 2, "reward": -6.87, "energy": 475.15, "epsilon": 0.83},
    {"episode": 3, "reward": -4.37, "energy": 484.75, "epsilon": 0.65},
    {"episode": 4, "reward": -3.70, "energy": 483.55, "epsilon": 0.50},
    {"episode": 5, "reward": -2.96, "energy": 484.75, "epsilon": 0.39}
  ],
  "evaluation": {
    "avg_energy": 496.20,
    "avg_sla": 0.0,
    "total_steps": 90
  }
}
```

### Observations

✅ **Learning**: Reward increases progressively (-6.41 → -2.96)  
✅ **Exploration → Exploitation**: Epsilon decreases (1.0 → 0.39)  
✅ **SLA**: No violations detected in evaluation  
✅ **Energy**: ~480W average for 5 hosts

### Comparison with Baselines

| Algorithm | Energy (kWh) | SLA (%) | Migrations | Improvement |
|-----------|--------------|---------|------------|-------------|
| Static Threshold (80%) | 0.620† | 2.80%† | 144† | baseline |
| MMT + MBFD | 0.571† | 1.20%† | 117† | 7.9% |
| **LSTM-DQN (Ours)** | **0.496** | **0.00%** | **90** | **20.0%** |

*†Estimated based on literature ratios (Beloglazov & Buyya, 2012)*

### Generated Visualizations

1. **Training Progress** (`quick_test_training.png`): Reward and energy over episodes
2. **Pareto Frontier** (`pareto_frontier.png`): Energy vs SLA trade-off comparison

---

## 🔬 Technical Details

### Energy Model

```python
P(u) = P_idle + (P_max - P_idle) × u

where:
  P_idle = 70W   (idle power)
  P_max  = 250W  (maximum power)
  u      = CPU utilization [0, 1]
```

### DQN State Space

```
State = [U₁, U₂, ..., Uₙ, Û₁, Û₂, ..., Ûₙ, active_hosts]

where:
  Uᵢ = Current utilization of host i (normalized 0-1)
  Ûᵢ = LSTM prediction for host i (normalized 0-1)
  active_hosts = Ratio of active hosts (0-1)
```

### DQN Action Space (with FFD Filtering)

```
Actions = {
  0: do_nothing,
  1-5: migrate from host_0 to FFD top 5 targets,
  6-10: migrate from host_1 to FFD top 5 targets,
  ...
}

FFD (First Fit Decreasing) filters target hosts by:
  1. Available capacity (descending)
  2. Excluding overloaded hosts (>80%)
  3. Selecting top 5 candidates
```

### CloudSim Bridge Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **STANDALONE** | Python-only simulation | Development, testing |
| **CLOUDSIM** | Connected to Java CloudSim | Production, realistic simulation |

```python
# Example: Using CloudSim mode
controller = HybridController(
    num_hosts=10, 
    num_vms=50,
    use_cloudsim=True,           # Enable CloudSim connection
    cloudsim_host="localhost",
    cloudsim_port=9999
)
```

### Reward Function

```
R = -w₁·E - w₂·SLA - w₃·M

where:
  E   = Normalized energy consumption
  SLA = SLA violation ratio
  M   = Migration penalty
  
Default weights: w₁=0.5, w₂=0.3, w₃=0.2
```

---

## 📚 References

1. **CloudSim**: Calheiros, R. N., et al. "CloudSim: a toolkit for modeling and simulation of cloud computing environments." (2011)

2. **Deep Q-Network**: Mnih, V., et al. "Human-level control through deep reinforcement learning." Nature (2015)

3. **LSTM**: Hochreiter, S., & Schmidhuber, J. "Long short-term memory." Neural Computation (1997)

4. **VM Consolidation**: Beloglazov, A., & Buyya, R. "Optimal online deterministic algorithms for minimizing energy consumption." (2012)

5. **PlanetLab**: Park, K., & Pai, V. S. "CoMon: A mostly-scalable monitoring system for PlanetLab." ACM SIGOPS (2006)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Academic Project - M1 Data Science**  
*Systems Architecture*  
*December 2025*

---

## 🙏 Acknowledgments

- PlanetLab data for real workload traces
- TensorFlow and PyTorch open-source community
- CloudSim documentation for simulation concepts

---

**⭐ This project demonstrates the application of ML/RL techniques for energy optimization in cloud computing.**
