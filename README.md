# Accelerating Decision-Making for Smart Stormwater Management Through SWMM Surrogate Modeling for Reinforcement Learning

This repository contains the official code and resources for the current preprint: "Accelerating Decision-Making for Smart Stormwater Management Through SWMM Surrogate Modeling for Reinforcement Learning".

## Overview

This research addresses the computational bottleneck in applying Deep Reinforcement Learning (DRL) to the Real-Time Control (RTC) of urban stormwater systems. We develop and validate a Bidirectional LSTM (BiLSTM) surrogate model to emulate a high-fidelity, multi-pond SWMM model of the Conner Creek watershed in Knoxville, TN.

The surrogate model enables a 33-fold acceleration in the training of an Advantage Actor-Critic (A2C) agent, allowing it to converge to a competent multi-objective control policy for flood and erosion mitigation within a practical time budget.

## Repository Structure
├── swmm_model/ # Contains the SWMM .inp file for the environment
├── data/ # Contains the rainfall data used for training/testing
├── saved_models/ # Contains the pre-trained surrogate model and data scaler
├── src/ # Contains the core Python source code
│ ├── a2c_agent.py # The A2C agent model and training functions
│ ├── connor_creek_env.py # The custom Gymnasium environment for the watershed
│ └── subprocess_env.py # Wrapper to run the environment in a separate process
├── train.py # Main script to run the comparative training experiment
├── requirements.txt # Required Python packages
└── README.md # This documentation file

## Installation

1.  **Clone or download the repository.**

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage: Reproducing the Paper's Experiment

The main script `train.py` is used to run the training experiments. You must specify which environment to use with the `--mode` argument.

#### Train the Surrogate Agent (Agent S)

To train the agent on the fast surrogate model for the 5-hour time budget described in the paper:
```bash
python train.py --mode surrogate --time_budget 5
```

#### Train the SWMM Agent (Agent W)
To train the agent on the high-fidelity SWMM simulator for the 5-hour time budget described in the paper:

```bash
python train.py --mode swmm --time_budget 5
```

The script will create a new folder (e.g., training_run_surrogate or training_run_swmm) containing the saved agent weights, a CSV of the training history, and a plot of the learning curve.

## Citation
If you use this code or research in your work, please cite the original paper:

```
Bibtex
@article{Fletcher2026,
  title={Accelerating Decision-Making for Smart Stormwater Management Through SWMM Surrogate Modeling for Reinforcement Learning},
  author={Fletcher, Isidora and Hathaway, Jon and Khojandi, Anahita},
  year={2026},
  month={January},
  journal={ResearchGate Preprint},
  doi={10.13140/RG.2.2.35368.43527},
  note={Preprint}
}
```
