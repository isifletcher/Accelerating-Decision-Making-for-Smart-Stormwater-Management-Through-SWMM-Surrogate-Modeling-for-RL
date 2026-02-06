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
