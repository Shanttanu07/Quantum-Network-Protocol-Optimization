# Quantum Repeater Chain Protocol Optimization  

This project focuses on optimizing Quantum Repeater Chain Protocols using Reinforcement Learning (RL) techniques to enhance the efficiency of quantum communication networks. Specifically, the project implements two main versions of RL agents for adaptive cut-off threshold optimization to maximize the secret key rate and reduce latency. These agents utilize advanced simulation models to evaluate the success probabilities, Werner parameters, and waiting time distributions in various repeater chain protocols.  

## Project Structure  

### Main Files  

1. **RL.py**  
   - Implements a Q-learning agent for optimizing cut-off parameters within a Quantum Repeater Chain Protocol. The Q-learning agent learns to adjust the cut-off thresholds dynamically based on the reward feedback from the environment, aiming to balance fidelity, waiting time, and secret key rate.  
   - Core Components:  
     - `QuantumProtocolEnv`: An environment representing the repeater chain protocol setup, where the agent interacts and receives rewards based on the secret key rate and operational costs.  
     - `QAgent`: Defines the Q-learning algorithm, including action selection, Q-table updates, and exploration-exploitation strategies.  
     - `train_qlearning`: Manages the training loop, integrating environment interactions and Q-table updates for policy improvement.  

2. **RL_1.py**  
   - This file implements a Proximal Policy Optimization (PPO) agent, providing an advanced, efficient alternative to the Q-learning model. The PPO agent optimizes cut-off parameters with improved stability and adaptability, particularly suited for complex, continuous action spaces in the quantum repeater protocol.  
   - Key Features:  
     - `PPOAgent`: Defines the PPO algorithm with policy and value networks, enabling stable policy updates through clipped loss functions.  
     - `train_ppo`: Manages the training loop for PPO, coordinating between the agent's interactions and gradient updates.  
     - Enhanced exploration and exploitation strategies, providing better performance in large-scale networks.  

3. **repeater_algorithm.py**  
   - Contains the core simulation functions for the Repeater Chain Protocols, providing methods to compute convolutional waiting times, protocol success probabilities, and Werner parameters.  
   - Major Functions:  
     - `RepeaterChainSimulation`: Main class to run the iterative convolution algorithm, allowing for various configurations like memory cut-offs, fidelity cut-offs, and GPU support.  
     - `plot_algorithm`: Visualization utility to plot the Probability Mass Function (PMF) and Werner parameters across multiple protocol units.  

4. **protocol_units.py**  
   - Defines individual protocol units (e.g., swap, distillation, and cut-off functions) used in the repeater chain simulation. Each function in this module encapsulates the necessary operations for a specific protocol action, such as calculating success probability and adjusting Werner parameters.  
   - Functions include:  
     - `get_swap_wout`, `get_dist_prob_suc`, and `get_dist_prob_wout`: Calculate success probabilities and resulting Werner parameters for entanglement swapping and distillation.  
     - `fidelity_cut_off`, `memory_cut_off`: Implement the threshold mechanisms for adaptive cut-off strategies.  

5. **repeater_mc.py**  
   - Runs Monte Carlo simulations for the Quantum Repeater Protocol, providing a comparative approach to the deterministic algorithm. The simulations produce probabilistic results for the waiting time and Werner parameter distributions, which are used to validate the effectiveness of the RL-based optimization.  
   - Key Functions:  
     - `sample_swap` and `sample_dist`: Sampling functions for swap and distillation events.  
     - `create_pmf_from_samples` and `compute_werner`: Compute probability distributions and Werner parameters from sampled data.  

6. **utility_functions.py**  
   - Contains various utility functions to support the simulation and RL models, including mathematical operations on Werner parameters, probability functions, and entropy calculations.  
   - Important Functions:  
     - `werner_to_matrix` and `matrix_to_werner`: Transform Werner parameters to matrix representations and vice versa.  
     - `secret_key_rate`: Computes the secret key rate for a given probability distribution and Werner function.  
     - `get_mean_werner` and `get_mean_waiting_time`: Calculate mean Werner parameters and waiting times from probability distributions.  

7. **examples.py**  
   - Provides example scripts for running both Monte Carlo and deterministic simulations of the repeater chain protocol. Includes a `swap_protocol` example, which calculates the waiting time distribution and Werner parameter using both approaches.  
   - Functionality:  
     - Demonstrates different protocol configurations, showing how the agent optimizes across varied setups.  
     - Outputs visualizations for key metrics, like the Probability Mass Function and Werner parameters.  

8. **logging_utilities.py**  
   - Handles logging of simulation parameters and results, providing a mechanism to track and save data across different protocol runs.  
   - Features:  
     - `log_params` and `log_finish`: Log the parameters and completion status of simulations.  
     - `save_data` and `load_data`: Save and retrieve simulation data for post-analysis.  

## Installation and Setup  

1. **Requirements**: Ensure the following Python packages are installed:  
   ```bash  
   pip install numpy torch matplotlib numba scipy

2. **Running the Optimization Code**
   - Example for training the Q-learning Agent:
   ```bash
   python RL.py

  - To use the PPO agent:

   ```bash
   python RL_1.py
