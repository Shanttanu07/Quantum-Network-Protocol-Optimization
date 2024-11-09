import numpy as np
import random
from examples import run_protocol_simulation  
from repeater_algorithm import RepeaterChainSimulation
from optimize_cutoff import CutoffOptimizer
from protocol_units import *
from protocol_units_efficient import *
from utility_functions import *
from logging_utilities import log_init
class QuantumProtocolEnv:
    def __init__(self, hardware_params, nesting_levels=3):
        self.hardware_params = hardware_params
        self.protocol_count = 8  # Number of protocol options
        self.nesting_levels = nesting_levels
        self.mt_cut_range = (10, 50)  # Smaller range
        self.rt_cut_range = (10, 50)  # Smaller range
        self.w_cut_range = (0.8, 1.0)  # Higher fidelity range
        self.memory_range = (1, 3)  # Fewer memory options


    def reset(self):
        # Initialize cut-offs and N_memory for each level
        cutoffs = {
            "mt_cut": tuple(random.randint(*self.mt_cut_range) for _ in range(self.nesting_levels)),
            "w_cut": tuple(random.uniform(*self.w_cut_range) for _ in range(self.nesting_levels)),
            "rt_cut": tuple(random.randint(*self.rt_cut_range) for _ in range(self.nesting_levels))
        }
        N_memory = random.randint(*self.memory_range)
        # Convert state to a fully hashable format with tuples
        self.state = (tuple(sorted(cutoffs.items())), N_memory)
        return self.state

    def step(self, action):
        protocol_index = action[0]  # Protocol choice (0-7)
        cutoffs = action[1]         # Tuple of cut-off parameters for each level
        N_memory = action[2]        # Number of quantum memories

        # Convert cutoffs tuple back to dictionary
        cutoffs_dict = dict(cutoffs)

        # Run the simulation with given parameters
        secret_key_rate, cost = self.simulate_protocol(protocol_index, cutoffs_dict, N_memory)
        
        # Calculate reward based on secret_key_rate and memory cost
        reward = secret_key_rate - cost * N_memory
        # Convert state to a fully hashable format with tuples
        self.state = (tuple(sorted(cutoffs_dict.items())), N_memory)
        
        return self.state, reward, secret_key_rate


    def simulate_protocol(self, protocol_index, cutoffs, N_memory):
        """
        Simulate the protocol based on the given parameters and returns the secret key rate and cost.
        
        Args:
            protocol_index (int): Protocol index to run.
            cutoffs (tuple): Tuple containing sorted key-value pairs of cut-off parameters.
            N_memory (int): Number of quantum memories per node.
            
        Returns:
            float: secret_key_rate from the protocol.
            float: cost associated with using the memory.
        """
        # Convert cutoffs tuple back to dictionary
        cutoffs_dict = dict(cutoffs)

        # Copy hardware parameters and add protocol-specific settings
        parameters = self.hardware_params.copy()
        parameters["mt_cut"] = cutoffs_dict["mt_cut"]
        parameters["w_cut"] = cutoffs_dict["w_cut"]
        parameters["rt_cut"] = cutoffs_dict["rt_cut"]
        parameters["N_memory"] = N_memory

        # Run the protocol simulation
        secret_key_rate = run_protocol_simulation(2, parameters)

        # Define cost associated with using quantum memories
        cost = 0.1  # Cost per quantum memory, adjust based on your setup

        return secret_key_rate, cost


class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.99):
        self.env = env
        self.q_table = {}  # Dictionary to store Q-values
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            # Explore: randomly choose an action
            protocol = random.randint(0, self.env.protocol_count - 1)
            cutoffs = {
                "mt_cut": tuple(random.randint(*self.env.mt_cut_range) for _ in range(self.env.nesting_levels)),
                "w_cut": tuple(random.uniform(*self.env.w_cut_range) for _ in range(self.env.nesting_levels)),
                "rt_cut": tuple(random.randint(*self.env.rt_cut_range) for _ in range(self.env.nesting_levels))
            }
            N_memory = random.randint(*self.env.memory_range)
            # Return action as a tuple
            return (protocol, tuple(sorted(cutoffs.items())), N_memory)
        else:
            # Exploit: choose the best action based on Q-table
            if state in self.q_table:
                return max(self.q_table[state], key=self.q_table[state].get)
            else:
                return self.choose_action(state)

    def update_q_table(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = {}

        # Convert action to a hashable tuple
        action = (action[0], action[1], action[2])

        if action not in self.q_table[state]:
            self.q_table[state][action] = 0

        # Ensure next_state is in q_table
        if next_state not in self.q_table:
            self.q_table[next_state] = {}

        # Q-learning update rule
        # Get the best action for next_state as a tuple, defaulting to a hashable tuple
        best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get, default=(0, (), 0))
        td_target = reward + self.discount_factor * self.q_table[next_state].get(best_next_action, 0)
        td_delta = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_delta




    def train(self, episodes=1000):
        for episode in range(episodes):
            state = self.env.reset()
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, secret_key_rate = self.env.step(action)
                self.update_q_table(state, action, reward, next_state)
                
                state = next_state
                done = reward > 0.9 * secret_key_rate  # Stop based on a threshold
            
            self.exploration_rate *= self.exploration_decay
            print(f"Episode {episode+1}: Exploration Rate {self.exploration_rate}")
        
        print("Training complete")


    def get_optimal_configuration(self):
        """Prints the optimal configuration (cut-offs and N_memory) with the highest Q-value."""
        optimal_state = None
        optimal_action = None
        max_q_value = float("-inf")

        # Iterate through the Q-table to find the state-action pair with the highest Q-value
        for state, actions in self.q_table.items():
            for action, q_value in actions.items():
                if q_value > max_q_value:
                    max_q_value = q_value
                    optimal_state = state
                    optimal_action = action

        # Print the optimal configuration
        print("\nOptimal Configuration Found:")
        print(f"Optimal State (Cut-offs): {optimal_state[0]}")
        print(f"Optimal N_memory: {optimal_state[1]}")
        print(f"Optimal Protocol Index: {optimal_action[0]}")
        print(f"Cut-offs (mt_cut, w_cut, rt_cut per level): {optimal_action[1]}")
        print(f"N_memory: {optimal_action[2]}")
        print(f"Highest Q-value (Reward): {max_q_value}")

hardware_params = {
    "p_gen": 0.1,
    "p_swap": 0.5,
    "w0": 0.85,
    "t_coh": 400,
    "t_trunc": 50000  
}


# Initialize environment and agent
env = QuantumProtocolEnv(hardware_params)
agent = QLearningAgent(env)

# Train the agent
agent.train(episodes=2) 