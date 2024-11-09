import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from multiprocessing import Pool, cpu_count
import random
from examples import run_protocol_simulation
from repeater_algorithm import RepeaterChainSimulation
from protocol_units import *
from protocol_units_efficient import *

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc_policy = nn.Linear(128, action_size)
        self.fc_value = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        policy_logits = self.fc_policy(x)
        value = self.fc_value(x)
        return policy_logits, value

class PPOAgent:
    def __init__(self, state_size, action_size, lr=3e-4, gamma=0.99, eps_clip=0.2):
        self.policy = ActorCritic(state_size, action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.state_size = state_size
        self.action_size = action_size
        self.optimal_configuration = None
        self.max_reward = float('-inf')

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        policy_logits, _ = self.policy(state)
        dist = Categorical(logits=policy_logits)
        protocol_index = dist.sample().item()  # Select protocol index as before
        
        # Generate cutoffs and N_memory as part of the action tuple
        cutoffs = {
            "mt_cut": [random.randint(10, 50) for _ in range(3)],   # Example range, adjust as needed
            "w_cut": [random.uniform(0.8, 1.0) for _ in range(3)],
            "rt_cut": [random.randint(10, 50) for _ in range(3)]
        }
        N_memory = random.randint(1, 3)
        
        # Format action as a tuple: (protocol_index, cutoffs, N_memory)
        action = (protocol_index, tuple(sorted(cutoffs.items())), N_memory)
        
        return action, dist.log_prob(torch.tensor(protocol_index))

    def optimize_model(self, memory):
        # Unpack the memory
        states, actions, rewards, next_states, dones, old_log_probs = memory

        # Convert each part of the memory to tensors
        states = torch.tensor(states, dtype=torch.float32)
        # Extract the first element from each action tuple if actions are tuples
        actions = torch.tensor([a[0] if isinstance(a, tuple) else a for a in actions], dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)

        # Compute values using the value network
        _, values = self.policy(states)
        values = values.squeeze()

        # Compute loss with correct input types
        loss = self.compute_loss(old_log_probs, states, actions, rewards, next_states, dones, values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def compute_loss(self, old_log_probs, states, actions, rewards, next_states, dones, values):
        new_log_probs, returns, advantages = [], [], []
        discounted_reward = 0

        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)

        # Convert returns to tensor and normalize
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        
        # Calculate advantages
        advantages = returns - values.detach()

        # Calculate the surrogate loss
        for log_prob, adv in zip(old_log_probs, advantages):
            ratio = torch.exp(log_prob - old_log_probs)  # Ensure old_log_probs is now a tensor
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * adv
            loss = -torch.min(surr1, surr2) + 0.5 * (returns - values)**2
            new_log_probs.append(loss)

        return torch.cat(new_log_probs).mean()


    def evaluate_policy(self, env):
        state = env.reset()
        done = False
        cumulative_reward = 0
        action_log = []

        while not done:
            action, _ = self.select_action(state)
            next_state, reward, done = env.step(action)
            cumulative_reward += reward
            action_log.append((state, action, reward))
            state = next_state

        if cumulative_reward > self.max_reward:
            self.max_reward = cumulative_reward
            self.optimal_configuration = action_log

    def get_optimal_configuration(self):
        """Prints the optimal configuration and reward obtained."""
        print("\nOptimal Configuration Achieved:")
        print(f"Max Cumulative Reward: {self.max_reward}")
        for i, (state, action, reward) in enumerate(self.optimal_configuration):
            print(f"Step {i+1}: State={state}, Action={action}, Reward={reward}")

class QuantumProtocolEnv:
    def __init__(self, hardware_params, nesting_levels=3):
        self.hardware_params = hardware_params
        self.protocol_count = 8
        self.nesting_levels = nesting_levels
        self.mt_cut_range = (10, 50)
        self.rt_cut_range = (10, 50)
        self.w_cut_range = (0.8, 1.0)
        self.memory_range = (1, 3)

    def reset(self):
        cutoffs = {
            "mt_cut": [random.randint(*self.mt_cut_range) for _ in range(self.nesting_levels)],
            "w_cut": [random.uniform(*self.w_cut_range) for _ in range(self.nesting_levels)],
            "rt_cut": [random.randint(*self.rt_cut_range) for _ in range(self.nesting_levels)]
        }
        N_memory = random.randint(*self.memory_range)
        
        # Flatten the state as a list of numeric values for compatibility
        self.state = cutoffs["mt_cut"] + cutoffs["w_cut"] + cutoffs["rt_cut"] + [N_memory]
        return self.state

    def step(self, action):
        protocol_index = action[0]
        cutoffs = action[1]
        N_memory = action[2]
        
        # Convert cutoffs tuple back to dictionary
        cutoffs_dict = dict(cutoffs)
        
        # Run the protocol simulation and calculate reward
        secret_key_rate, cost = self.simulate_protocol(protocol_index, cutoffs_dict, N_memory)
        reward = secret_key_rate - cost * N_memory
        
        # Update state as a numeric list for compatibility
        self.state = list(cutoffs_dict["mt_cut"]) + list(cutoffs_dict["w_cut"]) + list(cutoffs_dict["rt_cut"]) + [N_memory]
        done = reward > 0.9 * secret_key_rate
        return self.state, reward, done

    def simulate_protocol(self, protocol_index, cutoffs, N_memory):
        cutoffs_dict = dict(cutoffs)
        parameters = self.hardware_params.copy()
        parameters["mt_cut"] = cutoffs_dict["mt_cut"]
        parameters["w_cut"] = cutoffs_dict["w_cut"]
        parameters["rt_cut"] = cutoffs_dict["rt_cut"]
        parameters["N_memory"] = N_memory
        secret_key_rate = run_protocol_simulation(2, parameters)
        cost = 0.1
        return secret_key_rate, cost


def parallel_env_step(args):
    env, action = args
    return env.step(action)

def train_ppo(env, agent, episodes=1000, batch_size=32):
    for episode in range(episodes):
        state = env.reset()
        memory = []

        for _ in range(batch_size):
            action, log_prob = agent.select_action(state)
            with Pool(cpu_count()) as pool:
                result = pool.apply(parallel_env_step, ((env, action),))
            next_state, reward, done = result
            memory.append((state, action, reward, next_state, done, log_prob))

            state = next_state
            if done:
                break

        agent.optimize_model(zip(*memory))
        agent.evaluate_policy(env)  # Evaluate policy after each episode
        print(f"Episode {episode+1}/{episodes} completed.")

    agent.get_optimal_configuration()  # Print optimal configuration

if __name__ == "__main__":
    # Define hardware parameters
    hardware_params = {
        "p_gen": 0.1,
        "p_swap": 0.5,
        "w0": 0.85,
        "t_coh": 400,
        "t_trunc": 50000
    }

    # Initialize environment and calculate state size dynamically based on initial reset state
    env = QuantumProtocolEnv(hardware_params)
    initial_state = env.reset()
    state_size = len(initial_state)  # Calculate state size based on the environment's reset state
    action_size = env.protocol_count  # Number of protocol options

    # Initialize PPO agent with dynamically calculated state_size
    agent = PPOAgent(state_size, action_size)

    # Train the PPO agent
    train_ppo(env, agent, episodes=2)


