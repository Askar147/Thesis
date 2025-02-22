import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import gym
from gym import spaces

class MECEnvironment(gym.Env):
    """Custom Environment for Mobile Edge Computing with Different Task Complexities"""
    def __init__(self, num_edge_servers=5):
        super().__init__()
        
        self.num_edge_servers = num_edge_servers
        self.task_types = 3  # Small, Medium, Large Dijkstra
        
        # Action space: Which edge server to offload to
        self.action_space = spaces.Discrete(num_edge_servers)
        
        # Observation space: Task queue states and server states
        self.observation_space = spaces.Dict({
            'task_queue': spaces.Box(
                low=np.zeros(self.task_types),
                high=np.ones(self.task_types),
                dtype=np.float32
            ),
            'server_states': spaces.Box(
                low=np.zeros(num_edge_servers),
                high=np.ones(num_edge_servers),  # Normalized server load
                dtype=np.float32
            ),
            'network_conditions': spaces.Box(
                low=np.zeros(num_edge_servers),
                high=np.ones(num_edge_servers),  # Normalized network conditions
                dtype=np.float32
            )
        })
        
        # Task characteristics
        self.task_sizes = {
            'small': 0.2,    # 200x200 Dijkstra
            'medium': 0.5,   # 416x160 Dijkstra
            'large': 1.0     # 832x320 Dijkstra
        }
        
        self.server_processing_speeds = np.random.uniform(0.5, 1.0, num_edge_servers)
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        # Initialize task queue with random tasks
        self.task_queue = np.random.rand(self.task_types)
        
        # Initialize server loads
        self.server_loads = np.zeros(self.num_edge_servers)
        
        # Initialize network conditions
        self.network_conditions = np.random.uniform(0.5, 1.0, self.num_edge_servers)
        
        return self._get_observation()
    
    def step(self, action):
        """Execute action and return new state"""
        # Get the heaviest waiting task
        task_idx = np.argmax(self.task_queue)
        task_size = list(self.task_sizes.values())[task_idx]
        
        # Calculate execution metrics
        energy = self._calculate_energy(action, task_size)
        latency = self._calculate_latency(action, task_size)
        
        # Update server load
        self.server_loads[action] += task_size
        self.server_loads = np.clip(self.server_loads, 0, 1)  # Normalize
        
        # Mark task as completed
        self.task_queue[task_idx] = 0
        
        # Calculate reward
        reward = self._calculate_reward(energy, latency, self.server_loads[action])
        
        # Update network conditions (add some dynamicity)
        self._update_network_conditions()
        
        # Get new observation
        observation = self._get_observation()
        
        # Check if episode should end
        done = np.all(self.task_queue == 0) or np.any(self.server_loads >= 1.0)
        
        return observation, reward, done, {}
    
    def _get_observation(self):
        """Construct the observation dictionary"""
        return {
            'task_queue': self.task_queue.astype(np.float32),
            'server_states': self.server_loads.astype(np.float32),
            'network_conditions': self.network_conditions.astype(np.float32)
        }
    
    def _calculate_energy(self, server_idx, task_size):
        """Calculate energy consumption for the offloading decision"""
        # Energy cost depends on task size, network condition and server load
        transmission_energy = task_size / self.network_conditions[server_idx]
        processing_energy = task_size * (1 + self.server_loads[server_idx])
        return transmission_energy + processing_energy
    
    def _calculate_latency(self, server_idx, task_size):
        """Calculate latency for the offloading decision"""
        # Latency includes transmission and processing time
        transmission_time = task_size / self.network_conditions[server_idx]
        processing_time = task_size / (self.server_processing_speeds[server_idx] * 
                                     (1 - self.server_loads[server_idx]))
        return transmission_time + processing_time
    
    def _calculate_reward(self, energy, latency, server_load):
        """Calculate reward based on energy, latency and load balancing"""
        energy_weight = 0.4
        latency_weight = 0.4
        load_weight = 0.2
        
        # Normalize values
        norm_energy = -energy / 2.0  # Assuming max energy is 2.0
        norm_latency = -latency / 2.0  # Assuming max latency is 2.0
        load_balancing = -abs(server_load - np.mean(self.server_loads))
        
        return (energy_weight * norm_energy + 
                latency_weight * norm_latency + 
                load_weight * load_balancing)
    
    def _update_network_conditions(self):
        """Update network conditions with some random fluctuation"""
        fluctuation = np.random.uniform(-0.1, 0.1, self.num_edge_servers)
        self.network_conditions += fluctuation
        self.network_conditions = np.clip(self.network_conditions, 0.1, 1.0)

class DQNNetwork(nn.Module):
    """Deep Q-Network with extended architecture"""
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, action_size)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        return self.fc4(x)

def train_mec_dqn():
    """Main training loop for MEC environment"""
    # Environment setup
    env = MECEnvironment(num_edge_servers=5)
    
    # Calculate state size
    state_size = (env.task_types + 
                 env.num_edge_servers + 
                 env.num_edge_servers)  # task_queue + server_states + network_conditions
    action_size = env.action_space.n
    
    # Initialize DQN agent
    agent = DQNAgent(state_size, action_size)
    
    # Training parameters
    num_episodes = 1000
    max_steps = 100
    target_update_frequency = 10
    eval_frequency = 50
    
    # Metrics tracking
    rewards_history = []
    
    for episode in range(num_episodes):
        state = env.reset()
        # Flatten the state dictionary
        state = np.concatenate([
            state['task_queue'],
            state['server_states'],
            state['network_conditions']
        ])
        
        episode_reward = 0
        
        for step in range(max_steps):
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            # Flatten the next state
            next_state = np.concatenate([
                next_state['task_queue'],
                next_state['server_states'],
                next_state['network_conditions']
            ])
            
            # Store transition and train
            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.train()
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Update target network
        if episode % target_update_frequency == 0:
            agent.update_target_network()
        
        # Store rewards for plotting
        rewards_history.append(episode_reward)
        
        # Print progress
        if episode % eval_frequency == 0:
            avg_reward = np.mean(rewards_history[-eval_frequency:])
            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"Average Reward: {avg_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.2f}")
    
    return agent, rewards_history

if __name__ == "__main__":
    agent, rewards = train_mec_dqn()
    
    # Plot training results
    import matplotlib.pyplot as plt
    plt.plot(rewards)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()