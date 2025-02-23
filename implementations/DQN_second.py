import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime


class MECEnvironment(gym.Env):
    """Enhanced MEC Environment with realistic latency components"""
    def __init__(self, num_edge_servers=3, continuous_action=False):
        super().__init__()
        
        self.num_edge_servers = num_edge_servers
        self.continuous_action = continuous_action
        
        # Action space definition based on algorithm type
        if continuous_action:
            # For SAC: Continuous values between 0 and 1 for each server
            self.action_space = spaces.Box(
                low=0, high=1, shape=(num_edge_servers,), dtype=np.float32
            )
        else:
            # For DQN/DDQN: Discrete server selection
            self.action_space = spaces.Discrete(num_edge_servers)
        
        # Enhanced observation space with more state information
        self.observation_space = spaces.Dict({
            'task_size': spaces.Box(
                low=0,
                high=1,
                shape=(1,),
                dtype=np.float32
            ),
            'server_speeds': spaces.Box(
                low=np.zeros(num_edge_servers),
                high=np.ones(num_edge_servers),
                dtype=np.float32
            ),
            'server_loads': spaces.Box(
                low=np.zeros(num_edge_servers),
                high=np.ones(num_edge_servers),
                dtype=np.float32
            ),
            'network_conditions': spaces.Box(
                low=np.zeros(num_edge_servers),
                high=np.ones(num_edge_servers),
                dtype=np.float32
            ),
            'server_distances': spaces.Box(
                low=np.zeros(num_edge_servers),
                high=np.ones(num_edge_servers),
                dtype=np.float32
            )
        })
        
        # Initialize server characteristics
        self.server_speeds = np.random.uniform(0.5, 1.0, num_edge_servers)  # Processing capability
        self.server_distances = np.random.uniform(0.1, 1.0, num_edge_servers)  # Normalized distances
        self.bandwidth_up = np.random.uniform(0.5, 1.0, num_edge_servers)  # Uplink bandwidth
        self.bandwidth_down = np.random.uniform(0.6, 1.0, num_edge_servers)  # Downlink bandwidth
        self.reset()
        
    def reset(self):
        """Reset environment state"""
        self.current_task_size = np.random.uniform(0.2, 1.0)
        self.server_loads = np.random.uniform(0.1, 0.4, self.num_edge_servers)  # Initial load
        self.network_conditions = np.random.uniform(0.7, 1.0, self.num_edge_servers)  # Network quality
        
        return self._get_observation()
    
    def step(self, action):
        """Execute action and return new state"""
        if self.continuous_action:
            # Convert continuous action to server selection using softmax
            action_probs = F.softmax(torch.FloatTensor(action), dim=0).numpy()
            selected_server = np.argmax(action_probs)
        else:
            selected_server = action
        
        # Calculate total latency components
        total_latency = self._calculate_total_latency(selected_server)
        
        # Update server loads based on the new task
        self._update_server_loads(selected_server)
        
        # Update network conditions with some randomness
        self._update_network_conditions()
        
        # Generate new task and get new observation
        self.current_task_size = np.random.uniform(0.2, 1.0)
        new_observation = self._get_observation()
        
        # Calculate reward (negative normalized latency)
        reward = -np.tanh(total_latency / 10.0)
        
        # Episode never ends
        done = False
        
        # Additional info for monitoring
        info = {
            'selected_server': selected_server,
            'server_load': self.server_loads[selected_server],
            'network_quality': self.network_conditions[selected_server],
            'total_latency': total_latency
        }
        
        return new_observation, reward, done, info
    
    def _calculate_total_latency(self, server_idx):
        """Calculate total latency including all components"""
        # 1. Uplink transmission delay
        uplink_delay = (self.current_task_size / self.bandwidth_up[server_idx]) * \
                      (1 / self.network_conditions[server_idx])
        
        # 2. Propagation delay based on distance
        prop_delay = self.server_distances[server_idx] * 0.1  # Normalized propagation delay
        
        # 3. Processing delay (affected by server load)
        effective_speed = self.server_speeds[server_idx] * (1 - self.server_loads[server_idx])
        processing_delay = self.current_task_size / max(effective_speed, 0.1)
        
        # 4. Downlink transmission delay (assume result size is proportional to task size)
        result_size = self.current_task_size * 0.1  # Assume result is 10% of input size
        downlink_delay = (result_size / self.bandwidth_down[server_idx]) * \
                        (1 / self.network_conditions[server_idx])
        
        # 5. Queuing delay (simple model based on server load)
        queue_delay = self.server_loads[server_idx] * processing_delay
        
        return uplink_delay + prop_delay + processing_delay + downlink_delay + queue_delay
    
    def _update_server_loads(self, selected_server):
        """Update server loads after task assignment"""
        # Increase load for selected server
        self.server_loads[selected_server] = min(
            self.server_loads[selected_server] + self.current_task_size * 0.1,
            1.0
        )
        
        # Decrease other server loads (natural decay)
        for i in range(self.num_edge_servers):
            if i != selected_server:
                self.server_loads[i] = max(
                    self.server_loads[i] - 0.05,
                    0.1
                )
    
    def _update_network_conditions(self):
        """Update network conditions with random fluctuations"""
        fluctuation = np.random.uniform(-0.1, 0.1, self.num_edge_servers)
        self.network_conditions += fluctuation
        self.network_conditions = np.clip(self.network_conditions, 0.3, 1.0)
    
    def _get_observation(self):
        """Construct the observation dictionary"""
        return {
            'task_size': np.array([self.current_task_size], dtype=np.float32),
            'server_speeds': self.server_speeds.astype(np.float32),
            'server_loads': self.server_loads.astype(np.float32),
            'network_conditions': self.network_conditions.astype(np.float32),
            'server_distances': self.server_distances.astype(np.float32)
        }
    

class DQNNetwork(nn.Module):
    """Deep Q-Network with extended architecture"""
    def __init__(self, state_size, action_size, device):
        super(DQNNetwork, self).__init__()
        
        self.device = device
        
        # Verify state size
        print(f"Network initialized with state_size: {state_size}")
        
        # Network layers
        self.fc1 = nn.Linear(state_size, 256)  # Increased from 64 to handle more features
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_size)
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(256)
        self.ln2 = nn.LayerNorm(256)
        self.ln3 = nn.LayerNorm(128)
        
        self.dropout = nn.Dropout(0.1)
        
        # Move to device
        self.to(device)
        
    def forward(self, x):
        x = x.to(self.device)
        x = torch.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.ln3(self.fc3(x)))
        return self.fc4(x)
    

def train_mec_dqn():
    """Training loop for simplified MEC environment"""
    # Environment setup
    env = MECEnvironment(num_edge_servers=10, continuous_action=False)
    
    # Calculate state size
    state_size = (1 +  # task_size
                env.num_edge_servers +  # server_speeds
                env.num_edge_servers +  # server_loads
                env.num_edge_servers +  # network_conditions
                env.num_edge_servers)   # server_distances
    action_size = env.action_space.n

    print(f"Calculated state size: {state_size}")
    print(f"Action size: {action_size}")
    
    # Initialize DQN agent
    agent = DQNAgent(state_size, action_size)
    
    # Training parameters
    num_episodes = 2000
    max_steps = 100
    target_update_frequency = 10
    eval_frequency = 50
    
    # Enhanced metrics tracking
    metrics = {
        'rewards': [],
        'losses': [],
        'epsilons': [],
        'avg_rewards': [],
        'latencies': [],  # Track latencies
        'server_loads': []  # Track server loads
    }
    
    for episode in range(num_episodes):
        obs = env.reset()
        # Print observation structure in first episode
        if episode == 0:
            print("\nObservation structure:")
            for key, value in obs.items():
                print(f"{key}: shape {value.shape}")
        
        # Flatten the observation
        state = np.concatenate([
            obs['task_size'],
            obs['server_speeds'],
            obs['server_loads'],
            obs['network_conditions'],
            obs['server_distances']
        ])
        
        episode_reward = 0
        episode_losses = []
        episode_latencies = []
        
        for step in range(max_steps):
            # Select and perform action
            action = agent.select_action(state)
            next_obs, reward, done, info = env.step(action)
            
            # Flatten the next observation
            next_state = np.concatenate([
                next_obs['task_size'],
                next_obs['server_speeds'],
                next_obs['server_loads'],
                next_obs['network_conditions'],
                next_obs['server_distances']
            ])
            
            # Store transition and train
            agent.replay_buffer.push(state, action, reward, next_state, done)
            loss = agent.train()
            if loss is not None:
                episode_losses.append(loss)
            
            # Track metrics
            episode_latencies.append(info['total_latency'])
            state = next_state
            episode_reward += reward
        
        # Update target network and episode counter
        if episode % target_update_frequency == 0:
            agent.update_target_network()
        agent.current_episode = episode
        
        # Store metrics
        metrics['rewards'].append(episode_reward)
        metrics['latencies'].append(np.mean(episode_latencies))
        if episode_losses:
            metrics['losses'].append(np.mean(episode_losses))
        metrics['epsilons'].append(agent.epsilon)
        metrics['server_loads'].append(np.mean(obs['server_loads']))
        
        # Calculate running average
        if episode >= eval_frequency:
            avg_reward = np.mean(metrics['rewards'][-eval_frequency:])
            metrics['avg_rewards'].append(avg_reward)
        
        # Print progress
        if episode % eval_frequency == 0:
            avg_reward = np.mean(metrics['rewards'][-eval_frequency:])
            avg_latency = np.mean(metrics['latencies'][-eval_frequency:])
            avg_loss = np.mean(metrics['losses'][-eval_frequency:]) if metrics['losses'] else 0
            print(f"Episode {episode}/{num_episodes}, "
                  f"Average Reward: {avg_reward:.2f}, "
                  f"Average Latency: {avg_latency:.2f}, "
                  f"Average Loss: {avg_loss:.4f}, "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    return agent, metrics


class ReplayBuffer:
    """Experience Replay Buffer"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
    
class DQNAgent:
    """DQN Agent for VEC task offloading"""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # DQN hyperparameters
        self.gamma = 0.99  # discount factor
        self.epsilon = 1.0  # starting exploration rate
        self.epsilon_min = 0.05  # minimum exploration rate
        self.epsilon_decay_steps = 1500  # number of episodes to reach epsilon_min
        self.current_episode = 0  # track current episode for linear decay
        
        self.learning_rate = 0.001
        self.batch_size = 128
        self.min_replay_size = 1000
        
        # Create Q-Networks (current and target)
        self.q_network = DQNNetwork(state_size, action_size, self.device)
        self.target_network = DQNNetwork(state_size, action_size, self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), 
                                  lr=self.learning_rate,
                                  weight_decay=1e-5)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(100000)
    
    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        # Update epsilon using linear decay
        if self.current_episode < self.epsilon_decay_steps:
            self.epsilon = 1.0 - (1.0 - self.epsilon_min) * (self.current_episode / self.epsilon_decay_steps)
        else:
            self.epsilon = self.epsilon_min
            
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state)
            if state.dim() == 1:
                state = state.unsqueeze(0)
            state = state.to(self.device)
            q_values = self.q_network(state)
            return q_values.cpu().argmax().item()
    
    def train(self):
        """Train the agent using experience replay"""
        if len(self.replay_buffer) < self.min_replay_size:
            return None
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors and move to device
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Compute next Q values with target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and update
        loss = F.huber_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network parameters"""
        self.target_network.load_state_dict(self.q_network.state_dict())

import os
import json
from datetime import datetime

def save_training_results(metrics, save_dir="results"):
    """Save training metrics and plots to specified directory"""
    # Create results directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save metrics to JSON file
    metrics_filename = os.path.join(save_dir, f"metrics_{timestamp}.json")
    json_metrics = {
        'rewards': [float(r) for r in metrics['rewards']],  # Convert numpy floats to Python floats
        'losses': [float(l) for l in metrics['losses']],
        'epsilons': [float(e) for e in metrics['epsilons']],
        'avg_rewards': [float(ar) for ar in metrics.get('avg_rewards', [])]
    }
    with open(metrics_filename, 'w') as f:
        json.dump(json_metrics, f, indent=4)
    
    # Create and save plot
    plt.figure(figsize=(15, 10))
    
    # Plot rewards
    plt.subplot(3, 1, 1)
    plt.plot(metrics['rewards'], alpha=0.6, label='Episode Reward')
    
    # Calculate and plot moving average
    window_size = 10
    if len(metrics['rewards']) > window_size:
        moving_avg = np.convolve(metrics['rewards'], 
                               np.ones(window_size)/window_size, 
                               mode='valid')
        x_avg = np.arange(window_size-1, len(metrics['rewards']))
        plt.plot(x_avg, moving_avg, 'r-', label='Moving Average')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.title('Training Rewards')
    plt.grid(True)
    
    # Plot losses
    plt.subplot(3, 1, 2)
    plt.plot(metrics['losses'], label='Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    
    # Plot epsilon
    plt.subplot(3, 1, 3)
    plt.plot(metrics['epsilons'], label='Epsilon')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Exploration Rate (Epsilon)')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = os.path.join(save_dir, f"training_plot_{timestamp}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Results saved to directory: {save_dir}")
    print(f"Metrics saved as: metrics_{timestamp}.json")
    print(f"Plot saved as: training_plot_{timestamp}.png")

if __name__ == "__main__":
    # Create directory for this specific run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("results", f"run_{timestamp}")
    
    # Train the agent
    agent, metrics = train_mec_dqn()
    
    # Save results
    save_training_results(metrics, save_dir=run_dir)
    
    # Also display the plot
    plt.show()