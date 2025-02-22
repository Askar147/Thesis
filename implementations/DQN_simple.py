import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime


class MECEnvironment(gym.Env):
    """Simplified MEC Environment focusing only on latency optimization"""
    def __init__(self, num_edge_servers=3):
        super().__init__()
        
        self.num_edge_servers = num_edge_servers
        
        # Action space: Which edge server to offload to
        self.action_space = spaces.Discrete(num_edge_servers)
        
        # Observation space: Task size and server processing speeds
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
            )
        })
        
        # Fixed server processing speeds (normalized)
        self.server_speeds = np.random.uniform(0.5, 1.0, num_edge_servers)
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        # Generate a new random task size
        self.current_task_size = np.random.uniform(0.2, 1.0)
        
        return self._get_observation()
    
    def step(self, action):
        """Execute action and return new state"""
        # Calculate latency for the chosen server
        latency = self._calculate_latency(action)
        
        # Reward is negative latency (we want to minimize latency)
        reward = -latency
        
        # Generate new task for next state
        self.current_task_size = np.random.uniform(0.2, 1.0)
        
        # Get new observation
        observation = self._get_observation()
        
        # Episode never ends
        done = False
        
        return observation, reward, done, {}
    
    def _get_observation(self):
        """Construct the observation dictionary"""
        return {
            'task_size': np.array([self.current_task_size], dtype=np.float32),
            'server_speeds': self.server_speeds.astype(np.float32)
        }
    
    def _calculate_latency(self, server_idx):
        """Calculate latency for the offloading decision"""
        # Simple latency model: task_size / server_speed
        processing_time = self.current_task_size / self.server_speeds[server_idx]
        return processing_time
    

class DQNNetwork(nn.Module):
    """Simplified DQN Network for latency optimization"""
    def __init__(self, state_size, action_size, device):
        super(DQNNetwork, self).__init__()
        
        self.device = device
        
        # Simpler network architecture
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
        # Move network to device
        self.to(device)
        
    def forward(self, x):
        x = x.to(self.device)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    

def train_mec_dqn():
    """Training loop for simplified MEC environment"""
    # Environment setup
    env = MECEnvironment(num_edge_servers=3)
    
    # Calculate state size (task_size + server_speeds)
    state_size = 1 + env.num_edge_servers
    action_size = env.action_space.n
    
    # Initialize DQN agent
    agent = DQNAgent(state_size, action_size)
    
    # Training parameters
    num_episodes = 3000
    max_steps = 100
    target_update_frequency = 10
    eval_frequency = 50
    
    # Metrics tracking
    metrics = {
        'rewards': [],
        'losses': [],
        'epsilons': [],
        'avg_rewards': []
    }
    
    for episode in range(num_episodes):
        state = env.reset()
        # Flatten the state dictionary
        state = np.concatenate([
            state['task_size'],
            state['server_speeds']
        ])
        
        episode_reward = 0
        episode_losses = []
        
        for step in range(max_steps):
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            # Flatten the next state
            next_state = np.concatenate([
                next_state['task_size'],
                next_state['server_speeds']
            ])
            
            # Store transition and train
            agent.replay_buffer.push(state, action, reward, next_state, done)
            loss = agent.train()
            if loss is not None:
                episode_losses.append(loss)
            
            state = next_state
            episode_reward += reward
        
        # Update target network and episode counter
        if episode % target_update_frequency == 0:
            agent.update_target_network()
        agent.current_episode = episode
        
        # Store metrics
        metrics['rewards'].append(episode_reward)
        if episode_losses:
            metrics['losses'].append(np.mean(episode_losses))
        metrics['epsilons'].append(agent.epsilon)
        
        # Calculate running average
        if episode >= eval_frequency:
            avg_reward = np.mean(metrics['rewards'][-eval_frequency:])
            metrics['avg_rewards'].append(avg_reward)
        
        # Print progress
        if episode % eval_frequency == 0:
            avg_reward = np.mean(metrics['rewards'][-eval_frequency:])
            avg_loss = np.mean(metrics['losses'][-eval_frequency:]) if metrics['losses'] else 0
            print(f"Episode {episode}/{num_episodes}, "
                  f"Average Reward: {avg_reward:.2f}, "
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
        self.epsilon_min = 0.1  # minimum exploration rate
        self.epsilon_decay_steps = 2500  # number of episodes to reach epsilon_min
        self.epsilon_min = 0.05  # slightly lower minimum exploration
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
            return 0
        
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
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
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