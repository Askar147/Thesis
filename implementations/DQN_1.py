import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt

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
        energy_weight = 0
        latency_weight = 100
        load_weight = 0
        
        # Normalize values
        norm_energy = -energy / 2.0  # Assuming max energy is 2.0
        norm_latency = -latency / 10.0  # Assuming max latency is 2.0
        load_balancing = -abs(server_load - np.mean(self.server_loads))
        
        return (energy_weight * norm_energy + 
                latency_weight * norm_latency + 
                load_weight * load_balancing)
    
    def _update_network_conditions(self):
        """Update network conditions with some random fluctuation"""
        fluctuation = np.random.uniform(-0.1, 0.1, self.num_edge_servers)
        self.network_conditions += fluctuation
        self.network_conditions = np.clip(self.network_conditions, 0.1, 1.0)


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
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01  # Increased minimum epsilon
        self.epsilon_decay = 0.999  # Slower decay rate
        self.learning_rate = 0.001  # Reduced learning rate
        self.batch_size = 128  # Increased batch size
        self.min_replay_size = 1000  # Minimum replay buffer size before training
        
        # Create Q-Networks (current and target)
        self.q_network = DQNNetwork(state_size, action_size, self.device)
        self.target_network = DQNNetwork(state_size, action_size, self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), 
                                  lr=self.learning_rate,
                                  weight_decay=1e-5)  # Added L2 regularization
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(100000)
        
    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
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
            return 0  # Return loss for monitoring
        
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
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update epsilon with slower decay
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
        return loss.item()  # Return loss for monitoring
        
    def update_target_network(self):
        """Update target network parameters"""
        self.target_network.load_state_dict(self.q_network.state_dict())

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


class DQNNetwork(nn.Module):
    """Deep Q-Network with extended architecture"""
    def __init__(self, state_size, action_size, device):
        super(DQNNetwork, self).__init__()
        
        self.device = device
        
        # Network layers
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_size)
        
        # Layer normalization instead of batch normalization
        self.ln1 = nn.LayerNorm(256)
        self.ln2 = nn.LayerNorm(256)
        self.ln3 = nn.LayerNorm(128)
        
        self.dropout = nn.Dropout(0.2)
        
        # Move the network to the specified device
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
    """Main training loop for MEC environment"""
    # Environment setup
    env = MECEnvironment(num_edge_servers=5)
    
    # Calculate state size
    state_size = (env.task_types + 
                 env.num_edge_servers + 
                 env.num_edge_servers)
    action_size = env.action_space.n
    
    # Initialize DQN agent
    agent = DQNAgent(state_size, action_size)
    
    # Training parameters
    num_episodes = 3000
    max_steps = 200  # Increased max steps
    target_update_frequency = 20  # Less frequent target updates
    eval_frequency = 10
    
    # Metrics tracking
    rewards_history = []
    losses_history = []
    epsilons_history = []
    avg_rewards_history = []
    
    for episode in range(num_episodes):
        state = env.reset()
        state = np.concatenate([
            state['task_queue'],
            state['server_states'],
            state['network_conditions']
        ])
        
        episode_reward = 0
        episode_losses = []
        
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
            loss = agent.train()
            if loss is not None:
                episode_losses.append(loss)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Update target network
        if episode % target_update_frequency == 0:
            agent.update_target_network()
        
        # Store metrics
        rewards_history.append(episode_reward)
        if episode_losses:
            losses_history.append(np.mean(episode_losses))
        epsilons_history.append(agent.epsilon)
        
        # Calculate running average
        if episode >= eval_frequency:
            avg_reward = np.mean(rewards_history[-eval_frequency:])
            avg_rewards_history.append(avg_reward)
        
        # Print progress
        if episode % eval_frequency == 0:
            avg_reward = np.mean(rewards_history[-eval_frequency:])
            avg_loss = np.mean(losses_history[-eval_frequency:]) if losses_history else 0
            print(f"Episode {episode}/{num_episodes}, "
                  f"Average Reward: {avg_reward:.2f}, "
                  f"Average Loss: {avg_loss:.4f}, "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    return agent, {
        'rewards': rewards_history,
        'losses': losses_history,
        'epsilons': epsilons_history,
        'avg_rewards': avg_rewards_history
    }



def plot_training_results(metrics):
    """Plot training metrics including rewards, losses, and epsilon"""
    plt.figure(figsize=(15, 10))
    
    # Plot rewards
    plt.subplot(3, 1, 1)
    plt.plot(metrics['rewards'], alpha=0.6, label='Episode Reward')
    
    # Calculate and plot moving average with correct dimensions
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
    plt.show()

if __name__ == "__main__":
    agent, metrics = train_mec_dqn()
    plot_training_results(metrics)