import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import gym
from gym import spaces
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

# Import the enhanced environment from DQN_second.py
from DQN_second import MECEnvironment

##############################################
# DDQN Network with Dueling Architecture
##############################################
class DDQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, device):
        super(DDQNNetwork, self).__init__()
        self.device = device

        # Shared feature layers
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)

        # Value stream
        self.value_fc = nn.Linear(256, 128)
        self.value = nn.Linear(128, 1)

        # Advantage stream
        self.adv_fc = nn.Linear(256, 128)
        self.advantage = nn.Linear(128, action_size)

        self.to(device)

    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        val = F.relu(self.value_fc(x))
        val = self.value(val)
        adv = F.relu(self.adv_fc(x))
        adv = self.advantage(adv)
        # Combine streams: Q = value + advantage - mean(advantage)
        q = val + adv - adv.mean(dim=1, keepdim=True)
        return q

##############################################
# DDQN Agent
##############################################
class DDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay_steps = 800
        self.current_episode = 0

        self.learning_rate = 0.0005
        self.batch_size = 64
        self.min_replay_size = 1000
        self.tau = 0.005  # Soft update rate

        self.online_network = DDQNNetwork(state_size, action_size, self.device)
        self.target_network = DDQNNetwork(state_size, action_size, self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())

        self.optimizer = optim.Adam(self.online_network.parameters(), lr=self.learning_rate)
        self.replay_buffer = ReplayBuffer(100000)

    def select_action(self, state):
        # Linear epsilon decay
        if self.current_episode < self.epsilon_decay_steps:
            self.epsilon = 1.0 - (1.0 - self.epsilon_min) * (self.current_episode / self.epsilon_decay_steps)
        else:
            self.epsilon = self.epsilon_min

        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.online_network(state_tensor)
            return q_values.argmax().item()

    def train(self):
        if len(self.replay_buffer) < self.min_replay_size:
            return 0

        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q = self.online_network(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_actions = self.online_network(next_states).argmax(1).unsqueeze(1)
            next_q = self.target_network(next_states).gather(1, next_actions)
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), 1.0)
        self.optimizer.step()

        # Soft update target network
        for target_param, online_param in zip(self.target_network.parameters(), self.online_network.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1 - self.tau) * target_param.data)

        return loss.item()

##############################################
# Replay Buffer
##############################################
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)

##############################################
# Training Loop
##############################################
def train_mec_ddqn():
    num_edge_servers = 10
    # Instantiate the enhanced environment from DQN_second.py
    env = MECEnvironment(num_edge_servers=num_edge_servers, continuous_action=False)

    # Flatten the observation:
    # Keys: 'task_size' (1), 'server_speeds' (num_edge_servers), 'server_loads' (num_edge_servers),
    # 'network_conditions' (num_edge_servers), 'server_distances' (num_edge_servers)
    state_size = 1 + 4 * num_edge_servers
    action_size = env.action_space.n

    print(f"State size: {state_size}, Action size: {action_size}")
    agent = DDQNAgent(state_size, action_size)

    num_episodes = 1000
    max_steps = 100
    metrics = {'rewards': [], 'losses': [], 'epsilons': []}

    for episode in range(num_episodes):
        obs = env.reset()
        state = np.concatenate([
            obs['task_size'],
            obs['server_speeds'],
            obs['server_loads'],
            obs['network_conditions'],
            obs['server_distances']
        ])
        episode_reward = 0
        episode_losses = []
        for step in range(max_steps):
            action = agent.select_action(state)
            next_obs, reward, done, info = env.step(action)
            next_state = np.concatenate([
                next_obs['task_size'],
                next_obs['server_speeds'],
                next_obs['server_loads'],
                next_obs['network_conditions'],
                next_obs['server_distances']
            ])
            agent.replay_buffer.push(state, action, reward, next_state, done)
            loss = agent.train()
            episode_losses.append(loss)
            state = next_state
            episode_reward += reward
        agent.current_episode = episode
        metrics['rewards'].append(episode_reward)
        metrics['losses'].append(np.mean(episode_losses))
        metrics['epsilons'].append(agent.epsilon)
        if episode % 50 == 0:
            avg_reward = np.mean(metrics['rewards'][-50:])
            avg_loss = np.mean(metrics['losses'][-50:])
            print(f"Episode {episode}/{num_episodes}, Average Reward: {avg_reward:.2f}, Average Loss: {avg_loss:.4f}, Epsilon: {agent.epsilon:.3f}")
    return agent, metrics

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
    run_dir = os.path.join("results", f"ddqn_run_{timestamp}")
    
    # Train the agent
    agent, metrics = train_mec_ddqn()
    
    # Save results
    save_training_results(metrics, save_dir=run_dir)
    
    # Also display the plot
    plt.show()
