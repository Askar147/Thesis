import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import gym
from gym import spaces
from collections import deque
import random
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

class MECEnvironment(gym.Env):
    """MEC Environment with continuous action space for SAC"""
    def __init__(self, num_edge_servers=3):
        super().__init__()
        
        self.num_edge_servers = num_edge_servers
        
        # Action space: Continuous values between 0 and 1 for each server
        # Will be interpreted as server selection weights
        self.action_space = spaces.Box(
            low=0, high=1, shape=(num_edge_servers,), dtype=np.float32
        )
        
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
        self.current_task_size = np.random.uniform(0.2, 1.0)
        return self._get_observation()
    
    def step(self, action):
        """Execute action and return new state"""
        # Convert continuous action to server selection
        # Using softmax to get probabilities
        action_probs = F.softmax(torch.FloatTensor(action), dim=0).numpy()
        selected_server = np.argmax(action_probs)
        
        # Calculate latency for the chosen server
        latency = self._calculate_latency(selected_server)
        
        # Normalize reward between -1 and 0
        reward = -latency / 100.0
        
        # Generate new task for next state
        self.current_task_size = np.random.uniform(0.2, 1.0)
        
        # Get new observation
        observation = self._get_observation()
        
        # Episode never ends
        done = False
        
        return observation, reward, done, {}
    
    def _get_observation(self):
        return {
            'task_size': np.array([self.current_task_size], dtype=np.float32),
            'server_speeds': self.server_speeds.astype(np.float32)
        }
    
    def _calculate_latency(self, server_idx):
        """Calculate latency for the offloading decision"""
        processing_time = self.current_task_size / self.server_speeds[server_idx]
        return processing_time


class ReplayBuffer:
    """Experience Replay Buffer for SAC"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of experiences"""
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        
        # Convert to tensors for batch processing
        return (
            torch.FloatTensor(np.array(state)),
            torch.FloatTensor(np.array(action)),
            torch.FloatTensor(np.array(reward)).unsqueeze(1),
            torch.FloatTensor(np.array(next_state)),
            torch.FloatTensor(np.array(done)).unsqueeze(1)
        )
    
    def __len__(self):
        """Return current size of buffer"""
        return len(self.buffer)

class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        
        self.log_std_min = -20
        self.log_std_max = 2
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        
        # No tanh here since we want the output to be positive (for server selection)
        action = F.softplus(x_t)  # Ensures positive outputs
        
        log_prob = normal.log_prob(x_t)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob, mean

class SoftQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(SoftQNetwork, self).__init__()
        
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class SACAgent:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.policy = GaussianPolicy(state_dim, action_dim).to(self.device)
        self.q1 = SoftQNetwork(state_dim, action_dim).to(self.device)
        self.q2 = SoftQNetwork(state_dim, action_dim).to(self.device)
        self.target_q1 = SoftQNetwork(state_dim, action_dim).to(self.device)
        self.target_q2 = SoftQNetwork(state_dim, action_dim).to(self.device)
        
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=3e-4)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=3e-4)
        
        self.target_entropy = -action_dim/2
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
        
        self.replay_buffer = ReplayBuffer(100000)
        
        self.gamma = 0.99
        self.tau = 0.001  # Slower target network updates
        self.batch_size = 256  # Larger batch size for more stable updates
        self.min_training_size = 5000  # Wait for more samples before training
        
    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if evaluate:
                _, _, action = self.policy.sample(state)
            else:
                action, _, _ = self.policy.sample(state)
        
        return action.cpu().numpy()[0]
    
    def train(self):
        """Train the agent using SAC"""
        # Don't train until we have enough samples
        if len(self.replay_buffer) < self.min_training_size:
            return None, None, None, None
        # Sample from replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = \
            self.replay_buffer.sample(self.batch_size)
        
        # Move to device
        state_batch = state_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        done_batch = done_batch.to(self.device)
        
        alpha = self.log_alpha.exp()
        
        # Update Q-functions
        with torch.no_grad():
            next_action, next_log_pi, _ = self.policy.sample(next_state_batch)
            target_q1 = self.target_q1(next_state_batch, next_action)
            target_q2 = self.target_q2(next_state_batch, next_action)
            target_q = torch.min(target_q1, target_q2) - alpha * next_log_pi
            target_q = reward_batch + (1 - done_batch) * self.gamma * target_q
        
        # Q1 update
        current_q1 = self.q1(state_batch, action_batch)
        q1_loss = F.mse_loss(current_q1, target_q)
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        # Q2 update
        current_q2 = self.q2(state_batch, action_batch)
        q2_loss = F.mse_loss(current_q2, target_q)
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # Policy update
        new_actions, log_pi, _ = self.policy.sample(state_batch)
        q1_new = self.q1(state_batch, new_actions)
        q2_new = self.q2(state_batch, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        policy_loss = (alpha * log_pi - q_new).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update temperature
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Soft update target networks
        self._soft_update_target_network(self.q1, self.target_q1)
        self._soft_update_target_network(self.q2, self.target_q2)
        
        return q1_loss.item(), q2_loss.item(), policy_loss.item(), alpha_loss.item()
    
    def _soft_update_target_network(self, source, target):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + source_param.data * self.tau
            )

def train_mec_sac():
    """Training loop for SAC"""
    env = MECEnvironment(num_edge_servers=3)
    
    state_size = 1 + env.num_edge_servers  # task_size + server_speeds
    action_size = env.num_edge_servers
    
    agent = SACAgent(state_size, action_size)
    
    num_episodes = 3000
    max_steps = 100
    eval_frequency = 50
    
    metrics = {
        'rewards': [],
        'q1_losses': [],
        'q2_losses': [],
        'policy_losses': [],
        'alpha_losses': [],
        'alphas': [],
        'avg_rewards': []
    }
    
    for episode in range(num_episodes):
        state = env.reset()
        state = np.concatenate([
            state['task_size'],
            state['server_speeds']
        ])
        
        episode_reward = 0
        episode_losses = []
        
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            next_state = np.concatenate([
                next_state['task_size'],
                next_state['server_speeds']
            ])
            
            agent.replay_buffer.push(state, action, reward, next_state, done)
            losses = agent.train()
            
            if losses is not None:
                q1_loss, q2_loss, policy_loss, alpha_loss = losses
                if all(x is not None for x in [q1_loss, q2_loss, policy_loss, alpha_loss]):
                    episode_losses.append({
                        'q1': q1_loss,
                        'q2': q2_loss,
                        'policy': policy_loss,
                        'alpha': alpha_loss
                    })
            
            state = next_state
            episode_reward += reward
        
        # Store metrics
        metrics['rewards'].append(episode_reward)
        metrics['alphas'].append(agent.log_alpha.exp().item())
        
        # Calculate average losses for the episode if we have any valid losses
        if episode_losses:
            # Filter out any potential None values
            q1_losses = [l['q1'] for l in episode_losses if l['q1'] is not None]
            q2_losses = [l['q2'] for l in episode_losses if l['q2'] is not None]
            policy_losses = [l['policy'] for l in episode_losses if l['policy'] is not None]
            alpha_losses = [l['alpha'] for l in episode_losses if l['alpha'] is not None]
            
            # Only compute means if we have valid losses
            metrics['q1_losses'].append(np.mean(q1_losses) if q1_losses else 0)
            metrics['q2_losses'].append(np.mean(q2_losses) if q2_losses else 0)
            metrics['policy_losses'].append(np.mean(policy_losses) if policy_losses else 0)
            metrics['alpha_losses'].append(np.mean(alpha_losses) if alpha_losses else 0)
        else:
            # If no losses yet, append zeros
            metrics['q1_losses'].append(0)
            metrics['q2_losses'].append(0)
            metrics['policy_losses'].append(0)
            metrics['alpha_losses'].append(0)
            metrics['alphas'].append(agent.log_alpha.exp().item())
        
        # Calculate running average
        if episode >= eval_frequency:
            avg_reward = np.mean(metrics['rewards'][-eval_frequency:])
            metrics['avg_rewards'].append(avg_reward)
        
        # Print progress
        if episode % eval_frequency == 0:
            avg_reward = np.mean(metrics['rewards'][-eval_frequency:])
            print(f"Episode {episode}/{num_episodes}, "
                  f"Average Reward: {avg_reward:.2f}, "
                  f"Alpha: {agent.log_alpha.exp().item():.3f}")
    
    return agent, metrics

def plot_sac_results(metrics):
    """Plot training metrics for SAC"""
    plt.figure(figsize=(15, 12))
    
    # Plot rewards
    plt.subplot(4, 1, 1)
    plt.plot(metrics['rewards'], alpha=0.6, label='Episode Reward')
    
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
    plt.subplot(4, 1, 2)
    plt.plot(metrics['q1_losses'], label='Q1 Loss')
    plt.plot(metrics['q2_losses'], label='Q2 Loss')
    plt.plot(metrics['policy_losses'], label='Policy Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Losses')
    plt.grid(True)
    
    # Plot alpha
    plt.subplot(4, 1, 3)
    plt.plot(metrics['alphas'], label='Alpha')
    plt.xlabel('Episode')
    plt.ylabel('Alpha')
    plt.legend()
    plt.title('Temperature Parameter')
    plt.grid(True)
    
    # Plot alpha losses
    plt.subplot(4, 1, 4)
    plt.plot(metrics['alpha_losses'], label='Alpha Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Temperature Loss')
    plt.grid(True)
    
    plt.tight_layout()

if __name__ == "__main__":
    # Create directory for this specific run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("results", f"sac_run_{timestamp}")
    
    # Train the agent
    agent, metrics = train_mec_sac()
    
    # Save results
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    
    # Save metrics to JSON file
    metrics_filename = os.path.join(run_dir, f"sac_metrics_{timestamp}.json")
    json_metrics = {
        'rewards': [float(r) for r in metrics['rewards']],
        'q1_losses': [float(l) for l in metrics['q1_losses']],
        'q2_losses': [float(l) for l in metrics['q2_losses']],
        'policy_losses': [float(l) for l in metrics['policy_losses']],
        'alpha_losses': [float(l) for l in metrics['alpha_losses']],
        'alphas': [float(a) for a in metrics['alphas']],
        'avg_rewards': [float(ar) for ar in metrics.get('avg_rewards', [])]
    }
    with open(metrics_filename, 'w') as f:
        json.dump(json_metrics, f, indent=4)
    
    # Create and save plot
    plt.figure(figsize=(15, 12))
    plot_sac_results(metrics)
    plot_filename = os.path.join(run_dir, f"sac_training_plot_{timestamp}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    
    print(f"Results saved to directory: {run_dir}")
    print(f"Metrics saved as: sac_metrics_{timestamp}.json")
    print(f"Plot saved as: sac_training_plot_{timestamp}.png")
    
    # Also display the plot
    plt.show()