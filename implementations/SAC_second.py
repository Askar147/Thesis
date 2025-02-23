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
    """Enhanced MEC Environment with realistic latency components for SAC"""
    def __init__(self, num_edge_servers=10):
        super().__init__()
        
        self.num_edge_servers = num_edge_servers
        
        # Action space: Continuous values for server selection probabilities
        self.action_space = spaces.Box(
            low=0, high=1, shape=(num_edge_servers,), dtype=np.float32
        )
        
        # Observation space with all components
        self.observation_space = spaces.Dict({
            'task_size': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'server_speeds': spaces.Box(low=np.zeros(num_edge_servers), high=np.ones(num_edge_servers), dtype=np.float32),
            'server_loads': spaces.Box(low=np.zeros(num_edge_servers), high=np.ones(num_edge_servers), dtype=np.float32),
            'network_conditions': spaces.Box(low=np.zeros(num_edge_servers), high=np.ones(num_edge_servers), dtype=np.float32),
            'server_distances': spaces.Box(low=np.zeros(num_edge_servers), high=np.ones(num_edge_servers), dtype=np.float32)
        })
        
        # Initialize server characteristics
        self.server_speeds = np.random.uniform(0.7, 1.0, num_edge_servers)
        self.server_distances = np.random.uniform(0.1, 0.8, num_edge_servers)
        self.bandwidth_up = np.random.uniform(0.6, 1.0, num_edge_servers)
        self.bandwidth_down = np.random.uniform(0.7, 1.0, num_edge_servers)
        
        # Scaling factors
        self.uplink_scale = 0.8
        self.prop_scale = 0.05
        self.downlink_scale = 0.8
        self.queue_factor = 1.2
        self.decay_factor = 0.95
        
        self.reset()
    
    def reset(self):
        """Reset environment state"""
        self.current_task_size = np.random.uniform(0.2, 0.8)
        self.server_loads = np.random.uniform(0.1, 0.3, self.num_edge_servers)
        self.network_conditions = np.random.uniform(0.8, 1.0, self.num_edge_servers)
        return self._get_observation()
    
    def _calculate_total_latency(self, server_idx, action_weight):
        """Calculate total latency with all components"""
        # Uplink transmission delay
        uplink_delay = (self.current_task_size / self.bandwidth_up[server_idx]) * \
                      (1 / self.network_conditions[server_idx]) * self.uplink_scale
        
        # Propagation delay
        prop_delay = self.server_distances[server_idx] * self.prop_scale
        
        # Processing delay
        effective_speed = self.server_speeds[server_idx] * (1 - self.server_loads[server_idx])
        processing_delay = self.current_task_size / max(effective_speed, 0.1)
        
        # Downlink transmission delay
        result_size = self.current_task_size * 0.05
        downlink_delay = (result_size / self.bandwidth_down[server_idx]) * \
                        (1 / self.network_conditions[server_idx]) * self.downlink_scale
        
        # Queuing delay
        queue_delay = self.server_loads[server_idx] * processing_delay * self.queue_factor
        
        # Total delay scaled by action weight
        total_delay = (uplink_delay + prop_delay + processing_delay + downlink_delay + queue_delay) / max(action_weight, 0.1)
        return total_delay
    
    def step(self, action):
        """Execute action and return new state"""
        # Convert action to probabilities
        action_probs = F.softmax(torch.FloatTensor(action), dim=0).numpy()
        selected_server = np.argmax(action_probs)
        
        # Calculate latency
        total_latency = self._calculate_total_latency(selected_server, action_probs[selected_server])
        
        # Calculate reward
        base_reward = -np.tanh(total_latency / 5.0)
        
        # Add performance bonuses
        available_speeds = self.server_speeds * (1 - self.server_loads)
        if selected_server == np.argmax(available_speeds):
            base_reward += 0.3
        elif available_speeds[selected_server] >= np.percentile(available_speeds, 75):
            base_reward += 0.1
        
        # Apply load penalty
        if self.server_loads[selected_server] > 0.8:
            base_reward -= 0.2
        
        # Update environment
        self._update_server_loads(selected_server, action_probs[selected_server])
        self._update_network_conditions()
        self.current_task_size = np.random.uniform(0.2, 0.8)
        
        return self._get_observation(), base_reward, False, {
            'selected_server': selected_server,
            'server_load': self.server_loads[selected_server],
            'network_quality': self.network_conditions[selected_server],
            'total_latency': total_latency,
            'effective_speed': available_speeds[selected_server],
            'action_prob': action_probs[selected_server]
        }
    
    def _update_server_loads(self, selected_server, action_weight):
        """Update server loads with action weight consideration"""
        load_increase = self.current_task_size * 0.1 * action_weight
        self.server_loads[selected_server] = min(
            self.server_loads[selected_server] + load_increase,
            1.0
        )
        
        for i in range(self.num_edge_servers):
            if i != selected_server:
                self.server_loads[i] = max(self.server_loads[i] * self.decay_factor, 0.1)
    
    def _update_network_conditions(self):
        """Update network conditions"""
        fluctuation = np.random.uniform(-0.1, 0.1, self.num_edge_servers)
        self.network_conditions += fluctuation
        self.network_conditions = np.clip(self.network_conditions, 0.3, 1.0)
    
    def _get_observation(self):
        """Get current observation"""
        return {
            'task_size': np.array([self.current_task_size], dtype=np.float32),
            'server_speeds': self.server_speeds.astype(np.float32),
            'server_loads': self.server_loads.astype(np.float32),
            'network_conditions': self.network_conditions.astype(np.float32),
            'server_distances': self.server_distances.astype(np.float32)
        }

class GaussianPolicy(nn.Module):
    """Gaussian Policy Network for SAC"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
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
        
        # Use softplus to ensure positive outputs
        action = F.softplus(x_t)
        
        log_prob = normal.log_prob(x_t)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob, mean

class SoftQNetwork(nn.Module):
    """Soft Q-Network for SAC"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(SoftQNetwork, self).__init__()
        
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        
        # Layer normalization for better stability
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.ln1(self.linear1(x)))
        x = F.relu(self.ln2(self.linear2(x)))
        x = self.linear3(x)
        return x

class ReplayBuffer:
    """Experience Replay Buffer"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return (
            torch.FloatTensor(np.array(state)),
            torch.FloatTensor(np.array(action)),
            torch.FloatTensor(np.array(reward)).unsqueeze(1),
            torch.FloatTensor(np.array(next_state)),
            torch.FloatTensor(np.array(done)).unsqueeze(1)
        )
    
    def __len__(self):
        return len(self.buffer)
    
class SACAgent:
    """SAC Agent for MEC task offloading with improved entropy management"""
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize dimensions and hyperparameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = 256
        self.gamma = 0.99  # discount factor
        self.tau = 0.005   # target network update rate
        self.batch_size = 256
        self.min_replay_size = 10000
        
        # Initialize networks
        self.policy = GaussianPolicy(state_dim, action_dim, self.hidden_dim).to(self.device)
        self.q1 = SoftQNetwork(state_dim, action_dim, self.hidden_dim).to(self.device)
        self.q2 = SoftQNetwork(state_dim, action_dim, self.hidden_dim).to(self.device)
        self.target_q1 = SoftQNetwork(state_dim, action_dim, self.hidden_dim).to(self.device)
        self.target_q2 = SoftQNetwork(state_dim, action_dim, self.hidden_dim).to(self.device)
        
        # Initialize target networks
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        
        # Initialize optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=3e-4)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=3e-4)
        
        # Initialize automatic entropy tuning with better defaults
        self.target_entropy = -action_dim/2  # Less negative target entropy
        self.log_alpha = torch.log(torch.ones(1) * 1.0).to(self.device).requires_grad_(True)  # Start with alpha = 1.0
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=1e-4)  # Lower learning rate for alpha
        
        # Add minimum alpha threshold
        self.min_alpha = 0.01  # Prevent alpha from going too close to zero
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(1000000)
        
        # Initialize tracking variables
        self.total_steps = 0
        self.current_alpha = 1.0
    
    def select_action(self, state, evaluate=False):
        """Select action using the policy network"""
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if evaluate:
                _, _, action = self.policy.sample(state)
            else:
                action, _, _ = self.policy.sample(state)
            return action.cpu().numpy()[0]
    
    def train(self):
        """Training step for the agent"""
        if len(self.replay_buffer) < self.min_replay_size:
            return None
        
        # Sample batch from replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = \
            self.replay_buffer.sample(self.batch_size)
        
        # Move to device
        state_batch = state_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        done_batch = done_batch.to(self.device)
        
        # Get current alpha value
        alpha = self.log_alpha.exp()
        
        # Update Q-networks
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
        
        # Alpha update with minimum threshold and gradient clipping
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        torch.nn.utils.clip_grad_norm_([self.log_alpha], 0.5)  # Add gradient clipping
        self.alpha_optimizer.step()
        
        # Apply minimum alpha threshold
        with torch.no_grad():
            self.log_alpha.data = torch.max(
                self.log_alpha.data,
                torch.log(torch.tensor(self.min_alpha)).to(self.device)
            )
        
        self.current_alpha = self.log_alpha.exp().item()
        
        # Update target networks
        self._soft_update_target_network(self.q1, self.target_q1)
        self._soft_update_target_network(self.q2, self.target_q2)
        
        return q1_loss.item(), q2_loss.item(), policy_loss.item(), alpha_loss.item()
    
    def _soft_update_target_network(self, source, target):
        """Soft update of target network from source network"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + source_param.data * self.tau
            )

def train_mec_sac():
    """Training loop for SAC with enhanced MEC environment"""
    # Create environment
    env = MECEnvironment(num_edge_servers=10)
    
    # Calculate state size from all observation components
    state_size = (1 +  # task_size
                env.num_edge_servers +  # server_speeds
                env.num_edge_servers +  # server_loads
                env.num_edge_servers +  # network_conditions
                env.num_edge_servers)   # server_distances
    action_size = env.num_edge_servers
    
    print(f"State size: {state_size}, Action size: {action_size}")
    
    # Initialize agent
    agent = SACAgent(state_size, action_size)
    
    # Training parameters
    num_episodes = 2000
    max_steps = 100
    eval_frequency = 50
    
    # Initialize metrics tracking
    metrics = {
        'rewards': [],
        'q1_losses': [],
        'q2_losses': [],
        'policy_losses': [],
        'alpha_losses': [],
        'alphas': [],
        'latencies': [],
        'server_loads': [],
        'avg_rewards': []
    }
    
    # Training loop
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
        episode_latencies = []
        
        for step in range(max_steps):
            # Select and perform action
            action = agent.select_action(state)
            next_obs, reward, done, info = env.step(action)
            
            # Process next state
            next_state = np.concatenate([
                next_obs['task_size'],
                next_obs['server_speeds'],
                next_obs['server_loads'],
                next_obs['network_conditions'],
                next_obs['server_distances']
            ])
            
            # Store transition and train
            agent.replay_buffer.push(state, action, reward, next_state, done)
            losses = agent.train()
            
            if losses is not None:
                episode_losses.append(losses)
            
            episode_latencies.append(info['total_latency'])
            state = next_state
            episode_reward += reward
        
        # Update metrics
        metrics['rewards'].append(episode_reward)
        metrics['latencies'].append(np.mean(episode_latencies))
        metrics['server_loads'].append(np.mean(next_obs['server_loads']))
        metrics['alphas'].append(agent.current_alpha)
        
        if episode_losses:
            avg_losses = np.mean(episode_losses, axis=0)
            metrics['q1_losses'].append(avg_losses[0])
            metrics['q2_losses'].append(avg_losses[1])
            metrics['policy_losses'].append(avg_losses[2])
            metrics['alpha_losses'].append(avg_losses[3])
        
        # Calculate running average
        if episode >= eval_frequency:
            avg_reward = np.mean(metrics['rewards'][-eval_frequency:])
            metrics['avg_rewards'].append(avg_reward)
        
        # Print progress
        if episode % eval_frequency == 0:
            avg_reward = np.mean(metrics['rewards'][-eval_frequency:])
            avg_latency = np.mean(metrics['latencies'][-eval_frequency:])
            print(f"Episode {episode}/{num_episodes}, "
                  f"Average Reward: {avg_reward:.2f}, "
                  f"Average Latency: {avg_latency:.2f}, "
                  f"Alpha: {agent.current_alpha:.3f}")
    
    return agent, metrics

def plot_training_results(metrics):
    """Plot training metrics"""
    plt.figure(figsize=(15, 15))
    
    # Plot rewards
    plt.subplot(5, 1, 1)
    plt.plot(metrics['rewards'], alpha=0.6, label='Episode Reward')
    if len(metrics['avg_rewards']) > 0:
        plt.plot(range(50, len(metrics['rewards'])), 
                metrics['avg_rewards'], 'r-', 
                label='Moving Average')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.title('Training Rewards')
    plt.grid(True)
    
    # Plot latencies
    plt.subplot(5, 1, 2)
    plt.plot(metrics['latencies'], label='Average Latency')
    plt.xlabel('Episode')
    plt.ylabel('Latency')
    plt.legend()
    plt.title('Average Latencies')
    plt.grid(True)
    
    # Plot losses
    plt.subplot(5, 1, 3)
    plt.plot(metrics['q1_losses'], label='Q1 Loss')
    plt.plot(metrics['q2_losses'], label='Q2 Loss')
    plt.plot(metrics['policy_losses'], label='Policy Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Losses')
    plt.grid(True)
    
    # Plot alpha
    plt.subplot(5, 1, 4)
    plt.plot(metrics['alphas'], 'b-', label='Alpha')
    plt.plot(metrics['alpha_losses'], 'r--', label='Alpha Loss', alpha=0.6)
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Temperature Parameter (Alpha) and Loss')
    plt.grid(True)
    
    # Plot server loads
    plt.subplot(5, 1, 5)
    plt.plot(metrics['server_loads'], 'g-', label='Average Server Load')
    plt.axhline(y=0.8, color='r', linestyle='--', label='High Load Threshold')
    plt.xlabel('Episode')
    plt.ylabel('Load')
    plt.legend()
    plt.title('Average Server Loads')
    plt.grid(True)
    
    plt.tight_layout()
    return plt.gcf()

if __name__ == "__main__":
    # Create directory for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("results", f"sac_run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Train the agent
    print("Starting SAC training...")
    agent, metrics = train_mec_sac()
    
    # Save metrics
    metrics_filename = os.path.join(run_dir, f"sac_metrics_{timestamp}.json")
    json_metrics = {k: [float(v) for v in vals] for k, vals in metrics.items()}
    with open(metrics_filename, 'w') as f:
        json.dump(json_metrics, f, indent=4)
    
    # Create and save plot
    fig = plot_training_results(metrics)
    plot_filename = os.path.join(run_dir, f"sac_training_plot_{timestamp}.png")
    fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
    
    print(f"\nTraining completed!")
    print(f"Results saved to: {run_dir}")
    plt.show()