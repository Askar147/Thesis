import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import gym
from collections import deque
import random

class GaussianPolicy(nn.Module):
    """SAC Policy Network (Actor)"""
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(GaussianPolicy, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        
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
        
        # Reparameterization trick
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        
        # Calculate log probability of action
        log_prob = normal.log_prob(x_t)
        
        # Enforce action bounds
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob, torch.tanh(mean)

class SoftQNetwork(nn.Module):
    """Soft Q-Network"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
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
            torch.FloatTensor(state),
            torch.FloatTensor(action),
            torch.FloatTensor(reward).unsqueeze(1),
            torch.FloatTensor(next_state),
            torch.FloatTensor(done).unsqueeze(1)
        )
    
    def __len__(self):
        return len(self.buffer)

class SACAgent:
    """SAC Agent for VEC task offloading"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dim).to(self.device)
        self.q1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_q1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_q2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        
        # Copy parameters to target networks
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        
        # Initialize optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=3e-4)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=3e-4)
        
        # Initialize automatic entropy tuning
        self.target_entropy = -action_dim  # Heuristic value
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(1000000)
        
        # SAC hyperparameters
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 256
        self.min_action = -1.0
        self.max_action = 1.0
        
    def select_action(self, state, evaluate=False):
        """Select action using the policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if evaluate:
                _, _, action = self.policy.sample(state)
            else:
                action, _, _ = self.policy.sample(state)
        
        return action.cpu().numpy()[0]
    
    def update(self):
        """Update networks using SAC algorithm"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
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
        """Soft update of target network parameters"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + source_param.data * self.tau
            )

def train_sac():
    """Main training loop"""
    env = VECEnvironment()  # Your custom environment
    
    # Calculate state and action dimensions
    state_dim = (env.observation_space['vehicle_states'].shape[0] + 
                env.observation_space['server_states'].shape[0])
    action_dim = env.action_space.shape[0]  # Assuming continuous action space
    
    # Initialize agent
    agent = SACAgent(state_dim, action_dim)
    
    # Training parameters
    num_episodes = 1000
    max_steps_per_episode = 1000
    updates_per_step = 1
    
    for episode in range(num_episodes):
        state = env.reset()
        state = np.concatenate([state['vehicle_states'], state['server_states']])
        episode_reward = 0
        
        for step in range(max_steps_per_episode):
            # Select action
            action = agent.select_action(state)
            
            # Take action in environment
            next_state, reward, done, _ = env.step(action)
            next_state = np.concatenate([
                next_state['vehicle_states'],
                next_state['server_states']
            ])
            
            # Store experience
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # Update networks
            for _ in range(updates_per_step):
                q1_loss, q2_loss, policy_loss, alpha_loss = agent.update()
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Logging
        print(f"Episode {episode + 1}")
        print(f"Total Reward: {episode_reward}")
        if step > 0:  # Only print losses if we did at least one update
            print(f"Q1 Loss: {q1_loss:.3f}")
            print(f"Q2 Loss: {q2_loss:.3f}")
            print(f"Policy Loss: {policy_loss:.3f}")
            print(f"Alpha Loss: {alpha_loss:.3f}")
            print(f"Temperature: {agent.log_alpha.exp().item():.3f}")
        print("------------------------")

if __name__ == "__main__":
    train_sac()