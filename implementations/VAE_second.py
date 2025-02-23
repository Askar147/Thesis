import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from collections import deque
import random
from DQN_second import MECEnvironment

class MECEncoder(nn.Module):
    """Encoder network for MEC VAE"""
    def __init__(self, state_dim, hidden_dim=256, latent_dim=32):
        super(MECEncoder, self).__init__()
        
        # Encoder layers with layer normalization
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Mean and log variance layers
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class MECDecoder(nn.Module):
    """Decoder network for MEC VAE with separate heads"""
    def __init__(self, latent_dim, state_dim, num_servers, hidden_dim=256):
        super(MECDecoder, self).__init__()
        
        self.num_servers = num_servers
        
        # Main decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Separate heads for different components
        self.task_head = nn.Linear(hidden_dim, 1)  # task size
        self.server_speeds_head = nn.Linear(hidden_dim, num_servers)
        self.server_loads_head = nn.Linear(hidden_dim, num_servers)
        self.network_conditions_head = nn.Linear(hidden_dim, num_servers)
        self.server_distances_head = nn.Linear(hidden_dim, num_servers)
    
    def forward(self, z):
        x = self.decoder(z)
        
        # Generate different components
        task_size = torch.sigmoid(self.task_head(x))
        server_speeds = torch.sigmoid(self.server_speeds_head(x))
        server_loads = torch.sigmoid(self.server_loads_head(x))
        network_conditions = torch.sigmoid(self.network_conditions_head(x))
        server_distances = torch.sigmoid(self.server_distances_head(x))
        
        return {
            'task_size': task_size,
            'server_speeds': server_speeds,
            'server_loads': server_loads,
            'network_conditions': network_conditions,
            'server_distances': server_distances
        }

class MECVAE(nn.Module):
    """VAE for MEC system state modeling"""
    def __init__(self, state_dim, num_servers, hidden_dim=256, latent_dim=32):
        super(MECVAE, self).__init__()
        
        self.encoder = MECEncoder(state_dim, hidden_dim, latent_dim)
        self.decoder = MECDecoder(latent_dim, state_dim, num_servers, hidden_dim)
        
        self.latent_dim = latent_dim
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # Encode
        mu, logvar = self.encoder(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        decoded = self.decoder(z)
        
        return decoded, mu, logvar
    
    def generate(self, num_samples, device):
        """Generate new system states"""
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            return self.decoder(z)

class VAEReplayBuffer:
    """Replay buffer with VAE-based state representation"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class MECVAEAgent:
    """VAE-based agent for MEC task offloading"""
    def __init__(self, state_dim, num_servers, hidden_dim=256, latent_dim=32,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.state_dim = state_dim
        self.num_servers = num_servers
        
        # Initialize VAE
        self.vae = MECVAE(state_dim, num_servers, hidden_dim, latent_dim).to(device)
        
        # Initialize policy network
        self.policy_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_servers)
        ).to(device)
        
        # Initialize optimizers
        self.vae_optimizer = optim.Adam(self.vae.parameters(), lr=0.0003)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0003)
        
        # Initialize replay buffer
        self.replay_buffer = VAEReplayBuffer(100000)
        
        # Hyperparameters
        self.gamma = 0.99
        self.batch_size = 64
        self.min_replay_size = 1000
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.vae_train_frequency = 10  # Train VAE every n steps
        self.steps = 0
    
    def state_to_tensor(self, state_dict):
        """Convert state dictionary to tensor"""
        return torch.FloatTensor(np.concatenate([
            state_dict['task_size'],
            state_dict['server_speeds'],
            state_dict['server_loads'],
            state_dict['network_conditions'],
            state_dict['server_distances']
        ])).to(self.device)
    
    def select_action(self, state, evaluate=False):
        if not evaluate and random.random() < self.epsilon:
            return random.randrange(self.num_servers)
        
        with torch.no_grad():
            state_tensor = self.state_to_tensor(state).unsqueeze(0)
            mu, _ = self.vae.encoder(state_tensor)
            q_values = self.policy_net(mu)
            return q_values.max(1)[1].item()
    
    def train_vae(self, states):
        """Train VAE on a batch of states"""
        # Forward pass
        decoded, mu, logvar = self.vae(states)
        
        # Reconstruction loss for each component
        recon_loss = sum(
            F.mse_loss(decoded[key], states_target)
            for key, states_target in decoded.items()
        )
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        vae_loss = recon_loss + 0.1 * kl_loss
        
        # Optimize
        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()
        
        return vae_loss.item()
    
    def train(self):
        if len(self.replay_buffer) < self.min_replay_size:
            return None
        
        self.steps += 1
        
        # Sample from replay buffer
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = list(zip(*transitions))
        
        # Process states
        state_batch = torch.stack([self.state_to_tensor(s) for s in batch[0]])
        action_batch = torch.tensor(batch[1], device=self.device)
        reward_batch = torch.tensor(batch[2], device=self.device)
        next_state_batch = torch.stack([self.state_to_tensor(s) for s in batch[3]])
        done_batch = torch.tensor(batch[4], device=self.device)
        
        # Train VAE periodically
        if self.steps % self.vae_train_frequency == 0:
            vae_loss = self.train_vae(state_batch)
        else:
            vae_loss = None
        
        # Get latent representations
        with torch.no_grad():
            state_mu, _ = self.vae.encoder(state_batch)
            next_state_mu, _ = self.vae.encoder(next_state_batch)
        
        # Get current Q values
        current_q = self.policy_net(state_mu)
        current_q = current_q.gather(1, action_batch.unsqueeze(1))
        
        # Get target Q values
        with torch.no_grad():
            target_q = self.policy_net(next_state_mu)
            target_q = reward_batch.unsqueeze(1) + \
                      (1 - done_batch.unsqueeze(1)) * self.gamma * target_q.max(1)[0].unsqueeze(1)
        
        # Compute policy loss
        policy_loss = F.smooth_l1_loss(current_q, target_q)
        
        # Optimize policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.policy_optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return policy_loss.item(), vae_loss

def train_mec_vae():
    """Training loop for VAE in MEC environment"""
    # Initialize environment
    env = MECEnvironment(num_edge_servers=10)
    
    # Calculate state size
    state_size = (1 +  # task_size
                env.num_edge_servers +  # server_speeds
                env.num_edge_servers +  # server_loads
                env.num_edge_servers +  # network_conditions
                env.num_edge_servers)   # server_distances
    
    print(f"State size: {state_size}")
    
    # Initialize agent
    agent = MECVAEAgent(state_size, env.num_edge_servers)
    
    # Training parameters
    num_episodes = 1000
    max_steps = 100
    eval_frequency = 50
    
    # Metrics tracking
    metrics = {
        'rewards': [],
        'policy_losses': [],
        'vae_losses': [],
        'latencies': [],
        'server_loads': [],
        'avg_rewards': []
    }
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_policy_losses = []
        episode_vae_losses = []
        episode_latencies = []
        
        for step in range(max_steps):
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            # Store transition and train
            agent.replay_buffer.push(state, action, reward, next_state, done)
            losses = agent.train()
            
            if losses is not None:
                policy_loss, vae_loss = losses
                episode_policy_losses.append(policy_loss)
                if vae_loss is not None:
                    episode_vae_losses.append(vae_loss)
            
            episode_latencies.append(info['total_latency'])
            state = next_state
            episode_reward += reward
        
        # Update metrics
        metrics['rewards'].append(episode_reward)
        metrics['latencies'].append(np.mean(episode_latencies))
        metrics['server_loads'].append(np.mean(state['server_loads']))
        
        if episode_policy_losses:
            metrics['policy_losses'].append(np.mean(episode_policy_losses))
        if episode_vae_losses:
            metrics['vae_losses'].append(np.mean(episode_vae_losses))
        
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
                  f"Epsilon: {agent.epsilon:.3f}")
    
    return agent, metrics

if __name__ == "__main__":
    agent, metrics = train_mec_vae()