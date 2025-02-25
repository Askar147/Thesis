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
    """Encoder network for MEC VAE with residual connections and layer normalization"""
    def __init__(self, state_dim, hidden_dim=256, latent_dim=32):
        super(MECEncoder, self).__init__()
        
        # Layer 1
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)  # Use LayerNorm instead of BatchNorm
        
        # Layer 2 with residual connection
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)  # Use LayerNorm instead of BatchNorm
        
        # Layer 3 with residual connection
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)  # Use LayerNorm instead of BatchNorm
        
        # Mean and log variance layers
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        # Layer 1
        x1 = F.relu(self.ln1(self.fc1(x)))
        
        # Layer 2 with residual connection
        x2 = F.relu(self.ln2(self.fc2(x1)))
        x2 = x2 + x1  # Residual connection
        
        # Layer 3 with residual connection
        x3 = F.relu(self.ln3(self.fc3(x2)))
        x3 = x3 + x2  # Residual connection
        
        # Mean and log variance
        mu = self.fc_mu(x3)
        logvar = self.fc_logvar(x3)
        logvar = torch.clamp(logvar, -10, 2)  # Prevent numerical instability
        
        return mu, logvar

class MECDecoder(nn.Module):
    """Decoder network for MEC VAE with residual connections and layer normalization"""
    def __init__(self, latent_dim, state_dim, num_servers, hidden_dim=256):
        super(MECDecoder, self).__init__()
        
        self.num_servers = num_servers
        
        # Main decoder layers
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)  # Use LayerNorm instead of BatchNorm
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)  # Use LayerNorm instead of BatchNorm
        
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)  # Use LayerNorm instead of BatchNorm
        
        # Separate heads for different components
        self.task_head = nn.Linear(hidden_dim, 1)  # task size
        self.server_speeds_head = nn.Linear(hidden_dim, num_servers)
        self.server_loads_head = nn.Linear(hidden_dim, num_servers)
        self.network_conditions_head = nn.Linear(hidden_dim, num_servers)
        self.server_distances_head = nn.Linear(hidden_dim, num_servers)
    
    def forward(self, z):
        # Layer 1
        x1 = F.relu(self.ln1(self.fc1(z)))
        
        # Layer 2 with residual connection
        x2 = F.relu(self.ln2(self.fc2(x1)))
        x2 = x2 + x1  # Residual connection
        
        # Layer 3 with residual connection
        x3 = F.relu(self.ln3(self.fc3(x2)))
        x3 = x3 + x2  # Residual connection
        
        # Generate different components
        task_size = torch.sigmoid(self.task_head(x3))
        server_speeds = torch.sigmoid(self.server_speeds_head(x3))
        server_loads = torch.sigmoid(self.server_loads_head(x3))
        network_conditions = torch.sigmoid(self.network_conditions_head(x3))
        server_distances = torch.sigmoid(self.server_distances_head(x3))
        
        return {
            'task_size': task_size,
            'server_speeds': server_speeds,
            'server_loads': server_loads,
            'network_conditions': network_conditions,
            'server_distances': server_distances
        }

class MECVAE(nn.Module):
    """VAE for MEC system state modeling with improved architecture"""
    def __init__(self, state_dim, num_servers, hidden_dim=256, latent_dim=32):
        super(MECVAE, self).__init__()
        
        self.encoder = MECEncoder(state_dim, hidden_dim, latent_dim)
        self.decoder = MECDecoder(latent_dim, state_dim, num_servers, hidden_dim)
        
        self.latent_dim = latent_dim
        
    def reparameterize(self, mu, logvar):
        """Reparameterization trick with numerical stability improvements"""
        std = torch.exp(0.5 * logvar)
        # Clamp std for numerical stability
        std = torch.clamp(std, min=1e-6, max=10.0)
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

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer for VAE-based RL"""
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha  # determines how much prioritization is used
        self.beta = beta_start  # importance-sampling correction
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
    
    def push(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
        
        # Compute the sampling probabilities
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Update beta
        self.beta = min(1.0, self.beta + (1.0 - self.beta_start) / self.beta_frames * self.frame)
        self.frame += 1
        
        # Sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Compute importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = torch.FloatTensor(weights)
        
        return list(zip(*samples)), indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)

class DoubleQNetwork(nn.Module):
    """Dueling Double Q-Network for value-based policy"""
    def __init__(self, latent_dim, num_actions, hidden_dim=256):
        super(DoubleQNetwork, self).__init__()
        
        # Shared layers
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)  # Use LayerNorm instead of BatchNorm
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)  # Use LayerNorm instead of BatchNorm
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions)
        )
    
    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combine value and advantage to get Q-values using the dueling architecture
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values

class MECVAEAgent:
    """Improved VAE-based agent for MEC task offloading"""
    def __init__(self, state_dim, num_servers, hidden_dim=256, latent_dim=32,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.state_dim = state_dim
        self.num_servers = num_servers
        
        # Initialize VAE
        self.vae = MECVAE(state_dim, num_servers, hidden_dim, latent_dim).to(device)
        
        # Initialize policy networks
        self.policy_net = DoubleQNetwork(latent_dim, num_servers, hidden_dim).to(device)
        self.target_net = DoubleQNetwork(latent_dim, num_servers, hidden_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Initialize optimizers with learning rate scheduling
        self.vae_optimizer = optim.Adam(self.vae.parameters(), lr=0.001)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        
        # Learning rate schedulers
        self.vae_scheduler = optim.lr_scheduler.StepLR(self.vae_optimizer, step_size=200, gamma=0.5)
        self.policy_scheduler = optim.lr_scheduler.StepLR(self.policy_optimizer, step_size=200, gamma=0.5)
        
        # Initialize replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(100000)
        
        # Hyperparameters
        self.gamma = 0.99
        self.batch_size = 128  # Increased batch size
        self.min_replay_size = 1000
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.vae_train_frequency = 5  # Train VAE more frequently
        self.target_update_frequency = 10  # Update target network more frequently
        self.steps = 0
        
        # Beta parameter for VAE KL divergence weight
        self.vae_beta = 0.05  # Lower value for better reconstruction
        
        # Gradient clipping value
        self.clip_value = 1.0
    
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
        """Select action with epsilon-greedy policy and optional exploration noise"""
        if not evaluate and random.random() < self.epsilon:
            return random.randrange(self.num_servers)
        
        with torch.no_grad():
            # Set models to evaluation mode
            self.vae.eval()
            self.policy_net.eval()
            
            state_tensor = self.state_to_tensor(state).unsqueeze(0)
            mu, _ = self.vae.encoder(state_tensor)
            
            # Add some noise for exploration in latent space (only during training)
            if not evaluate and random.random() < 0.3:
                exploration_noise = torch.randn_like(mu) * 0.1
                mu = mu + exploration_noise
            
            q_values = self.policy_net(mu)
            
            # Set models back to training mode if not evaluating
            if not evaluate:
                self.vae.train()
                self.policy_net.train()
                
            return q_values.max(1)[1].item()
    
    def train_vae(self, states, weights=None):
        """Train VAE with optional importance weights"""
        # Set to training mode (just to be sure)
        self.vae.train()
        
        # Forward pass
        decoded, mu, logvar = self.vae(states)
        
        # Split the input states tensor into its components
        num_servers = self.num_servers
        task_target = states[:, :1]
        server_speeds_target = states[:, 1:1+num_servers]
        server_loads_target = states[:, 1+num_servers:1+2*num_servers]
        network_conditions_target = states[:, 1+2*num_servers:1+3*num_servers]
        server_distances_target = states[:, 1+3*num_servers:1+4*num_servers]
        
        # Individual reconstruction losses
        task_loss = F.mse_loss(decoded['task_size'], task_target, reduction='none').mean(1)
        speeds_loss = F.mse_loss(decoded['server_speeds'], server_speeds_target, reduction='none').mean(1)
        loads_loss = F.mse_loss(decoded['server_loads'], server_loads_target, reduction='none').mean(1)
        network_loss = F.mse_loss(decoded['network_conditions'], network_conditions_target, reduction='none').mean(1)
        distance_loss = F.mse_loss(decoded['server_distances'], server_distances_target, reduction='none').mean(1)
        
        # Calculate per-item reconstruction loss
        recon_loss_per_item = task_loss + speeds_loss + loads_loss + network_loss + distance_loss
        
        # Calculate KL divergence (per item)
        kl_loss_per_item = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        
        # Apply importance weights if provided
        if weights is not None:
            recon_loss = (recon_loss_per_item * weights).mean()
            kl_loss = (kl_loss_per_item * weights).mean()
        else:
            recon_loss = recon_loss_per_item.mean()
            kl_loss = kl_loss_per_item.mean()
        
        # Total loss (with adjustable beta coefficient)
        vae_loss = recon_loss + self.vae_beta * kl_loss
        
        # Optimize
        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.vae.parameters(), self.clip_value)
        self.vae_optimizer.step()
        
        return vae_loss.item(), recon_loss.item(), kl_loss.item()
    
    def train(self):
        """Train the agent using prioritized experience replay and double DQN"""
        if len(self.replay_buffer) < self.min_replay_size:
            return None
        
        # Ensure models are in training mode
        self.vae.train()
        self.policy_net.train()
        
        self.steps += 1
        
        # Sample from replay buffer with priorities
        batch, indices, weights = self.replay_buffer.sample(self.batch_size)
        weights = weights.to(self.device)
        
        # Process states
        state_batch = torch.stack([self.state_to_tensor(s) for s in batch[0]])
        action_batch = torch.tensor(batch[1], dtype=torch.long, device=self.device)
        reward_batch = torch.tensor(batch[2], dtype=torch.float32, device=self.device)
        next_state_batch = torch.stack([self.state_to_tensor(s) for s in batch[3]])
        done_batch = torch.tensor(batch[4], dtype=torch.float32, device=self.device)
        
        # Train VAE periodically
        if self.steps % self.vae_train_frequency == 0:
            vae_loss, recon_loss, kl_loss = self.train_vae(state_batch, weights)
        else:
            vae_loss, recon_loss, kl_loss = None, None, None
        
        # Get latent representations
        with torch.no_grad():
            state_mu, _ = self.vae.encoder(state_batch)
            next_state_mu, _ = self.vae.encoder(next_state_batch)
        
        # Get current Q values
        current_q = self.policy_net(state_mu)
        current_q_values = current_q.gather(1, action_batch.unsqueeze(1))
        
        # Get target Q values using Double Q-learning
        with torch.no_grad():
            # Get actions from policy network
            next_q_policy = self.policy_net(next_state_mu)
            next_actions = next_q_policy.max(1)[1].unsqueeze(1)
            
            # Get Q-values from target network for those actions
            next_q_target = self.target_net(next_state_mu)
            next_q_values = next_q_target.gather(1, next_actions)
            
            # Compute expected Q values
            expected_q_values = reward_batch.unsqueeze(1) + \
                              (1.0 - done_batch.unsqueeze(1)) * self.gamma * next_q_values
        
        # Compute td error for prioritized replay
        td_error = torch.abs(current_q_values - expected_q_values).detach().cpu().numpy()
        
        # Update priorities in replay buffer
        self.replay_buffer.update_priorities(indices, td_error.squeeze() + 1e-5)
        
        # Compute Huber loss with importance sampling weights
        loss = F.smooth_l1_loss(current_q_values, expected_q_values, reduction='none')
        weighted_loss = (loss * weights.unsqueeze(1)).mean()
        
        # Optimize policy
        self.policy_optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.clip_value)
        self.policy_optimizer.step()
        
        # Update target network periodically
        if self.steps % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Step the learning rate schedulers
        if self.steps % 1000 == 0:
            self.vae_scheduler.step()
            self.policy_scheduler.step()
        
        return weighted_loss.item(), vae_loss, recon_loss, kl_loss

def train_mec_vae():
    """Training loop for improved VAE in MEC environment"""
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
    
    # Make sure models start in training mode
    agent.vae.train()
    agent.policy_net.train()
    agent.target_net.eval()  # Target network is always in eval mode
    
    # Training parameters
    num_episodes = 2000
    max_steps = 100
    eval_frequency = 50
    
    # Early stopping parameters
    early_stop_patience = 10
    best_reward = -float('inf')
    no_improvement_count = 0
    
    # Metrics tracking
    metrics = {
        'rewards': [],
        'policy_losses': [],
        'vae_losses': [],
        'recon_losses': [],
        'kl_losses': [],
        'latencies': [],
        'server_loads': [],
        'avg_rewards': []
    }
    
    # Create evaluation function
    def evaluate_agent(eval_episodes=5):
        eval_rewards = []
        eval_latencies = []
        
        # Set models to evaluation mode during evaluation
        agent.vae.eval()
        agent.policy_net.eval()
        
        for _ in range(eval_episodes):
            state = env.reset()
            episode_reward = 0
            episode_latencies = []
            
            for step in range(max_steps):
                action = agent.select_action(state, evaluate=True)
                next_state, reward, done, info = env.step(action)
                episode_latencies.append(info['total_latency'])
                state = next_state
                episode_reward += reward
            
            eval_rewards.append(episode_reward)
            eval_latencies.append(np.mean(episode_latencies))
        
        # Set models back to training mode after evaluation
        agent.vae.train()
        agent.policy_net.train()
        
        return np.mean(eval_rewards), np.mean(eval_latencies)
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_policy_losses = []
        episode_vae_losses = []
        episode_recon_losses = []
        episode_kl_losses = []
        episode_latencies = []
        
        for step in range(max_steps):
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            # Store transition and train
            agent.replay_buffer.push(state, action, reward, next_state, done)
            losses = agent.train()
            
            if losses is not None:
                policy_loss, vae_loss, recon_loss, kl_loss = losses
                episode_policy_losses.append(policy_loss)
                if vae_loss is not None:
                    episode_vae_losses.append(vae_loss)
                    episode_recon_losses.append(recon_loss)
                    episode_kl_losses.append(kl_loss)
            
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
            metrics['recon_losses'].append(np.mean(episode_recon_losses))
            metrics['kl_losses'].append(np.mean(episode_kl_losses))
        
        # Calculate running average
        if episode >= eval_frequency:
            avg_reward = np.mean(metrics['rewards'][-eval_frequency:])
            metrics['avg_rewards'].append(avg_reward)
        
        # Update epsilon with a cyclical schedule to prevent getting stuck
        if episode < 1000:
            # Regular decay for the first 1000 episodes
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        else:
            # Cyclical epsilon for exploration after 1000 episodes
            cycle_length = 200
            cycle_position = (episode - 1000) % cycle_length
            if cycle_position == 0:
                # Reset epsilon to a higher value at the start of each cycle
                agent.epsilon = min(0.2, agent.epsilon * 2)
            else:
                # Decay within the cycle
                agent.epsilon = max(agent.epsilon_min, agent.epsilon * 0.98)
        
        # Print progress and evaluate
        if episode % eval_frequency == 0:
            # Regular evaluation
            avg_reward = np.mean(metrics['rewards'][-eval_frequency:])
            avg_latency = np.mean(metrics['latencies'][-eval_frequency:])
            
            # Run evaluation without exploration
            eval_reward, eval_latency = evaluate_agent()
            
            print(f"Episode {episode}/{num_episodes}, "
                  f"Average Reward: {avg_reward:.2f}, "
                  f"Average Latency: {avg_latency:.2f}, "
                  f"Eval Reward: {eval_reward:.2f}, "
                  f"Eval Latency: {eval_latency:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}, "
                  f"VAE LR: {agent.vae_scheduler.get_last_lr()[0]:.6f}")
            
            # Early stopping check
            if eval_reward > best_reward:
                best_reward = eval_reward
                no_improvement_count = 0
                # Save the best model
                torch.save({
                    'vae_state_dict': agent.vae.state_dict(),
                    'policy_state_dict': agent.policy_net.state_dict()
                }, "best_mec_vae.pth")
            else:
                no_improvement_count += 1
                
            if no_improvement_count >= early_stop_patience:
                print(f"Early stopping at episode {episode}")
                break
    
    # Load the best model for final evaluation
    checkpoint = torch.load("best_mec_vae.pth", weights_only=True)
    agent.vae.load_state_dict(checkpoint['vae_state_dict'])
    agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
    agent.target_net.load_state_dict(checkpoint['policy_state_dict'])
    
    # Final evaluation
    final_reward, final_latency = evaluate_agent(eval_episodes=50)
    print(f"Final evaluation - Reward: {final_reward:.2f}, Latency: {final_latency:.2f}")
    
    return agent, metrics

if __name__ == "__main__":
    agent, metrics = train_mec_vae()