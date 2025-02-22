import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch.distributions import Normal
import numpy as np

class TransformerComponent(nn.Module):
    """Transformer for temporal pattern recognition"""
    def __init__(self, input_dim, d_model=256, nhead=8, num_layers=3):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x, mask=None):
        x = self.input_projection(x)
        return self.transformer(x, mask)

class GNNComponent(nn.Module):
    """GNN for spatial relationships"""
    def __init__(self, node_features, hidden_dim=64):
        super().__init__()
        
        self.conv1 = GATConv(node_features, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        self.conv3 = GATConv(hidden_dim, hidden_dim)
        
    def forward(self, x, edge_index, edge_attr=None):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x

class VAEComponent(nn.Module):
    """VAE for learning state distributions"""
    def __init__(self, input_dim, hidden_dim=256, latent_dim=32):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class SACComponent(nn.Module):
    """SAC for decision making"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
        # Q networks
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        x = self.policy(state)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

class HybridVECSystem(nn.Module):
    """Hybrid system combining all components"""
    def __init__(self, 
                 input_dim,
                 action_dim,
                 node_features,
                 hidden_dim=256,
                 latent_dim=32):
        super().__init__()
        
        # Initialize components
        self.transformer = TransformerComponent(input_dim)
        self.gnn = GNNComponent(node_features)
        self.vae = VAEComponent(input_dim, hidden_dim, latent_dim)
        self.sac = SACComponent(input_dim, action_dim)
        
        # Feature fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output heads
        self.offload_head = nn.Linear(hidden_dim, action_dim)
        self.latency_head = nn.Linear(hidden_dim, 1)
        self.energy_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, temporal_data, spatial_data, edge_index):
        # Process temporal patterns
        temporal_features = self.transformer(temporal_data)
        
        # Process spatial relationships
        spatial_features = self.gnn(spatial_data, edge_index)
        
        # Learn state distribution
        vae_output, mu, logvar = self.vae(temporal_data)
        
        # Combine features
        combined_features = torch.cat([
            temporal_features.mean(1),
            spatial_features.mean(1),
            vae_output
        ], dim=1)
        
        # Fuse features
        fused_features = self.fusion_network(combined_features)
        
        # Get action distribution from SAC
        action_mean, action_log_std = self.sac(fused_features)
        
        # Get predictions
        offload_decision = self.offload_head(fused_features)
        latency = self.latency_head(fused_features)
        energy = self.energy_head(fused_features)
        
        return {
            'offload_decision': offload_decision,
            'latency': latency,
            'energy': energy,
            'action_dist': (action_mean, action_log_std),
            'vae_params': (mu, logvar)
        }

class HybridLoss(nn.Module):
    """Combined loss function for hybrid system"""
    def __init__(self, kld_weight=0.1, action_weight=1.0):
        super().__init__()
        self.kld_weight = kld_weight
        self.action_weight = action_weight
        
    def forward(self, outputs, targets):
        # Reconstruction loss from VAE
        vae_mu, vae_logvar = outputs['vae_params']
        kld_loss = -0.5 * torch.sum(1 + vae_logvar - vae_mu.pow(2) - vae_logvar.exp())
        
        # Task-specific losses
        offload_loss = F.mse_loss(outputs['offload_decision'], targets['offload_target'])
        latency_loss = outputs['latency'].mean()  # Minimize latency
        energy_loss = outputs['energy'].mean()    # Minimize energy
        
        # Action loss from SAC
        action_mean, action_log_std = outputs['action_dist']
        action_loss = self.compute_action_loss(action_mean, action_log_std, targets['action_target'])
        
        # Combine losses
        total_loss = (
            offload_loss +
            latency_loss +
            energy_loss +
            self.kld_weight * kld_loss +
            self.action_weight * action_loss
        )
        
        return total_loss, {
            'kld_loss': kld_loss.item(),
            'offload_loss': offload_loss.item(),
            'latency_loss': latency_loss.item(),
            'energy_loss': energy_loss.item(),
            'action_loss': action_loss.item()
        }
    
    def compute_action_loss(self, mean, log_std, target):
        dist = Normal(mean, log_std.exp())
        log_prob = dist.log_prob(target)
        return -log_prob.mean()

def train_hybrid_system():
    """Training function for hybrid system"""
    # Initialize parameters
    input_dim = 128
    action_dim = 10
    node_features = 64
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.001
    
    # Initialize model and optimizer
    model = HybridVECSystem(input_dim, action_dim, node_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = HybridLoss()
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in range(batch_size):
            # Generate dummy data (replace with your actual data)
            temporal_data = torch.randn(32, 10, input_dim)  # [batch, seq_len, features]
            spatial_data = torch.randn(32, node_features)   # [batch, node_features]
            edge_index = torch.randint(0, 32, (2, 50))     # Random edges
            
            # Forward pass
            outputs = model(temporal_data, spatial_data, edge_index)
            
            # Prepare targets (replace with your actual targets)
            targets = {
                'offload_target': torch.randn(32, action_dim),
                'action_target': torch.randn(32, action_dim)
            }
            
            # Calculate loss
            loss, metrics = criterion(outputs, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print epoch statistics
        avg_loss = total_loss / batch_size
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        print(f"Metrics: {metrics}")

if __name__ == "__main__":
    train_hybrid_system()