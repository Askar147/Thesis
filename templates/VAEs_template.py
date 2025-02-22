import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

class VECEncoder(nn.Module):
    """Encoder network for VEC VAE"""
    def __init__(self, input_dim, hidden_dim=256, latent_dim=32):
        super(VECEncoder, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Mean and log variance layers
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        # Encode input
        x = self.encoder(x)
        
        # Get mean and log variance
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar

class VECDecoder(nn.Module):
    """Decoder network for VEC VAE"""
    def __init__(self, latent_dim, hidden_dim=256, output_dim=None):
        super(VECDecoder, self).__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Separate heads for different outputs
        self.compute_load_head = nn.Linear(output_dim, output_dim // 3)
        self.network_state_head = nn.Linear(output_dim, output_dim // 3)
        self.resource_usage_head = nn.Linear(output_dim, output_dim // 3)
        
    def forward(self, z):
        # Decode latent vector
        x = self.decoder(z)
        
        # Get different aspects of the system
        compute_load = self.compute_load_head(x)
        network_state = self.network_state_head(x)
        resource_usage = self.resource_usage_head(x)
        
        return compute_load, network_state, resource_usage

class VECVAE(nn.Module):
    """VAE for VEC system state modeling"""
    def __init__(self, input_dim, hidden_dim=256, latent_dim=32):
        super(VECVAE, self).__init__()
        
        self.encoder = VECEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = VECDecoder(latent_dim, hidden_dim, input_dim)
        
        self.latent_dim = latent_dim
        
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # Encode
        mu, logvar = self.encoder(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        compute_load, network_state, resource_usage = self.decoder(z)
        
        return compute_load, network_state, resource_usage, mu, logvar
    
    def generate(self, num_samples):
        """Generate new system states"""
        with torch.no_grad():
            # Sample from latent space
            z = torch.randn(num_samples, self.latent_dim)
            
            # Decode samples
            compute_load, network_state, resource_usage = self.decoder(z)
            
            return compute_load, network_state, resource_usage

class VECDataProcessor:
    """Process and prepare VEC system data"""
    def __init__(self, num_vehicles, num_servers):
        self.num_vehicles = num_vehicles
        self.num_servers = num_servers
        
    def prepare_input(self, vehicle_states, server_states, network_states):
        """Combine different states into input vector"""
        return np.concatenate([
            vehicle_states.flatten(),
            server_states.flatten(),
            network_states.flatten()
        ])
    
    def decompose_output(self, compute_load, network_state, resource_usage):
        """Decompose output into meaningful components"""
        # Reshape outputs into meaningful structures
        vehicle_loads = compute_load[:self.num_vehicles]
        server_loads = compute_load[self.num_vehicles:]
        
        network_conditions = network_state.reshape(self.num_vehicles, self.num_servers)
        
        resource_allocation = resource_usage.reshape(self.num_vehicles, self.num_servers)
        
        return vehicle_loads, server_loads, network_conditions, resource_allocation

def vae_loss_function(compute_load, network_state, resource_usage, 
                     compute_load_target, network_state_target, resource_usage_target,
                     mu, logvar, kld_weight=0.1):
    """Custom loss function for VEC VAE"""
    # Reconstruction losses for different components
    compute_loss = F.mse_loss(compute_load, compute_load_target)
    network_loss = F.mse_loss(network_state, network_state_target)
    resource_loss = F.mse_loss(resource_usage, resource_usage_target)
    
    # KL divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    total_loss = compute_loss + network_loss + resource_loss + kld_weight * kld
    
    return total_loss, {
        'compute_loss': compute_loss.item(),
        'network_loss': network_loss.item(),
        'resource_loss': resource_loss.item(),
        'kld': kld.item()
    }

def train_vae():
    """Training function for VEC VAE"""
    # Initialize parameters
    num_vehicles = 10
    num_servers = 5
    input_dim = num_vehicles * 3 + num_servers * 3 + num_vehicles * num_servers
    hidden_dim = 256
    latent_dim = 32
    batch_size = 64
    num_epochs = 100
    learning_rate = 0.001
    
    # Initialize model
    model = VECVAE(input_dim, hidden_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize data processor
    data_processor = VECDataProcessor(num_vehicles, num_servers)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in range(batch_size):
            # Generate random training data
            vehicle_states = np.random.rand(num_vehicles, 3)  # position, velocity, compute need
            server_states = np.random.rand(num_servers, 3)    # load, capacity, efficiency
            network_states = np.random.rand(num_vehicles, num_servers)  # connectivity
            
            # Prepare input
            x = data_processor.prepare_input(vehicle_states, server_states, network_states)
            x = torch.FloatTensor(x)
            
            # Forward pass
            compute_load, network_state, resource_usage, mu, logvar = model(x)
            
            # Calculate loss
            loss, metrics = vae_loss_function(
                compute_load, network_state, resource_usage,
                x[:num_vehicles * 3], x[num_vehicles * 3:num_vehicles * 3 + num_servers * 3],
                x[num_vehicles * 3 + num_servers * 3:],
                mu, logvar
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print epoch statistics
        avg_loss = total_loss / batch_size
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        print(f"Metrics: {metrics}")

def generate_scenarios(model, data_processor, num_scenarios=10):
    """Generate different system state scenarios"""
    compute_load, network_state, resource_usage = model.generate(num_scenarios)
    
    scenarios = []
    for i in range(num_scenarios):
        vehicle_loads, server_loads, network_conditions, resource_allocation = \
            data_processor.decompose_output(
                compute_load[i], network_state[i], resource_usage[i]
            )
        
        scenarios.append({
            'vehicle_loads': vehicle_loads,
            'server_loads': server_loads,
            'network_conditions': network_conditions,
            'resource_allocation': resource_allocation
        })
    
    return scenarios

if __name__ == "__main__":
    train_vae()