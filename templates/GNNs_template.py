import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, MessagePassing
from torch_geometric.data import Data, Batch
import numpy as np

class EdgeConv(MessagePassing):
    """Custom edge convolution layer for VEC"""
    def __init__(self, in_channels, out_channels):
        super(EdgeConv, self).__init__(aggr='max')  # Maximum aggregation
        
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
    
    def forward(self, x, edge_index, edge_attr=None):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr=None):
        # Concatenate node features
        tmp = torch.cat([x_i, x_j], dim=1)
        
        # Apply MLP
        return self.mlp(tmp)

class VECGraphNet(nn.Module):
    """Graph Neural Network for VEC task offloading"""
    def __init__(self, 
                 node_features,
                 edge_features,
                 hidden_dim=64,
                 num_layers=3):
        super(VECGraphNet, self).__init__()
        
        # Initial node feature processing
        self.node_encoder = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Edge feature processing
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList([
            EdgeConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Attention layers for vehicle-server interactions
        self.attention_layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Output layers for different predictions
        self.offload_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Offloading decision
        )
        
        self.latency_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Latency prediction
        )
        
        self.energy_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Energy consumption prediction
        )
    
    def forward(self, data):
        # Get graph components
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Encode node features
        x = self.node_encoder(x)
        
        # Encode edge features if present
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
        
        # Apply graph convolutions with residual connections
        for conv, attention in zip(self.conv_layers, self.attention_layers):
            # Graph convolution
            x_conv = conv(x, edge_index, edge_attr)
            # Attention mechanism
            x_attention = attention(x, edge_index)
            # Combine with residual connection
            x = x + x_conv + x_attention
            # Apply non-linearity
            x = F.relu(x)
        
        # Generate predictions
        offload_decisions = self.offload_predictor(x)
        latency = self.latency_predictor(x)
        energy = self.energy_predictor(x)
        
        return offload_decisions, latency, energy

class VECGraphDataset:
    """Dataset class for VEC graph data"""
    def __init__(self, num_vehicles, num_servers):
        self.num_vehicles = num_vehicles
        self.num_servers = num_servers
    
    def create_graph(self, vehicle_states, server_states, connectivity):
        """Create a graph from vehicle and server states"""
        # Combine vehicle and server features
        node_features = torch.cat([
            torch.tensor(vehicle_states, dtype=torch.float),
            torch.tensor(server_states, dtype=torch.float)
        ])
        
        # Create edge indices based on connectivity
        edge_index = []
        edge_attr = []
        
        # Add edges between vehicles and servers
        for v in range(self.num_vehicles):
            for s in range(self.num_servers):
                if connectivity[v][s]:  # If connection exists
                    # Add edge in both directions
                    edge_index.append([v, self.num_vehicles + s])
                    edge_index.append([self.num_vehicles + s, v])
                    
                    # Add edge features (e.g., bandwidth, distance)
                    edge_attr.append(self._create_edge_features(v, s))
                    edge_attr.append(self._create_edge_features(v, s))
        
        # Convert to torch tensors
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        # Create PyG Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr
        )
        
        return data
    
    def _create_edge_features(self, vehicle_idx, server_idx):
        """Create features for edges (customize based on your needs)"""
        # Example edge features: distance, bandwidth, signal strength
        return [
            np.random.rand(),  # Distance
            np.random.rand(),  # Bandwidth
            np.random.rand()   # Signal strength
        ]

def train_gnn():
    """Training function for VEC Graph Neural Network"""
    # Initialize parameters
    num_vehicles = 10
    num_servers = 5
    node_features = 16  # Features per node
    edge_features = 3   # Features per edge
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.001
    
    # Initialize model
    model = VECGraphNet(
        node_features=node_features,
        edge_features=edge_features
    )
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize dataset
    dataset = VECGraphDataset(num_vehicles, num_servers)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in range(batch_size):
            # Generate random vehicle and server states
            vehicle_states = np.random.rand(num_vehicles, node_features)
            server_states = np.random.rand(num_servers, node_features)
            connectivity = np.random.rand(num_vehicles, num_servers) > 0.5
            
            # Create graph
            data = dataset.create_graph(vehicle_states, server_states, connectivity)
            
            # Forward pass
            offload_decisions, latency, energy = model(data)
            
            # Calculate loss
            loss = calculate_loss(offload_decisions, latency, energy, data)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print epoch statistics
        avg_loss = total_loss / batch_size
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

def calculate_loss(offload_decisions, latency, energy, data):
    """Calculate combined loss for offloading decisions"""
    # Example loss calculation (customize based on your needs)
    offload_loss = F.mse_loss(offload_decisions, torch.zeros_like(offload_decisions))
    latency_loss = F.mse_loss(latency, torch.zeros_like(latency))
    energy_loss = F.mse_loss(energy, torch.zeros_like(energy))
    
    # Combine losses with weights
    total_loss = (
        0.4 * offload_loss +
        0.3 * latency_loss +
        0.3 * energy_loss
    )
    
    return total_loss

def evaluate_gnn(model, test_dataset):
    """Evaluation function for VEC Graph Neural Network"""
    model.eval()
    
    with torch.no_grad():
        # Create test graph
        test_data = test_dataset.create_graph(
            np.random.rand(test_dataset.num_vehicles, 16),
            np.random.rand(test_dataset.num_servers, 16),
            np.random.rand(test_dataset.num_vehicles, test_dataset.num_servers) > 0.5
        )
        
        # Get predictions
        offload_decisions, latency, energy = model(test_data)
        
        # Calculate metrics
        calculate_metrics(offload_decisions, latency, energy, test_data)

def calculate_metrics(offload_decisions, latency, energy, data):
    """Calculate evaluation metrics"""
    # Example metrics (customize based on your needs)
    offload_accuracy = (offload_decisions > 0.5).float().mean()
    avg_latency = latency.mean()
    avg_energy = energy.mean()
    
    print(f"Offload Accuracy: {offload_accuracy:.4f}")
    print(f"Average Latency: {avg_latency:.4f}")
    print(f"Average Energy: {avg_energy:.4f}")

if __name__ == "__main__":
    train_gnn()