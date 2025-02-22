import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for temporal information"""
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class VECTransformerEncoder(nn.Module):
    """Transformer encoder adapted for VEC task offloading"""
    def __init__(self, 
                 input_dim,
                 d_model=256,
                 nhead=8,
                 num_layers=6,
                 dim_feedforward=1024,
                 dropout=0.1):
        super().__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection for different tasks
        self.server_load_predictor = nn.Linear(d_model, 1)  # Predict server load
        self.latency_predictor = nn.Linear(d_model, 1)      # Predict latency
        self.energy_predictor = nn.Linear(d_model, 1)       # Predict energy consumption
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Project input to d_model dimensions
        x = self.input_projection(src)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transform through encoder layers
        x = self.transformer_encoder(x, src_mask, src_key_padding_mask)
        
        # Get predictions
        server_load = self.server_load_predictor(x)
        latency = self.latency_predictor(x)
        energy = self.energy_predictor(x)
        
        return server_load, latency, energy

class VECTransformerDecoder(nn.Module):
    """Transformer decoder for offloading decisions"""
    def __init__(self,
                 d_model=256,
                 nhead=8,
                 num_layers=6,
                 dim_feedforward=1024,
                 dropout=0.1,
                 num_servers=10):
        super().__init__()
        
        # Decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection for offloading decisions
        self.offload_predictor = nn.Linear(d_model, num_servers)
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Transform through decoder layers
        x = self.transformer_decoder(
            tgt, memory,
            tgt_mask, memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask
        )
        
        # Get offloading decisions
        offload_decisions = self.offload_predictor(x)
        return offload_decisions

class VECTransformer(nn.Module):
    """Complete Transformer model for VEC task offloading"""
    def __init__(self,
                 input_dim,
                 d_model=256,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.1,
                 num_servers=10):
        super().__init__()
        
        self.encoder = VECTransformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        self.decoder = VECTransformerDecoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_servers=num_servers
        )
        
    def forward(self, src, tgt):
        # Generate masks
        src_mask, tgt_mask = self._generate_masks(src, tgt)
        
        # Encode input sequence
        server_load, latency, energy = self.encoder(src, src_mask)
        
        # Decode for offloading decisions
        offload_decisions = self.decoder(tgt, server_load, tgt_mask)
        
        return offload_decisions, server_load, latency, energy
    
    def _generate_masks(self, src, tgt):
        # Generate masks for transformer
        src_seq_len = src.size(1)
        tgt_seq_len = tgt.size(1)
        
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=src.device).bool()
        tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len, device=tgt.device)
        
        return src_mask, tgt_mask
    
    @staticmethod
    def _generate_square_subsequent_mask(sz, device):
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class VECDataset(torch.utils.data.Dataset):
    """Dataset for VEC task offloading"""
    def __init__(self, vehicle_data, server_data, sequence_length=50):
        self.vehicle_data = vehicle_data
        self.server_data = server_data
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.vehicle_data) - self.sequence_length
    
    def __getitem__(self, idx):
        # Get sequence of vehicle and server states
        vehicle_seq = self.vehicle_data[idx:idx + self.sequence_length]
        server_seq = self.server_data[idx:idx + self.sequence_length]
        
        # Combine into input sequence
        input_seq = torch.cat([vehicle_seq, server_seq], dim=-1)
        
        # Target is the next timestep's server states
        target = server_seq[1:]
        
        return input_seq, target

def train_transformer():
    """Training function for VEC Transformer"""
    # Initialize parameters
    input_dim = 128  # Combined dimension of vehicle and server states
    num_servers = 10
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.0001
    
    # Initialize model
    model = VECTransformer(
        input_dim=input_dim,
        num_servers=num_servers
    )
    
    # Initialize optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Create dummy dataset (replace with your actual data)
    vehicle_data = torch.randn(1000, 64)  # Vehicle states
    server_data = torch.randn(1000, 64)   # Server states
    dataset = VECDataset(vehicle_data, server_data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (input_seq, target) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward pass
            offload_decisions, server_load, latency, energy = model(input_seq, target)
            
            # Calculate loss
            loss = criterion(offload_decisions, target)
            loss += criterion(server_load, target)
            loss += criterion(latency, torch.zeros_like(latency))  # Minimize latency
            loss += criterion(energy, torch.zeros_like(energy))    # Minimize energy
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print epoch statistics
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")


def calculate_offload_accuracy(predictions, targets):
    """
    Calculate accuracy of offloading decisions
    
    Args:
        predictions: Predicted offloading decisions (batch_size, seq_len, num_servers)
        targets: Actual optimal offloading decisions (batch_size, seq_len, num_servers)
    
    Returns:
        float: Accuracy score between 0 and 1
    """
    # Convert probabilities to actual server selections
    pred_servers = torch.argmax(predictions, dim=-1)
    target_servers = torch.argmax(targets, dim=-1)
    
    # Calculate accuracy
    correct = (pred_servers == target_servers).float()
    accuracy = correct.mean().item()
    
    return accuracy

def calculate_energy_efficiency(energy_consumption):
    """
    Calculate energy efficiency score
    
    Args:
        energy_consumption: Predicted energy consumption (batch_size, seq_len, 1)
    
    Returns:
        float: Energy efficiency score (lower is better)
    """
    # Normalize energy consumption to 0-1 range
    normalized_energy = (energy_consumption - energy_consumption.min()) / \
                       (energy_consumption.max() - energy_consumption.min() + 1e-8)
    
    # Calculate efficiency score (inverse of normalized energy)
    efficiency_score = 1 - normalized_energy.mean().item()
    
    return efficiency_score

def calculate_latency_score(latency):
    """
    Calculate latency performance score
    
    Args:
        latency: Predicted latency values (batch_size, seq_len, 1)
    
    Returns:
        float: Latency score (lower is better)
    """
    # Normalize latency to 0-1 range
    normalized_latency = (latency - latency.min()) / \
                        (latency.max() - latency.min() + 1e-8)
    
    # Calculate latency score (inverse of normalized latency)
    latency_score = 1 - normalized_latency.mean().item()
    
    return latency_score

def calculate_comprehensive_score(offload_accuracy, energy_efficiency, latency_score, 
                                weights={'accuracy': 0.4, 'energy': 0.3, 'latency': 0.3}):
    """
    Calculate comprehensive performance score
    
    Args:
        offload_accuracy: Accuracy of offloading decisions
        energy_efficiency: Energy efficiency score
        latency_score: Latency performance score
        weights: Dictionary of weights for each metric
    
    Returns:
        float: Comprehensive performance score between 0 and 1
    """
    comprehensive_score = (
        weights['accuracy'] * offload_accuracy +
        weights['energy'] * energy_efficiency +
        weights['latency'] * latency_score
    )
    
    return comprehensive_score

# Enhanced evaluation function
def evaluate_transformer(model, test_loader, weights={'accuracy': 0.4, 'energy': 0.3, 'latency': 0.3}):
    """
    Comprehensive evaluation of the VEC Transformer model
    
    Args:
        model: VEC Transformer model
        test_loader: DataLoader containing test data
        weights: Dictionary of weights for different metrics
    
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    model.eval()
    total_metrics = {
        'offload_accuracy': 0,
        'energy_efficiency': 0,
        'latency_score': 0,
        'comprehensive_score': 0
    }
    num_batches = 0
    
    with torch.no_grad():
        for input_seq, target in test_loader:
            # Forward pass
            offload_decisions, server_load, latency, energy = model(input_seq, target)
            
            # Calculate individual metrics
            batch_accuracy = calculate_offload_accuracy(offload_decisions, target)
            batch_energy_efficiency = calculate_energy_efficiency(energy)
            batch_latency_score = calculate_latency_score(latency)
            
            # Calculate comprehensive score
            batch_comprehensive_score = calculate_comprehensive_score(
                batch_accuracy,
                batch_energy_efficiency,
                batch_latency_score,
                weights
            )
            
            # Accumulate metrics
            total_metrics['offload_accuracy'] += batch_accuracy
            total_metrics['energy_efficiency'] += batch_energy_efficiency
            total_metrics['latency_score'] += batch_latency_score
            total_metrics['comprehensive_score'] += batch_comprehensive_score
            
            num_batches += 1
    
    # Calculate averages
    for metric in total_metrics:
        total_metrics[metric] /= num_batches
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Offload Accuracy: {total_metrics['offload_accuracy']:.4f}")
    print(f"Energy Efficiency: {total_metrics['energy_efficiency']:.4f}")
    print(f"Latency Score: {total_metrics['latency_score']:.4f}")
    print(f"Comprehensive Score: {total_metrics['comprehensive_score']:.4f}")
    
    return total_metrics

def evaluate_transformer(model, test_loader):
    """Evaluation function for VEC Transformer"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for input_seq, target in test_loader:
            # Forward pass
            offload_decisions, server_load, latency, energy = model(input_seq, target)
            
            # Calculate metrics
            offload_accuracy = calculate_offload_accuracy(offload_decisions, target)
            energy_efficiency = calculate_energy_efficiency(energy)
            latency_score = calculate_latency_score(latency)
            
            print(f"Offload Accuracy: {offload_accuracy:.4f}")
            print(f"Energy Efficiency: {energy_efficiency:.4f}")
            print(f"Latency Score: {latency_score:.4f}")

if __name__ == "__main__":
    train_transformer()