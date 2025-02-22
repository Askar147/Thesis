import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchdiffeq import odeint_adjoint as odeint

class ODEFunc(nn.Module):
    """ODE function defining the dynamics of the VEC system"""
    def __init__(self, state_dim, hidden_dim=64):
        super(ODEFunc, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # Optional: Add state-specific dynamics
        self.vehicle_dynamics = nn.Linear(hidden_dim, hidden_dim)
        self.server_dynamics = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, t, state):
        """
        t: Current time point
        state: System state including vehicle and server states
        """
        # Split state into components
        vehicle_state = state[:, :self.vehicle_dim]
        server_state = state[:, self.vehicle_dim:]
        
        # Compute general dynamics
        state_deriv = self.net(state)
        
        # Add specific dynamics for vehicles and servers
        vehicle_deriv = self.vehicle_dynamics(vehicle_state)
        server_deriv = self.server_dynamics(server_state)
        
        # Combine derivatives
        combined_deriv = torch.cat([vehicle_deriv, server_deriv], dim=1)
        
        return state_deriv + combined_deriv

class VECNeuralODE(nn.Module):
    """Neural ODE model for VEC system"""
    def __init__(self, state_dim, hidden_dim=64):
        super(VECNeuralODE, self).__init__()
        
        self.state_dim = state_dim
        self.odefunc = ODEFunc(state_dim, hidden_dim)
        
        # Additional networks for processing input/output
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
    def forward(self, x, time_steps):
        """
        x: Initial state
        time_steps: Time points to evaluate
        """
        # Encode initial state
        x = self.encoder(x)
        
        # Solve ODE
        out = odeint(self.odefunc, x, time_steps)
        
        # Decode solution
        out = self.decoder(out)
        
        return out

class LatencyPredictor(nn.Module):
    """Predicts latency based on system state"""
    def __init__(self, state_dim, hidden_dim=32):
        super(LatencyPredictor, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        return self.net(state)

class EnergyPredictor(nn.Module):
    """Predicts energy consumption based on system state"""
    def __init__(self, state_dim, hidden_dim=32):
        super(EnergyPredictor, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        return self.net(state)

class VECSystem:
    """Complete VEC system using Neural ODEs"""
    def __init__(self, num_vehicles, num_servers, hidden_dim=64):
        # Calculate state dimensions
        self.vehicle_dim = num_vehicles * 3  # position(x,y) + task_size
        self.server_dim = num_servers * 2    # load + capacity
        self.state_dim = self.vehicle_dim + self.server_dim
        
        # Initialize models
        self.node_model = VECNeuralODE(self.state_dim, hidden_dim)
        self.latency_predictor = LatencyPredictor(self.state_dim)
        self.energy_predictor = EnergyPredictor(self.state_dim)
        
        # Initialize optimizers
        self.node_optimizer = optim.Adam(self.node_model.parameters())
        self.latency_optimizer = optim.Adam(self.latency_predictor.parameters())
        self.energy_optimizer = optim.Adam(self.energy_predictor.parameters())
        
        # Time steps for integration
        self.integration_time = torch.linspace(0., 1., 100)
    
    def prepare_state(self, vehicle_states, server_states):
        """Prepare system state for Neural ODE"""
        return torch.cat([
            torch.tensor(vehicle_states, dtype=torch.float32).flatten(),
            torch.tensor(server_states, dtype=torch.float32).flatten()
        ])
    
    def predict_trajectory(self, initial_state):
        """Predict system state trajectory"""
        state = self.prepare_state(initial_state)
        trajectory = self.node_model(state, self.integration_time)
        return trajectory
    
    def predict_metrics(self, state):
        """Predict latency and energy consumption"""
        latency = self.latency_predictor(state)
        energy = self.energy_predictor(state)
        return latency, energy
    
    def train_step(self, batch_initial_states, batch_target_states):
        """Training step"""
        # Predict trajectories
        predicted_trajectories = self.node_model(
            batch_initial_states, 
            self.integration_time
        )
        
        # Calculate trajectory loss
        trajectory_loss = nn.MSELoss()(
            predicted_trajectories, 
            batch_target_states
        )
        
        # Predict metrics
        latency, energy = self.predict_metrics(predicted_trajectories[-1])
        
        # Calculate metric losses
        latency_loss = nn.MSELoss()(
            latency,
            torch.zeros_like(latency)  # Target: minimize latency
        )
        
        energy_loss = nn.MSELoss()(
            energy,
            torch.zeros_like(energy)   # Target: minimize energy
        )
        
        # Total loss
        total_loss = trajectory_loss + 0.1 * latency_loss + 0.1 * energy_loss
        
        # Update models
        self.node_optimizer.zero_grad()
        self.latency_optimizer.zero_grad()
        self.energy_optimizer.zero_grad()
        
        total_loss.backward()
        
        self.node_optimizer.step()
        self.latency_optimizer.step()
        self.energy_optimizer.step()
        
        return {
            'trajectory_loss': trajectory_loss.item(),
            'latency_loss': latency_loss.item(),
            'energy_loss': energy_loss.item(),
            'total_loss': total_loss.item()
        }

def train_neural_ode():
    """Training function for VEC Neural ODE system"""
    # Initialize parameters
    num_vehicles = 10
    num_servers = 5
    hidden_dim = 64
    num_epochs = 100
    batch_size = 32
    
    # Initialize system
    vec_system = VECSystem(num_vehicles, num_servers, hidden_dim)
    
    # Training loop
    for epoch in range(num_epochs):
        total_losses = {
            'trajectory_loss': 0,
            'latency_loss': 0,
            'energy_loss': 0,
            'total_loss': 0
        }
        
        for batch in range(batch_size):
            # Generate random training data
            initial_vehicle_states = np.random.rand(num_vehicles, 3)
            initial_server_states = np.random.rand(num_servers, 2)
            
            target_vehicle_states = np.random.rand(num_vehicles, 3)
            target_server_states = np.random.rand(num_servers, 2)
            
            # Prepare states
            initial_states = vec_system.prepare_state(
                initial_vehicle_states,
                initial_server_states
            )
            
            target_states = vec_system.prepare_state(
                target_vehicle_states,
                target_server_states
            )
            
            # Training step
            losses = vec_system.train_step(
                initial_states.unsqueeze(0),
                target_states.unsqueeze(0)
            )
            
            # Accumulate losses
            for key in total_losses:
                total_losses[key] += losses[key]
        
        # Print epoch statistics
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for key in total_losses:
            avg_loss = total_losses[key] / batch_size
            print(f"Average {key}: {avg_loss:.4f}")
        print("------------------------")

def evaluate_predictions(vec_system, test_states):
    """Evaluate system predictions"""
    with torch.no_grad():
        # Predict trajectory
        trajectory = vec_system.predict_trajectory(test_states)
        
        # Predict metrics for final state
        latency, energy = vec_system.predict_metrics(trajectory[-1])
        
        return trajectory, latency, energy

if __name__ == "__main__":
    train_neural_ode()