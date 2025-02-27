import numpy as np
import gym
from gym import spaces
import torch
from collections import deque


class MECEnvironment(gym.Env):
    """
    Mobile Edge Computing Environment that simulates task offloading decisions
    between mobile vehicles (MVs) and edge servers (ESs).
    
    This environment models:
    - Multiple mobile vehicles generating computational tasks
    - Multiple edge servers with different computational capacities
    - Network conditions between MVs and ESs
    - Task-specific attributes (size, required CPU cycles, deadline)
    - Decision making for optimal task offloading
    - Resource allocation including computing and transmission resources
    """
    
    def __init__(self, 
                 num_mvs=20,
                 num_edge_servers=5,
                 continuous_action=True,
                 history_length=48,
                 difficulty='normal',
                 seed=None):
        """
        Initialize the MEC environment.
        
        Args:
            num_mvs: Number of mobile vehicles in the environment
            num_edge_servers: Number of edge servers in the environment
            continuous_action: Whether to use continuous action space
            history_length: Number of previous states to keep in history
            difficulty: Difficulty level - 'easy', 'normal', or 'hard'
            seed: Random seed for reproducibility
        """
        super(MECEnvironment, self).__init__()
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Environment parameters
        self.num_mvs = num_mvs
        self.num_edge_servers = num_edge_servers
        self.continuous_action = continuous_action
        self.history_length = history_length
        self.difficulty = difficulty
        
        # Define action space
        if continuous_action:
            # Action: [task_offload_decision(0-1), transmission_power(0-1), computation_resource(0-1)]
            # For each MV, decide offloading decision, transmission power, and computation resource
            self.action_space = spaces.Box(
                low=0, high=1, shape=(num_mvs, 3), dtype=np.float32
            )
        else:
            # Discrete offloading decisions: 0 = local processing, 1-num_edge_servers = offload to that ES
            self.action_space = spaces.Discrete(num_edge_servers + 1)
        
        # Define observation space
        # MV observations include task info, server info, and network info
        self.observation_space = spaces.Dict({
            # Task information
            'task_size': spaces.Box(low=0, high=1, shape=(num_mvs, 1), dtype=np.float32),
            'required_cpu_cycles': spaces.Box(low=0, high=1, shape=(num_mvs, 1), dtype=np.float32),
            'task_deadline': spaces.Box(low=0, high=1, shape=(num_mvs, 1), dtype=np.float32),
            
            # MV information
            'mv_compute_capacity': spaces.Box(low=0, high=1, shape=(num_mvs, 1), dtype=np.float32),
            'mv_local_queue': spaces.Box(low=0, high=1, shape=(num_mvs, 1), dtype=np.float32),
            'mv_energy': spaces.Box(low=0, high=1, shape=(num_mvs, 1), dtype=np.float32),
            'mv_locations': spaces.Box(low=0, high=1, shape=(num_mvs, 2), dtype=np.float32),  # x, y
            
            # ES information
            'es_compute_capacity': spaces.Box(low=0, high=1, shape=(num_edge_servers, 1), dtype=np.float32),
            'es_queue_length': spaces.Box(low=0, high=1, shape=(num_edge_servers, 1), dtype=np.float32),
            'es_locations': spaces.Box(low=0, high=1, shape=(num_edge_servers, 2), dtype=np.float32),  # x, y
            
            # Network conditions
            'channel_gain': spaces.Box(low=0, high=1, shape=(num_mvs, num_edge_servers), dtype=np.float32),
            'network_delay': spaces.Box(low=0, high=1, shape=(num_mvs, num_edge_servers), dtype=np.float32),
            
            # Historical ES queue information (for prediction)
            'es_queue_history': spaces.Box(low=0, high=1, shape=(history_length, num_edge_servers), dtype=np.float32)
        })
        
        # Set environment difficulty parameters
        self._set_difficulty_params(difficulty)
        
        # Initialize history buffers
        self.state_history = deque(maxlen=history_length)
        self.es_queue_history = deque(maxlen=history_length)
        
        # Performance tracking
        self.task_completion_history = deque(maxlen=100)
        self.reward_history = deque(maxlen=100)
        self.energy_history = deque(maxlen=100)
        self.latency_history = deque(maxlen=100)
        
        # Initialize environment state
        self.reset()
    
    def _set_difficulty_params(self, difficulty):
        """Set environment parameters based on difficulty level"""
        # Base parameters based on the paper's specifications
        self.task_size_range = (500, 10000)  # Data size in KB (from 500KB to 10MB)
        self.cpu_cycles_range = (49, 1123)   # CPU cycles required in megacycles
        self.deadline_range = (0.5, 2.5)     # Deadline in seconds
        
        # Network parameters
        self.bandwidth = 5  # Bandwidth in MHz, as specified in the paper
        self.path_loss_exponent = 4  # Path loss exponent for wireless transmission
        
        # Compute capabilities
        self.mv_compute_range = (2.5, 5.5)  # MV CPU capacity in GHz
        self.es_compute_range = (11.8, 21.8)  # ES CPU capacity in GHz
        
        # Power-related parameters
        self.max_transmission_power = 2.0  # Maximum transmission power in Watts
        self.mv_energy_range = (10, 20)    # Available energy in Joules
        self.noise_power = 1e-13           # Background noise power in Watts
        
        # ES queue parameters
        self.queue_init_range = (0.1, 0.3)  # Initial queue load as a fraction of capacity
        self.queue_update_factor = 0.1      # How quickly queues update
        self.mv_mobility_factor = 0.05      # How fast MVs move (in meters per time slot)
        
        # Weights for reward calculation
        self.latency_weight = 0.5
        self.energy_weight = 0.3
        self.completion_weight = 0.2
        
        # Parameters for channel gain calculation
        self.channel_gain_base = 1e-6  # Base channel gain at 1m distance
        
        # Adjust parameters based on difficulty
        if difficulty == 'easy':
            self.task_size_range = (500, 5000)  # Smaller tasks
            self.deadline_range = (0.8, 3.0)    # More generous deadlines
            self.mv_compute_range = (3.0, 5.5)  # Better MV computing power
            self.mv_mobility_factor = 0.03      # Slower MV movement
            self.queue_update_factor = 0.05     # Slower queue buildup
        elif difficulty == 'hard':
            self.task_size_range = (2000, 10000)  # Larger tasks
            self.deadline_range = (0.3, 1.5)      # Stricter deadlines
            self.mv_compute_range = (2.0, 4.5)    # Lower MV computing power
            self.mv_mobility_factor = 0.08        # Faster MV movement
            self.queue_update_factor = 0.15       # Faster queue buildup
            self.noise_power = 5e-13              # Higher noise power
    
    def reset(self):
        """Reset the environment to initial state"""
        # Initialize MV attributes
        self.mv_compute_capacity = np.random.uniform(
            self.mv_compute_range[0], 
            self.mv_compute_range[1], 
            (self.num_mvs, 1)
        )
        
        self.mv_energy = np.random.uniform(
            self.mv_energy_range[0], 
            self.mv_energy_range[1], 
            (self.num_mvs, 1)
        )
        
        # MV positions in a 1x1 grid
        self.mv_locations = np.random.uniform(0, 1, (self.num_mvs, 2))
        self.mv_local_queue = np.zeros((self.num_mvs, 1))
        
        # Initialize ES attributes
        self.es_compute_capacity = np.random.uniform(
            self.es_compute_range[0], 
            self.es_compute_range[1], 
            (self.num_edge_servers, 1)
        )
        
        # ES fixed positions in a 1x1 grid
        self.es_locations = np.random.uniform(0, 1, (self.num_edge_servers, 2))
        
        self.es_queue_length = np.random.uniform(
            self.queue_init_range[0], 
            self.queue_init_range[1], 
            (self.num_edge_servers, 1)
        )
        
        # Initialize network conditions
        self._update_network_conditions()
        
        # Generate new tasks for each MV
        self._generate_tasks()
        
        # Initialize queue history
        self.es_queue_history.clear()
        for _ in range(self.history_length):
            self.es_queue_history.append(self.es_queue_length.copy())
        
        # Get initial observation
        obs = self._get_observation()
        
        # Reset history
        self.state_history.clear()
        self.state_history.append(obs)
        
        return obs
    
    def _generate_tasks(self):
        """Generate new computational tasks for each MV"""
        self.task_size = np.random.uniform(
            self.task_size_range[0], 
            self.task_size_range[1], 
            (self.num_mvs, 1)
        )
        
        self.required_cpu_cycles = np.random.uniform(
            self.cpu_cycles_range[0], 
            self.cpu_cycles_range[1], 
            (self.num_mvs, 1)
        )
        
        self.task_deadline = np.random.uniform(
            self.deadline_range[0], 
            self.deadline_range[1], 
            (self.num_mvs, 1)
        )
    
    def _update_network_conditions(self):
        """Update channel conditions and network delays"""
        # Calculate distance between each MV and ES (in meters, assuming 1.0 = 1000m)
        distances = np.zeros((self.num_mvs, self.num_edge_servers))
        for i in range(self.num_mvs):
            for j in range(self.num_edge_servers):
                distances[i, j] = np.sqrt(np.sum((self.mv_locations[i] - self.es_locations[j]) ** 2)) * 1000
                # Ensure minimum distance
                distances[i, j] = max(distances[i, j], 1.0)
        
        # Calculate channel gain based on distance and path loss exponent
        self.channel_gain = np.zeros((self.num_mvs, self.num_edge_servers))
        for i in range(self.num_mvs):
            for j in range(self.num_edge_servers):
                # Channel gain based on free-space path loss model
                self.channel_gain[i, j] = self.channel_gain_base * (distances[i, j] ** (-self.path_loss_exponent))
                
                # Add small random variation (±20%)
                random_factor = np.random.uniform(0.8, 1.2)
                self.channel_gain[i, j] *= random_factor
        
        # Network delay increases with distance
        # Base delay (in seconds) + distance-dependent component
        self.network_delay = 0.01 + 0.05 * (distances / 1000)  # Convert distance to kilometers for delay calculation
    
    def _update_mv_locations(self):
        """Update MV locations based on mobility model"""
        # Simple random mobility model
        movement = np.random.normal(0, self.mv_mobility_factor, self.mv_locations.shape)
        self.mv_locations += movement
        
        # Ensure locations stay within bounds
        self.mv_locations = np.clip(self.mv_locations, 0, 1)
    
    def _calculate_transmission_rate(self, mv_idx, es_idx, power):
        """Calculate transmission rate from MV to ES in Mbps"""
        # Calculate distance between MV and ES in meters (assuming 1.0 = 1000m)
        distance = np.sqrt(np.sum((self.mv_locations[mv_idx] - self.es_locations[es_idx]) ** 2)) * 1000
        distance = max(distance, 1.0)  # Minimum distance of 1 meter
        
        # Calculate channel gain based on distance and path loss
        channel_gain = self.channel_gain_base * (distance ** (-self.path_loss_exponent))
        
        # Calculate actual transmit power in Watts (normalize from 0-1 to 0-max_power)
        actual_power = power * self.max_transmission_power
        
        # Calculate signal power
        signal_power = actual_power * channel_gain
        
        # Simplified interference (in a more complex model, we would consider other MVs)
        interference = 0
        
        # Shannon's formula for channel capacity in Mbps
        rate = self.bandwidth * np.log2(1 + (signal_power / (interference + self.noise_power)))
        return rate
    
    def _calculate_transmission_delay(self, mv_idx, es_idx, power):
        """Calculate delay for transmitting task from MV to ES"""
        rate = self._calculate_transmission_rate(mv_idx, es_idx, power)
        # Prevent division by zero
        rate = max(rate, 0.01)
        delay = self.task_size[mv_idx] / rate
        
        # Add network propagation delay
        total_delay = delay + self.network_delay[mv_idx, es_idx]
        return total_delay
    
    def _calculate_local_processing_time(self, mv_idx, computation_resource):
        """Calculate time to process task locally on MV in seconds"""
        # Ensure minimum resource allocation (normalized from 0-1)
        actual_resource = max(computation_resource, 0.1)
        
        # Get the actual MV computing capacity in cycles per second (Hz)
        # Convert GHz to Hz (multiply by 10^9)
        compute_capacity_hz = self.mv_compute_capacity[mv_idx] * 1e9
        
        # Get required CPU cycles in actual cycles (convert from megacycles)
        required_cycles = self.required_cpu_cycles[mv_idx] * 1e6
        
        # Processing time = cycles / (compute_capacity * resource_allocation)
        process_time = required_cycles / (compute_capacity_hz * actual_resource)
        
        # Add queuing delay (in seconds)
        total_time = process_time + self.mv_local_queue[mv_idx]
        return total_time
    
    def _calculate_es_processing_time(self, mv_idx, es_idx):
        """Calculate time to process task on selected ES in seconds"""
        # Get the ES computing capacity in cycles per second (Hz)
        # Convert GHz to Hz (multiply by 10^9)
        compute_capacity_hz = self.es_compute_capacity[es_idx] * 1e9
        
        # Get required CPU cycles in actual cycles (convert from megacycles)
        required_cycles = self.required_cpu_cycles[mv_idx] * 1e6
        
        # Processing time = cycles / compute_capacity
        process_time = required_cycles / compute_capacity_hz
        
        # Add queuing delay (in seconds)
        # Queue length is stored in normalized form, convert to seconds based on task size
        queue_delay = self.es_queue_length[es_idx] * 2  # Assuming max queue delay is 2 seconds
        
        total_time = process_time + queue_delay
        return total_time
    
    def _calculate_local_energy(self, mv_idx, computation_resource):
        """Calculate energy consumed for local processing in Joules"""
        # Energy consumption model based on the paper
        # E = κ * f^2 * C, where:
        # κ is the energy coefficient (typically around 10^-27)
        # f is the CPU frequency
        # C is the number of CPU cycles
        
        energy_coefficient = 1e-27  # Energy coefficient 
        
        # Actual frequency based on resource allocation (in Hz)
        frequency = self.mv_compute_capacity[mv_idx] * 1e9 * computation_resource
        
        # Required CPU cycles
        cycles = self.required_cpu_cycles[mv_idx] * 1e6
        
        # Calculate energy in Joules
        energy = energy_coefficient * (frequency ** 2) * cycles
        
        return energy
    
    def _calculate_transmission_energy(self, mv_idx, es_idx, power, duration):
        """Calculate energy consumed for transmission in Joules"""
        # Convert normalized power to actual power in Watts
        actual_power = power * self.max_transmission_power
        
        # Energy = power * time (in Joules)
        energy = actual_power * duration
        
        return energy
    
    def _update_queues(self, actions):
        """Update queue lengths based on task offloading decisions"""
        for mv_idx in range(self.num_mvs):
            if self.continuous_action:
                offload_decision = actions[mv_idx, 0]
                # Determine if task is processed locally or offloaded
                if offload_decision < 0.5:  # Local processing
                    # Local queue increases with processing time
                    self.mv_local_queue[mv_idx] += 0.1 * self.required_cpu_cycles[mv_idx]
                else:
                    # Determine which ES to offload to
                    es_idx = min(int(offload_decision * self.num_edge_servers), self.num_edge_servers - 1)
                    # ES queue increases
                    self.es_queue_length[es_idx] += self.queue_update_factor * self.required_cpu_cycles[mv_idx]
            else:
                # For discrete action space
                offload_decision = actions
                if offload_decision == 0:  # Local processing
                    self.mv_local_queue[mv_idx] += 0.1 * self.required_cpu_cycles[mv_idx]
                else:
                    es_idx = offload_decision - 1
                    self.es_queue_length[es_idx] += self.queue_update_factor * self.required_cpu_cycles[mv_idx]
        
        # Natural queue decrease over time
        self.mv_local_queue = np.maximum(0, self.mv_local_queue - 0.05)
        self.es_queue_length = np.maximum(0, self.es_queue_length - 0.03)
        
        # Store current ES queue state for history
        self.es_queue_history.append(self.es_queue_length.copy())
    
    def _get_observation(self):
        """Construct the observation dictionary"""
        # Convert queue history to array
        queue_history = np.array(list(self.es_queue_history))
        
        return {
            'task_size': self.task_size,
            'required_cpu_cycles': self.required_cpu_cycles,
            'task_deadline': self.task_deadline,
            'mv_compute_capacity': self.mv_compute_capacity,
            'mv_local_queue': self.mv_local_queue,
            'mv_energy': self.mv_energy,
            'mv_locations': self.mv_locations,
            'es_compute_capacity': self.es_compute_capacity,
            'es_queue_length': self.es_queue_length,
            'es_locations': self.es_locations,
            'channel_gain': self.channel_gain,
            'network_delay': self.network_delay,
            'es_queue_history': queue_history
        }
    
    def step(self, action):
        """
        Take action and return next state, reward, done, and info
        
        Args:
            action: For continuous action space, shape (num_mvs, 3) where
                   action[mv_idx, 0] = offload decision (0=local, 1=ES)
                   action[mv_idx, 1] = transmission power
                   action[mv_idx, 2] = computation resource
                   
                   For discrete action space, integer 0 (local) or 1 to num_edge_servers
        """
        rewards = np.zeros(self.num_mvs)
        tasks_completed = np.zeros(self.num_mvs, dtype=bool)
        total_energy = np.zeros(self.num_mvs)
        processing_times = np.zeros(self.num_mvs)
        
        # Process actions for each MV
        for mv_idx in range(self.num_mvs):
            if self.continuous_action:
                offload_decision = action[mv_idx, 0]
                power = action[mv_idx, 1]
                computation_resource = action[mv_idx, 2]
                
                # Local processing
                if offload_decision < 0.5:
                    processing_time = self._calculate_local_processing_time(mv_idx, computation_resource)
                    energy_consumption = self._calculate_local_energy(mv_idx, computation_resource)
                    
                    # Check if task meets deadline
                    if processing_time <= self.task_deadline[mv_idx]:
                        tasks_completed[mv_idx] = True
                    else:
                        tasks_completed[mv_idx] = False
                
                # Offload to ES
                else:
                    # Determine which ES to offload to
                    es_idx = min(int(offload_decision * self.num_edge_servers), self.num_edge_servers - 1)
                    
                    # Calculate transmission delay
                    transmission_delay = self._calculate_transmission_delay(mv_idx, es_idx, power)
                    
                    # Calculate ES processing time
                    es_processing_time = self._calculate_es_processing_time(mv_idx, es_idx)
                    
                    # Total processing time
                    processing_time = transmission_delay + es_processing_time
                    
                    # Energy for transmission
                    energy_consumption = self._calculate_transmission_energy(mv_idx, es_idx, power, transmission_delay)
                    
                    # Check if task meets deadline
                    if processing_time <= self.task_deadline[mv_idx]:
                        tasks_completed[mv_idx] = True
                    else:
                        tasks_completed[mv_idx] = False
            
            else:
                # For discrete action space
                offload_decision = action
                power = 0.5  # Default power
                computation_resource = 0.5  # Default computation resource
                
                if offload_decision == 0:  # Local processing
                    processing_time = self._calculate_local_processing_time(mv_idx, computation_resource)
                    energy_consumption = self._calculate_local_energy(mv_idx, computation_resource)
                    
                    if processing_time <= self.task_deadline[mv_idx]:
                        tasks_completed[mv_idx] = True
                    else:
                        tasks_completed[mv_idx] = False
                
                else:  # Offload to ES
                    es_idx = offload_decision - 1
                    
                    transmission_delay = self._calculate_transmission_delay(mv_idx, es_idx, power)
                    es_processing_time = self._calculate_es_processing_time(mv_idx, es_idx)
                    processing_time = transmission_delay + es_processing_time
                    
                    energy_consumption = self._calculate_transmission_energy(mv_idx, es_idx, power, transmission_delay)
                    
                    if processing_time <= self.task_deadline[mv_idx]:
                        tasks_completed[mv_idx] = True
                    else:
                        tasks_completed[mv_idx] = False
            
            # Record metrics
            processing_times[mv_idx] = processing_time
            total_energy[mv_idx] = energy_consumption
            
            # Calculate reward components
            latency_reward = -self.latency_weight * (processing_time / self.task_deadline[mv_idx])
            energy_reward = -self.energy_weight * energy_consumption
            completion_reward = self.completion_weight * (2 * tasks_completed[mv_idx] - 1)  # +0.2 if completed, -0.2 if not
            
            # Combined reward
            rewards[mv_idx] = latency_reward + energy_reward + completion_reward
        
        # Update environment state
        self._update_queues(action)
        self._update_mv_locations()
        self._update_network_conditions()
        self._generate_tasks()
        
        # Get new observation
        next_state = self._get_observation()
        
        # Update history
        self.state_history.append(next_state)
        
        # Update performance metrics
        self.task_completion_history.append(np.mean(tasks_completed))
        self.reward_history.append(np.mean(rewards))
        self.energy_history.append(np.mean(total_energy))
        self.latency_history.append(np.mean(processing_times))
        
        # Create info dict with detailed metrics
        info = {
            'task_completion_rate': np.mean(tasks_completed),
            'avg_processing_time': np.mean(processing_times),
            'avg_energy_consumption': np.mean(total_energy),
            'task_deadlines': self.task_deadline,
            'tasks_completed': tasks_completed,
            'mv_queue_lengths': self.mv_local_queue,
            'es_queue_lengths': self.es_queue_length
        }
        
        # In this environment, episodes don't terminate
        done = False
        
        return next_state, rewards, done, info
    
    def flatten_observation(self, obs):
        """Flatten the observation dictionary into a vector"""
        flattened = []
        for key, value in obs.items():
            if key != 'es_queue_history':  # Handle separately due to different shape
                flattened.append(value.flatten())
        
        # Flatten queue history
        queue_history = obs['es_queue_history'].flatten()
        flattened.append(queue_history)
        
        return np.concatenate(flattened)
    
    def get_state_history(self):
        """Return the history of states"""
        return list(self.state_history)
    
    def get_average_metrics(self):
        """Return average performance metrics"""
        return {
            'avg_task_completion_rate': np.mean(list(self.task_completion_history)),
            'avg_reward': np.mean(list(self.reward_history)),
            'avg_energy': np.mean(list(self.energy_history)),
            'avg_latency': np.mean(list(self.latency_history))
        }
    
    def render(self, mode='human'):
        """Render the environment - simple text output"""
        if mode == 'human':
            print("\n--- MEC Environment State ---")
            print(f"Number of MVs: {self.num_mvs}")
            print(f"Number of ESs: {self.num_edge_servers}")
            
            print("\nPerformance Metrics:")
            metrics = self.get_average_metrics()
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")
            
            print("\nTask Completion Rate:")
            completion_rate = np.mean(list(self.task_completion_history)[-10:]) if self.task_completion_history else 0
            print(f"  Last 10 steps: {completion_rate:.2f}")
            
            print("\nSample MV Status:")
            for i in range(min(3, self.num_mvs)):
                print(f"  MV {i}: Task Size={self.task_size[i][0]:.2f}, " +
                      f"Deadline={self.task_deadline[i][0]:.2f}, " +
                      f"Local Queue={self.mv_local_queue[i][0]:.2f}")
            
            print("\nSample ES Status:")
            for i in range(min(3, self.num_edge_servers)):
                print(f"  ES {i}: Queue Length={self.es_queue_length[i][0]:.2f}, " +
                      f"Compute Capacity={self.es_compute_capacity[i][0]:.2f}")
            
            print("----------------------------\n")
        else:
            super(MECEnvironment, self).render(mode=mode)


# Example usage:
if __name__ == "__main__":
    # Create environment
    env = MECEnvironment(num_mvs=10, num_edge_servers=3, difficulty='normal')
    
    # Reset environment
    obs = env.reset()
    
    # Run a few random steps
    for i in range(5):
        # Random action
        if env.continuous_action:
            action = np.random.random((env.num_mvs, 3))
        else:
            action = np.random.randint(0, env.num_edge_servers + 1)
        
        # Take step
        next_obs, reward, done, info = env.step(action)
        
        # Render environment state
        env.render()
        
        print(f"Step {i+1} - Avg Reward: {np.mean(reward):.4f}")
        print(f"Task Completion Rate: {info['task_completion_rate']:.4f}")
        print(f"Avg Processing Time: {info['avg_processing_time']:.4f}")
        print(f"Avg Energy Consumption: {info['avg_energy_consumption']:.4f}")
        print("-" * 50)