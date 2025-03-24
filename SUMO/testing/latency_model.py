import numpy as np
import math
import random
from collections import deque

class LatencyModel:
    
    def __init__(self, frequency_band=2.4, bandwidth=20, noise_floor=-95):
        """
            frequency_band: Carrier frequency in GHz (default: 2.4 GHz)
            bandwidth: Channel bandwidth in MHz (default: 20 MHz)
            noise_floor: Noise floor in dBm (default: -95 dBm)
        """
        self.frequency = frequency_band 
        self.bandwidth = bandwidth * 1e6  # Convert MHz to Hz
        self.noise_floor_dBm = noise_floor
        self.noise_floor_mW = 10**(noise_floor/10)  # Convert dBm to mW
        
        # Path loss model parameters (urban environment)
        self.path_loss_exponent = 3.5  
        
        # BS parameters
        self.bs_tx_power_dBm = 23  # BS transmission power in dBm
        self.bs_tx_power_mW = 10**(self.bs_tx_power_dBm/10)  # Convert to mW
        
        # Processing parameters
        self.base_processing_factor = 1e6  # Cycles per megabyte factor
        
        # Task data size parameters (based on work units)
        self.work_units = {
            1: 15,  # Linear Chain
            2: 5,   # Fork and Merge
            3: 5,   # Parallel Tasks
            4: 7,   # Sequential with Branching
            5: 8,   # Complex Merge
            6: 6,   # Double Fork
            7: 9,   # Loop Replanning
            8: 30,  # High Parallelism
            9: 12,  # Mixed Workload
            10: 20  # Extended DAG
        }
        self.upstream_unit_size = 65  # KB per work unit for task data
        self.downstream_unit_size = 10  # KB per work unit for result data
        
        # Historical data
        self.latency_history = deque(maxlen=1000)
        self.datarate_history = deque(maxlen=1000)
        
        # Cache for repeated calculations
        self.path_loss_cache = {}
    
    def get_task_data_size(self, scenario_id, is_upstream=True):
        # Default to scenario 1 if scenario_id is not in the dictionary
        units = self.work_units.get(scenario_id, 15)
        
        unit_size = self.upstream_unit_size if is_upstream else self.downstream_unit_size
        
        return units * unit_size
    
    def calculate_path_loss(self, distance):
        cache_key = round(distance, 1)
        
        if cache_key in self.path_loss_cache:
            return self.path_loss_cache[cache_key]
        
        # Calculate free space path loss at reference distance (1m)
        free_space_path_loss_ref = 20 * np.log10(4 * np.pi * self.frequency * 1e9 / 3e8)
        
        # Log-distance path loss model
        path_loss = free_space_path_loss_ref + 10 * self.path_loss_exponent * np.log10(max(1.0, distance))
        
        self.path_loss_cache[cache_key] = path_loss
        
        return path_loss
    
    def calculate_sinr(self, distance, interfering_distances=None):
        # Calculate received signal power
        path_loss = self.calculate_path_loss(distance)
        rx_power_dBm = self.bs_tx_power_dBm - path_loss
        rx_power_mW = 10**(rx_power_dBm/10)
        
        # Calculate interference if present
        interference_mW = 0
        if interfering_distances:
            for interfering_distance in interfering_distances:
                # Skip if same as serving BS
                if abs(interfering_distance - distance) < 1:
                    continue
                    
                int_path_loss = self.calculate_path_loss(interfering_distance)
                int_power_dBm = self.bs_tx_power_dBm - int_path_loss
                int_power_mW = 10**(int_power_dBm/10)
                interference_mW += int_power_mW
        
        # Calculate SINR
        sinr_linear = rx_power_mW / (self.noise_floor_mW + interference_mW)
        sinr_dB = 10 * np.log10(sinr_linear)
        
        return sinr_dB
    
    def calculate_data_rate(self, sinr_dB):
        # Convert from dB to linear scale
        sinr_linear = 10**(sinr_dB/10)
        
        # Shannon's formula with a practical efficiency factor (0.75)
        spectral_efficiency = min(8.0, 0.75 * np.log2(1 + sinr_linear))  # Cap at 8 bits/symbol
        
        # Calculate data rate
        data_rate = self.bandwidth * spectral_efficiency
        
        self.datarate_history.append(data_rate)
        
        return data_rate
    
    def get_transmission_latency(self, scenario_id, distance, is_upstream=True, interfering_distances=None):
        # Calculate data size from scenario ID
        data_size_kb = self.get_task_data_size(scenario_id, is_upstream)
        
        # Calculate SINR
        sinr_dB = self.calculate_sinr(distance, interfering_distances)
        
        # Calculate data rate
        data_rate = self.calculate_data_rate(sinr_dB)
        
        # Convert data size from KB to bits
        data_size_bits = data_size_kb * 8 * 1024
        
        # Calculate latency (with protocol overhead factor of 1.2)
        latency = (data_size_bits * 1.2) / data_rate if data_rate > 0 else float('inf')
        
        # Apply small random fluctuation to model network jitter (±5%)
        jitter_factor = np.random.uniform(0.95, 1.05)
        latency *= jitter_factor
        
        return latency
    
    def get_processing_latency(self, cpu_cycles, node_capacity, node_load):
        # Base processing time
        base_processing_time = cpu_cycles / node_capacity
        
        # Load factor affects processing rate (linear model for simplicity)
        load_factor = 1.0 + node_load
        
        # Total processing time
        processing_time = base_processing_time * load_factor
        
        return processing_time
    
    def get_required_cpu_cycles(self, scenario_id):
        base_cycles_per_unit = 25
        
        work_units = self.work_units.get(scenario_id, 15) 
        
        variability = random.uniform(0.8, 1.2)
        
        required_cycles = work_units * base_cycles_per_unit * variability
        
        return required_cycles


    def get_total_latency(self, scenario_id, cpu_cycles, distance, 
                          node_capacity, node_load, interfering_distances=None):
        # Uplink transmission latency
        uplink_latency = self.get_transmission_latency(
            scenario_id, 
            distance, 
            is_upstream=True, 
            interfering_distances=interfering_distances
        )
        
        # Processing latency
        processing_latency = self.get_processing_latency(
            cpu_cycles, 
            node_capacity, 
            node_load
        )
        
        # Downlink transmission latency (for results)
        downlink_latency = self.get_transmission_latency(
            scenario_id, 
            distance, 
            is_upstream=False, 
            interfering_distances=interfering_distances
        )
        
        # Total latency
        total_latency = uplink_latency + processing_latency + downlink_latency
        
        # Store in history
        self.latency_history.append(total_latency)
        
        return total_latency, (uplink_latency, processing_latency, downlink_latency)
    
    def get_average_latency(self, window=None):
        if not self.latency_history:
            return 0.0
            
        if window is None or window >= len(self.latency_history):
            return np.mean(self.latency_history)
        else:
            return np.mean(list(self.latency_history)[-window:])
    
    def get_link_quality(self, distance, interfering_distances=None):
        sinr_dB = self.calculate_sinr(distance, interfering_distances)
        
        if sinr_dB >= 25:
            quality = "Excellent"
        elif sinr_dB >= 20:
            quality = "Very Good"
        elif sinr_dB >= 15:
            quality = "Good"
        elif sinr_dB >= 10:
            quality = "Fair"
        elif sinr_dB >= 5:
            quality = "Poor"
        else:
            quality = "Very Poor"
            
        return quality, sinr_dB
    
    def reset_history(self):
        """Reset all history data"""
        self.latency_history.clear()
        self.datarate_history.clear()