import numpy as np
import pandas as pd
import re
from collections import deque

class EnhancedEnergyModel:
    """Enhanced energy consumption model for VEC tasks based on real-world data"""
    
    def __init__(self, csv_path=None):
        # Container ID to scenario mapping
        self.container_to_scenario = {
            "4855b4006e87": 1,  # Linear Chain Scenario (15 sequential tasks)
            "04463a03fa97": 2,  # Fork and Merge Scenario
            "e028f7bc7379": 3,  # Parallel Tasks Scenario (5 parallel tasks)
            "cd32d74a79a2": 4,  # Sequential with Branching Scenario
            "20b09d247631": 5,  # Complex Merge Scenario
            "34fa6c934d4c": 6,  # Double Fork Scenario
            "dceb7961b8aa": 7,  # Loop Replanning Scenario
            "095d68607caf": 8,  # High Parallelism Scenario (30 parallel tasks)
            "517431f3aa6d": 9,  # Mixed Workload Scenario
            "68fecb8e6b49": 10   # Extended DAG Scenario
        }
        
        
        # Default wake-up energy and idle power
        self.wake_up_energy = 5.0 
        self.idle_power = 2.0  
        
        # Energy model parameters
        self.scenario_energy = {}  # Average power consumption per scenario (W)
        self.scenario_duration = {}  # Average duration per scenario (s)
        self.scenario_load_impact = {}  # Load impact factor per scenario
        self.scenario_energy_variance = {}  # Energy consumption variance
        
        # Energy usage history for analytics
        self.energy_history = deque(maxlen=100)
        
        self.load_data(csv_path)

    
    def parse_cpu_array(self, cpu_str):
        try:
            # Extract values using regex
            pattern = r'(\d+)%@\d+'
            matches = re.findall(pattern, cpu_str)
            
            # Convert to float percentages (0-1)
            utilization = [float(match) / 100.0 for match in matches]
            
            return utilization
        except:
            # Return default values if parsing fails
            return [0.0, 0.0, 0.0, 0.0]
    
    def load_data(self, csv_path):
        df = pd.read_csv(csv_path)
        
        # Process data for each container/scenario
        for container_id, scenario_id in self.container_to_scenario.items():
            container_data = df[df['container_id'] == container_id]
            

            avg_power = container_data['pom_5v_in'].mean()
            power_variance = container_data['pom_5v_in'].var()
            
            container_data['timestamp'] = pd.to_datetime(container_data['d_timestamp'])
            duration = (container_data['timestamp'].max() - 
                        container_data['timestamp'].min()).total_seconds()
    
            if 't_cpu(arr[4])' in container_data.columns:
                container_df = container_data.copy()
                
                container_df['avg_cpu_util'] = np.nan
                
                for idx, cpu_str in container_df.loc[container_df['t_cpu(arr[4])'].notna(), 't_cpu(arr[4])'].items():
                    util_values = self.parse_cpu_array(cpu_str)
                    container_df.at[idx, 'avg_cpu_util'] = np.mean(util_values)
                
                valid_rows = container_df[['pom_5v_in', 'avg_cpu_util']].dropna()
                if len(valid_rows) > 5:  # Ensure we have enough data points
                    cpu_power_corr = valid_rows.corr().iloc[0, 1]
                    load_impact = max(0.05, abs(cpu_power_corr) * 0.5)  
                else:
                    load_impact = 0.2  # Default
            else:
                load_impact = 0.2  # Default
            
                
            self.idle_power = container_data['pom_5v_in'].quantile(0.05)  
            
            self.scenario_energy[scenario_id] = avg_power
            self.scenario_duration[scenario_id] = duration
            self.scenario_load_impact[scenario_id] = load_impact
            self.scenario_energy_variance[scenario_id] = power_variance
            
            print(f"Loaded data for Scenario {scenario_id} ({self._get_scenario_name(scenario_id)}): "
                    f"Avg. Power: {avg_power:.2f}W, Duration: {duration:.2f}s, "
                    f"Load Impact: {load_impact:.2f}")

        
    
    def get_energy_consumption(self, scenario_id, duration=None, node_load=0.0):
        
        
        if duration is None:
            duration = self.scenario_duration.get(scenario_id, 1.0) 

        base_power = self.scenario_energy.get(scenario_id, 5.0)
        load_impact = self.scenario_load_impact.get(scenario_id, 0.2)
        load_factor = 1.0 + (load_impact * node_load)
        
        adjusted_power = base_power * load_factor
        
        variance = self.scenario_energy_variance.get(scenario_id, 0.5)
        if variance > 0:
            # Add gaussian noise scaled by variance
            noise_scale = np.sqrt(variance) * 0.1  
            noise = np.random.normal(0, noise_scale)
            adjusted_power = max(self.idle_power, adjusted_power + noise)
        
        energy = adjusted_power * duration  # Joules
        
        self.energy_history.append(energy)
        
        return energy
    
    def get_wake_up_energy(self, node_type="standard", cold_start=False):
        base_wake_energy = self.wake_up_energy
        
        if node_type == "high_performance":
            type_factor = 2.0  
        elif node_type == "low_power":
            type_factor = 0.5  
        else:  # standard
            type_factor = 1.0
            
        # Adjust based on cold vs warm start
        cold_factor = 1.5 if cold_start else 1.0
        
        return base_wake_energy * type_factor * cold_factor
    
    def get_idle_power(self, node_type="standard"):
        base_idle = self.idle_power
        
        if node_type == "high_performance":
            type_factor = 2.5  
        elif node_type == "low_power":
            type_factor = 0.4 
        else:  # standard
            type_factor = 1.0
        
        return base_idle * type_factor
    
    def get_energy_efficiency_ratio(self, scenario_id):
        # Define a work metric for each scenario
        work_units = {
            1: 15,    # Number of tasks within a scenario
            2: 5,     
            3: 5,     
            4: 7,     
            5: 8,     
            6: 6,     
            7: 9,     
            8: 30,  
            9: 12,    
            10: 20    
        }
        
        work = work_units.get(scenario_id, 10)
        energy = self.get_energy_consumption(scenario_id)
        
        return work / max(0.1, energy) 
    
    def get_average_energy(self, window=None):
        if not self.energy_history:
            return 0.0
            
        if window is None or window >= len(self.energy_history):
            return np.mean(self.energy_history)
        else:
            return np.mean(list(self.energy_history)[-window:])
    
    def _get_scenario_name(self, scenario_id):
        """Get the human-readable name of a scenario"""
        scenario_names = {
            1: "Linear Chain",
            2: "Fork and Merge",
            3: "Parallel Tasks",
            4: "Sequential with Branching",
            5: "Complex Merge",
            6: "Double Fork",
            7: "Loop Replanning",
            8: "High Parallelism",
            9: "Mixed Workload",
            10: "Extended DAG"
        }
        return scenario_names.get(scenario_id, f"Scenario {scenario_id}")

