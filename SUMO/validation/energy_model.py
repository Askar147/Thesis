import numpy as np
import pandas as pd

class EnergyModel:
    """Energy consumption model for VEC tasks based on historical data"""
    
    def __init__(self, csv_path=None):
        """
        Initialize the energy model, optionally loading data from a CSV file
        
        Args:
            csv_path (str): Path to the CSV file with energy consumption data
        """
        # Container ID to scenario mapping
        self.container_to_scenario = {
            "4855b4006e87": 1,  # Scenario 1
            "04463a03fa97": 2,  # Scenario 2
            "e028f7bc7379": 3,  # Scenario 3
            "cd32d74a79a2": 4,  # Scenario 4
            "20b09d247631": 5,  # Scenario 5
            "34fa6c934d4c": 6,  # Scenario 6
            "dceb7961b8aa": 7,  # Scenario 7
            "095d68607caf": 8,  # Scenario 8
            "517431f3aa6d": 9,  # Scenario 9
            "68fecb8e6b49": 10   # Scenario 10
        }
        
        # Default energy values per scenario (Watts)
        self.default_energy_consumption = {
            1: 5.0,   # Scenario 1
            2: 5.5,   # Scenario 2
            3: 6.0,   # Scenario 3
            4: 6.5,   # Scenario 4
            5: 7.0,   # Scenario 5
            6: 7.5,   # Scenario 6
            7: 8.0,   # Scenario 7
            8: 8.5,   # Scenario 8
            9: 9.0,   # Scenario 9
            10: 9.5    # Scenario 10
        }
        
        # Energy model parameters
        self.scenario_energy = {}
        self.scenario_duration = {}
        
        # Load data if provided
        if csv_path:
            self.load_data(csv_path)
    
    def load_data(self, csv_path):
        """
        Load and process energy consumption data from CSV file
        
        Args:
            csv_path (str): Path to the CSV file with energy consumption data
        """
        try:
            # Read CSV data
            df = pd.read_csv(csv_path)
            
            # Process data for each container/scenario
            for container_id, scenario_id in self.container_to_scenario.items():
                # Filter data for this container
                container_data = df[df['container_id'] == container_id]
                
                if len(container_data) > 0:
                    # Calculate average power consumption (pom_5v_in in Watts)
                    avg_power = container_data['pom_5v_in'].mean()
                    
                    # Calculate task duration (in seconds)
                    # Assuming d_timestamp is in a format that can be converted to datetime
                    container_data['timestamp'] = pd.to_datetime(container_data['d_timestamp'])
                    duration = (container_data['timestamp'].max() - 
                               container_data['timestamp'].min()).total_seconds()
                    
                    # Store results
                    self.scenario_energy[scenario_id] = avg_power
                    self.scenario_duration[scenario_id] = duration
                    
                    print(f"Loaded data for Scenario {scenario_id}: "
                          f"Avg. Power: {avg_power:.2f}W, Duration: {duration:.2f}s")
                else:
                    print(f"No data found for container {container_id} (Scenario {scenario_id})")
                    # Use default values
                    self.scenario_energy[scenario_id] = self.default_energy_consumption[scenario_id]
                    self.scenario_duration[scenario_id] = 1.0  # Default duration
        
        except Exception as e:
            print(f"Error loading energy data from CSV: {e}")
            print("Using default energy values")
            self.scenario_energy = self.default_energy_consumption.copy()
            self.scenario_duration = {sid: 1.0 for sid in self.default_energy_consumption.keys()}
    
    def get_energy_consumption(self, scenario_id, duration=None):
        """
        Get the energy consumption for a specific scenario
        
        Args:
            scenario_id (int): ID of the scenario (1-10)
            duration (float, optional): Duration of the task in seconds.
                                       If None, uses average duration from data.
        
        Returns:
            float: Energy consumption in Joules (Watts * seconds)
        """
        # Get power consumption for the scenario (default to a conservative value if not found)
        power = self.scenario_energy.get(scenario_id, 10.0)  # Watts
        
        # Use provided duration or average from data
        if duration is None:
            duration = self.scenario_duration.get(scenario_id, 1.0)  # seconds
        
        # Calculate energy (Power * Time)
        energy = power * duration  # Joules
        
        return energy
    
    def get_average_power(self, scenario_id):
        """
        Get the average power consumption for a specific scenario
        
        Args:
            scenario_id (int): ID of the scenario (1-10)
        
        Returns:
            float: Average power consumption in Watts
        """
        return self.scenario_energy.get(scenario_id, self.default_energy_consumption.get(scenario_id, 5.0))

# Example usage
if __name__ == "__main__":
    # Create model and load data
    model = EnergyModel("merged_dag1.csv")
    
    # Get energy consumption for scenario 1
    energy = model.get_energy_consumption(1)
    print(f"Energy consumption for Scenario 1: {energy:.2f} Joules")
    
    # Get energy with custom duration
    energy = model.get_energy_consumption(1, duration=2.5)
    print(f"Energy consumption for Scenario 1 (2.5s): {energy:.2f} Joules")