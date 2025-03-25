obs = {
    'task_size': np.array([750.0]),  # KB
    'required_cpu_cycles': np.array([500.0]),  # Megacycles
    'task_deadline': np.array([8.0]),  # Seconds
    'task_scenario_id': np.array([3.0]),  # Scenario ID
    
    'vehicle_pos_x': np.array([1250.5]),  # X position
    'vehicle_pos_y': np.array([3780.2]),  # Y position
    'vehicle_speed': np.array([12.5]),  # m/s
    
    'distance_to_bs': np.array([320.7]),  # meters
    'bs_queue_length': np.array([5.0]),  # Number of tasks in queue
    
    'active_nodes': np.array([14.0]),  # Number of active nodes
    'node_loads': np.array([0.25, 0.5, 0.75, 0.0, 0.25, ...]),  # Load for each node
    'node_active_status': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]),
    
    'historical_loads': np.array([
        [0.25, 0.5, 0.75, 0.0, ...],  # Most recent
        [0.2, 0.4, 0.7, 0.0, ...],    # Previous
        # ... more history
    ]),
    
    'link_quality': np.array([0.85]),  # Normalized SINR (0-1)
    'energy_efficiency': np.array([3.2])  # Energy efficiency ratio
}

def prepare_observation_for_model(measured_params):
    # Create full observation with defaults
    obs = {
        # Parameters you can measure
        'task_size': np.array([measured_params.get('task_size', 0)]),
        'required_cpu_cycles': np.array([measured_params.get('required_cpu_cycles', 0)]),
        'task_deadline': np.array([measured_params.get('task_deadline', 10)]),
        'task_scenario_id': np.array([measured_params.get('task_scenario_id', 1)]),
        
        # Default values for parameters you can't measure
        'vehicle_pos_x': np.array([0]),
        'vehicle_pos_y': np.array([0]),
        'vehicle_speed': np.array([10]),  # Default middle-range speed
        
        # Other parameters
        'distance_to_bs': np.array([measured_params.get('distance_to_bs', 500)]),
        'bs_queue_length': np.array([measured_params.get('bs_queue_length', 0)]),
        # ...and so on
    }
    return obs