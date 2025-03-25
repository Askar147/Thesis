def load_ff_dqn_model(model_path, env):
    # Calculate state size
    temp_obs = env.reset()
    from ff_dqn_agent import DQNAgent, flatten_observation
    # Create temporary agent just to use flatten_observation
    temp_agent = DQNAgent(0, 0)
    flattened = temp_agent.flatten_observation(temp_obs)
    state_size = len(flattened)
    
    # Create agent with correct dimensions
    agent = DQNAgent(state_size, env.action_space.n)
    
    # Load weights from saved model
    agent.load_model(model_path)
    
    return agent

def load_transformer_model(model_path, env, seq_length=16, d_model=128, nhead=4, num_layers=3):
    # Calculate state size
    temp_obs = env.reset()
    from transformer_agent import flatten_observation, VECTransformerAgent
    flattened = flatten_observation(temp_obs)
    state_size = len(flattened)
    
    # Create agent with correct dimensions
    agent = VECTransformerAgent(state_size, env.action_space.n)
    
    # The transformer parameters are set during agent initialization
    # If you need to modify them, you would need to implement additional parameters
    # in the VECTransformerAgent constructor
    
    # Load weights from saved model
    agent.load_model(model_path)
    
    return agent