def ff_dqn_inference(agent, obs):
    # Flatten the observation
    state = agent.flatten_observation(obs)
    
    # Get action without exploration (epsilon=0)
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        q_values = agent.q_network(state_tensor)
        action = q_values.cpu().argmax().item()
    
    return action

def transformer_inference(agent, obs):
    # Flatten the observation
    state = flatten_observation(obs)
    
    # Add state to history
    agent.state_history.append(state)
    
    # Pad history if needed
    if len(agent.state_history) < agent.seq_length:
        padding = [state] * (agent.seq_length - len(agent.state_history))
        seq_states = padding + list(agent.state_history)
    else:
        seq_states = list(agent.state_history)
    
    # Get action without exploration
    with torch.no_grad():
        seq_tensor = torch.FloatTensor(np.array(seq_states)).to(agent.device)
        q_values, _ = agent.policy_net(seq_tensor)
        action = q_values.argmax().item()
    
    return action