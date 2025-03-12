import traci
import sumolib
import numpy as np
import pandas as pd

def run_simulation(config_file, output_csv):
    # Start SUMO in command-line mode (or use "sumo-gui" for visualization)
    sumoBinary = sumolib.checkBinary('sumo')
    traci.start([sumoBinary, "-c", config_file])
    
    # Prepare data logging
    data_log = []
    
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        
        # For each vehicle, compute a simple latency metric based on distance to a fixed edge node (example)
        vehicles = traci.vehicle.getIDList()
        for veh in vehicles:
            pos = traci.vehicle.getPosition(veh)
            # Example: fixed edge node position (x_edge, y_edge)
            x_edge, y_edge = 500, 500
            distance = np.linalg.norm(np.array(pos) - np.array([x_edge, y_edge]))
            # Example latency: distance divided by speed (add some constant delay)
            speed = traci.vehicle.getSpeed(veh)
            latency = distance / (speed + 1) + 0.5  # simple model
            
            # Placeholder energy consumption (can be adjusted later)
            energy = latency * 0.2  # arbitrary factor
            
            data_log.append({
                "time": step,
                "vehicle": veh,
                "x": pos[0],
                "y": pos[1],
                "distance": distance,
                "speed": speed,
                "latency": latency,
                "energy": energy
            })
        
        step += 1

    # Convert log to DataFrame and save
    df = pd.DataFrame(data_log)
    df.to_csv(output_csv, index=False)
    traci.close()

if __name__ == "__main__":
    run_simulation("hokkaido/test.sumocfg", "simulation_output.csv")
