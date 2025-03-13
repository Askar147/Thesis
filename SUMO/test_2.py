import traci
import sumolib
import numpy as np
import pandas as pd
import math
import os

# -------------------------------
# Configuration Variables
# -------------------------------
SUMO_CONFIG = "hokkaido/test.sumocfg"  # Path to your SUMO configuration file (you can start with a simple map)
SIMULATION_DURATION = 60      # Total simulation duration in seconds
TIME_STEP = 1                   # Simulation time step in seconds

# Define base stations based on your network file (astana.net.xml not used yet; using simple map)
# Using positions from the XML snippet:
# Junction "dl": (500, 0)
# Junction "dr": (1700, 0)
# Average of "tl" (500, 190.99) and "tr" (1700, 190.99) → (1100, 190.99)
BASE_STATIONS = [
    {"id": "BS1", "pos": (500, 0), "radius": 600},
    {"id": "BS2", "pos": (1700, 0), "radius": 600},
    {"id": "BS3", "pos": (1100, 190.99), "radius": 600}
]

# Constants for latency calculation:
DATA_SIZE = 1000  # in KB (arbitrary placeholder)
BANDWIDTH = 10    # in Mbps (arbitrary placeholder)
PROPAGATION_FACTOR = 0.001  # converts distance (meters) to delay (seconds)

# -------------------------------
# Helper Functions
# -------------------------------
def euclidean_distance(pos1, pos2):
    return math.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)

def compute_latency(distance, speed):
    """
    Simple latency model:
      - Transmission delay: DATA_SIZE / (BANDWIDTH * 125) seconds (1 Mbps ~125 KB/s)
      - Propagation delay: proportional to the distance
      - Plus a constant processing delay of 0.5 seconds
    """
    transmission_delay = DATA_SIZE / (BANDWIDTH * 125)
    propagation_delay = distance * PROPAGATION_FACTOR
    return transmission_delay + propagation_delay + 0.5

def get_nearest_bs(vehicle_pos):
    """
    Determines the nearest base station to a vehicle.
    Returns a tuple: (base_station_id, distance, in_coverage)
    """
    nearest_bs = None
    min_distance = float("inf")
    in_coverage = False

    for bs in BASE_STATIONS:
        dist = euclidean_distance(vehicle_pos, bs["pos"])
        if dist < min_distance:
            min_distance = dist
            nearest_bs = bs["id"]
            in_coverage = (dist <= bs["radius"])
    return nearest_bs, min_distance, in_coverage

# -------------------------------
# Main Simulation Function
# -------------------------------
def run_simulation():
    # Start SUMO (change 'sumo' to 'sumo-gui' if you want to visualize the simulation)
    sumoBinary = sumolib.checkBinary('sumo')
    traci.start([sumoBinary, "-c", SUMO_CONFIG])
    
    # Dictionary to keep track of the last assigned base station per vehicle (for handover detection)
    last_bs_assignment = {}
    
    # Log for offloading events
    data_log = []

    simulation_step = 0
    while traci.simulation.getMinExpectedNumber() > 0 and simulation_step < SIMULATION_DURATION:
        traci.simulationStep()
        vehicle_ids = traci.vehicle.getIDList()

        for veh in vehicle_ids:
            pos = traci.vehicle.getPosition(veh)  # (x, y)
            speed = traci.vehicle.getSpeed(veh)     # in m/s

            bs_id, distance, in_coverage = get_nearest_bs(pos)
            latency = compute_latency(distance, speed)

            # Determine if handover occurred (if the vehicle's base station has changed)
            previous_bs = last_bs_assignment.get(veh)
            handover = (previous_bs is not None and previous_bs != bs_id)
            last_bs_assignment[veh] = bs_id

            data_log.append({
                "time": simulation_step,
                "vehicle_id": veh,
                "vehicle_x": pos[0],
                "vehicle_y": pos[1],
                "speed": speed,
                "base_station": bs_id,
                "distance_to_bs": distance,
                "in_coverage": in_coverage,
                "latency": latency,
                "handover": handover
            })

        simulation_step += TIME_STEP

    traci.close()

    # Save data log to CSV
    df = pd.DataFrame(data_log)
    output_csv = os.path.join(os.getcwd(), "urban_mobility_dataset.csv")
    df.to_csv(output_csv, index=False)
    print("Simulation finished. Data saved to:", output_csv)

if __name__ == "__main__":
    run_simulation()
