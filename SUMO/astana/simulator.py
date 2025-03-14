import traci
import sumolib
import numpy as np
import pandas as pd
import math
import os
import random

# -------------------------------
# Configuration Variables
# -------------------------------
SUMO_CONFIG = "astana.sumocfg"   # Your Astana SUMO configuration file
SIMULATION_DURATION = 600      # Total simulation duration (seconds)
TIME_STEP = 10                   # Simulation step in seconds

# -------------------------------
# Base Station Definitions
# -------------------------------
# Using the provided coordinates and radii:
BASE_STATIONS = [
    {"id": "BS1", "pos": (875.45, 3032.15), "radius": 330.73},
    {"id": "BS2", "pos": (1549.92, 4944.37), "radius": 414.50},
    {"id": "BS3", "pos": (2289.71, 6792.01), "radius": 463.86},
    {"id": "BS4", "pos": (3135.35, 3166.73), "radius": 455.59},
    {"id": "BS5", "pos": (2722.92, 4764.39), "radius": 388.43},
    {"id": "BS6", "pos": (2120.44, 1237.72), "radius": 354.78},
    {"id": "BS7", "pos": (1296.73, 6717.62), "radius": 463.51},
    {"id": "BS8", "pos": (901.39, 1562.33), "radius": 702.21},
    {"id": "BS9", "pos": (1306.93, 4129.37), "radius": 382.54},
    {"id": "BS10", "pos": (1673.08, 7544.53), "radius": 457.62},
    {"id": "BS11", "pos": (2627.39, 2057.54), "radius": 410.56},
    {"id": "BS12", "pos": (1789.67, 3123.08), "radius": 330.13},
    {"id": "BS13", "pos": (2833.24, 7467.56), "radius": 385.73},
    {"id": "BS14", "pos": (1029.19, 4783.58), "radius": 429.91},
    {"id": "BS15", "pos": (2247.31, 3956.80), "radius": 401.91},
    {"id": "BS16", "pos": (1592.16, 2330.21), "radius": 442.36},
    {"id": "BS17", "pos": (1250.97, 6109.71), "radius": 480.02},
    {"id": "BS18", "pos": (3023.11, 5859.40), "radius": 562.39},
    {"id": "BS19", "pos": (406.44, 2038.44), "radius": 465.31},
    {"id": "BS20", "pos": (3251.80, 2649.83), "radius": 427.21},
    {"id": "BS21", "pos": (3151.02, 3752.47), "radius": 466.78},
    {"id": "BS22", "pos": (1744.09, 6079.44), "radius": 656.05},
    {"id": "BS23", "pos": (3044.59, 6958.27), "radius": 538.07},
    {"id": "BS24", "pos": (2746.31, 5410.53), "radius": 723.09},
    {"id": "BS25", "pos": (730.04, 3680.29), "radius": 505.85},
    {"id": "BS26", "pos": (2200.16, 2228.27), "radius": 383.20},
    {"id": "BS27", "pos": (500.32, 5240.41), "radius": 654.63},
    {"id": "BS28", "pos": (1448.58, 4546.47), "radius": 374.03},
    {"id": "BS29", "pos": (1105.26, 3528.77), "radius": 432.42},
    {"id": "BS30", "pos": (2098.22, 4736.26), "radius": 460.49},
    {"id": "BS31", "pos": (3107.09, 5072.15), "radius": 487.65},
    {"id": "BS32", "pos": (1921.03, 6463.16), "radius": 566.58},
    {"id": "BS33", "pos": (1673.86, 1272.91), "radius": 436.91},
    {"id": "BS34", "pos": (2304.54, 3110.66), "radius": 409.46},
    {"id": "BS35", "pos": (488.48, 3203.94), "radius": 623.90},
    {"id": "BS36", "pos": (481.13, 2640.78), "radius": 406.50},
    {"id": "BS37", "pos": (2492.59, 4390.75), "radius": 417.19},
    {"id": "BS38", "pos": (1294.83, 5360.39), "radius": 547.47},
    {"id": "BS39", "pos": (1713.83, 4223.22), "radius": 449.05},
    {"id": "BS40", "pos": (1131.46, 7420.82), "radius": 551.14},
    {"id": "BS41", "pos": (703.12, 6795.57), "radius": 502.18},
    {"id": "BS42", "pos": (1434.57, 1705.14), "radius": 479.23},
    {"id": "BS43", "pos": (1868.47, 2737.10), "radius": 387.49},
    {"id": "BS44", "pos": (2601.52, 4002.11), "radius": 407.20},
    {"id": "BS45", "pos": (3204.98, 4388.98), "radius": 506.67},
    {"id": "BS46", "pos": (3219.79, 7653.39), "radius": 389.98},
    {"id": "BS47", "pos": (2430.70, 7331.99), "radius": 423.44},
    {"id": "BS48", "pos": (1831.29, 7168.56), "radius": 478.47},
    {"id": "BS49", "pos": (3096.64, 2221.90), "radius": 389.94},
    {"id": "BS50", "pos": (2673.97, 2726.58), "radius": 482.06},
    {"id": "BS51", "pos": (902.58, 4199.47), "radius": 414.17},
    {"id": "BS52", "pos": (1303.38, 2843.25), "radius": 362.12},
    {"id": "BS53", "pos": (3144.03, 1707.56), "radius": 700.58},
    {"id": "BS54", "pos": (2706.96, 6406.38), "radius": 667.85},
    {"id": "BS55", "pos": (691.17, 6195.84), "radius": 547.95},
    {"id": "BS56", "pos": (2202.93, 5163.20), "radius": 652.93},
    {"id": "BS57", "pos": (430.00, 4449.78), "radius": 627.49},
    {"id": "BS58", "pos": (1350.55, 3244.54), "radius": 370.29},
    {"id": "BS59", "pos": (2686.93, 1353.64), "radius": 671.87},
    {"id": "BS60", "pos": (1865.22, 3654.01), "radius": 461.76},
    {"id": "BS61", "pos": (2629.40, 3503.17), "radius": 458.40},
    {"id": "BS62", "pos": (365.49, 1402.51), "radius": 515.42},
    {"id": "BS63", "pos": (2156.81, 1777.53), "radius": 457.11},
    {"id": "BS64", "pos": (1076.36, 2344.15), "radius": 477.28}
]

# -------------------------------
# Advanced Latency Model Parameters
# -------------------------------
# These parameters are placeholders and should be tuned to match real data.
PROCESSING_DELAY = 0.5         # seconds constant processing delay
PROPAGATION_FACTOR = 0.001     # seconds per meter
ALPHA_INTERFERENCE = 0.005     # interference delay per nearby vehicle (seconds)
NOISE_FACTOR = 0.1             # random noise in seconds

# Offloading data size mapping per scenario (0-9)
SCENARIO_DATA_SIZES = {
    0: 1000,
    1: 1200,
    2: 1500,
    3: 1800,
    4: 2000,
    5: 2200,
    6: 2500,
    7: 2800,
    8: 3000,
    9: 3500
}

# -------------------------------
# Helper Functions
# -------------------------------
def euclidean_distance(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

def compute_transmission_delay(data_size):
    # Assume BANDWIDTH = 10 Mbps (10 * 125 = 1250 KB/s)
    return data_size / (10 * 125)

def count_vehicles_in_coverage(vehicle_pos, radius):
    all_vehicles = traci.vehicle.getIDList()
    count = 0
    for veh in all_vehicles:
        pos = traci.vehicle.getPosition(veh)
        if euclidean_distance(vehicle_pos, pos) <= radius:
            count += 1
    return count

def compute_latency(distance, vehicle_pos, data_size):
    tx_delay = compute_transmission_delay(data_size)
    prop_delay = distance * PROPAGATION_FACTOR
    interference_count = count_vehicles_in_coverage(vehicle_pos, 300)
    interference_delay = ALPHA_INTERFERENCE * interference_count
    noise = random.uniform(0, NOISE_FACTOR)
    return tx_delay + prop_delay + interference_delay + PROCESSING_DELAY + noise

def compute_energy(latency):
    # Placeholder energy model; later, integrate detailed performance measurements.
    return latency * 0.2

def get_nearest_bs(vehicle_pos):
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

def assign_scenario():
    return random.randint(0, 9)

# -------------------------------
# Main Simulation Function
# -------------------------------
def run_simulation():
    random.seed(42)
    sumo_args = [sumolib.checkBinary('sumo'), "--ignore-route-errors", "-c", SUMO_CONFIG]
    traci.start(sumo_args)

    
    # For handover detection, store last base station per vehicle
    last_bs_assignment = {}
    data_log = []
    
    simulation_step = 0
    while traci.simulation.getMinExpectedNumber() > 0 and simulation_step < SIMULATION_DURATION:
        traci.simulationStep()
        vehicle_ids = traci.vehicle.getIDList()
        
        for veh in vehicle_ids:
            pos = traci.vehicle.getPosition(veh)  # (x, y)
            speed = traci.vehicle.getSpeed(veh)
            bs_id, distance, in_coverage = get_nearest_bs(pos)
            
            # Assign scenario ID and determine corresponding data size (KB)
            scenario_id = assign_scenario()
            data_size = SCENARIO_DATA_SIZES.get(scenario_id, 1000)
            
            latency = compute_latency(distance, pos, data_size)
            energy = compute_energy(latency)
            
            prev_bs = last_bs_assignment.get(veh)
            handover = (prev_bs is not None and prev_bs != bs_id)
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
                "energy": energy,
                "scenario_id": scenario_id,
                "data_size": data_size
            })
            
        simulation_step += TIME_STEP
    
    traci.close()
    
    df = pd.DataFrame(data_log)
    output_csv = os.path.join(os.getcwd(), "urban_mobility_advanced_dataset.csv")
    df.to_csv(output_csv, index=False)
    print("Simulation finished. Data saved to:", output_csv)

if __name__ == "__main__":
    run_simulation()
