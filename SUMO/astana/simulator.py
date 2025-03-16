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
SUMO_CONFIG = "astana.sumocfg"  # Your Astana map configuration file
SIMULATION_DURATION = 60       # Total simulation duration (seconds)
TIME_STEP = 1                  # Simulation time step (seconds)
QUEUE_PROCESS_INTERVAL = 5     # Process the queue every 5 simulation steps
MAX_QUEUE_LENGTH = 50          # Maximum number of tasks allowed in a base station queue

# -------------------------------
# Base Station Definitions (using provided coordinates)
# -------------------------------
# For demonstration, we use a simplified list.
BASE_STATIONS = [
    {"id": "BS1", "pos": (757.97, 1887.61), "radius": 1213.40},
    {"id": "BS2", "pos": (1195.33, 4681.07), "radius": 1564.12},
    {"id": "BS3", "pos": (1416.11, 7284.40), "radius": 1396.53},
    {"id": "BS4", "pos": (2681.28, 4058.17), "radius": 1019.69},
    {"id": "BS5", "pos": (2337.28, 1727.15), "radius": 1348.14},
    {"id": "BS6", "pos": (2821.19, 2909.84), "radius": 1188.25},
    {"id": "BS7", "pos": (2682.10, 7085.80), "radius": 1288.92},
    {"id": "BS8", "pos": (2753.03, 5058.93), "radius": 1297.73},
    {"id": "BS9", "pos": (1493.40, 6205.50), "radius": 1585.43},
    {"id": "BS10", "pos": (1206.33, 3227.31), "radius": 1302.67},
]

# -------------------------------
# Node and BaseStation Parameters
# -------------------------------
NODES_PER_BS = 20
MIN_ACTIVE_NODES = 10  # Minimum active nodes per BS

# -------------------------------
# Latency Model Parameters
# -------------------------------
PROCESSING_DELAY = 0.5         # Constant processing delay (s)
PROPAGATION_FACTOR = 0.001     # s per meter
ALPHA_INTERFERENCE = 0.005     # Interference delay per nearby vehicle (s)
NOISE_FACTOR = 0.1             # Random noise (s)

# -------------------------------
# Data Sizes (KB) and Deadlines
# -------------------------------
SCENARIO_DATA_SIZES = {
    0: 1000, 1: 1200, 2: 1500, 3: 1800, 4: 2000,
    5: 2200, 6: 2500, 7: 2800, 8: 3000, 9: 3500
}
SCENARIO_RESULT_SIZES = {
    0: 200, 1: 220, 2: 250, 3: 280, 4: 300,
    5: 320, 6: 350, 7: 380, 8: 400, 9: 450
}

def generate_deadline():
    return random.uniform(6, 10)

# -------------------------------
# Helper Functions
# -------------------------------
def euclidean_distance(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

def compute_transmission_delay(data_size):
    # Assume 10 Mbps => 1250 KB/s.
    return data_size / (10 * 125)

def count_vehicles_in_coverage(vehicle_pos, radius):
    count = 0
    for veh in traci.vehicle.getIDList():
        pos = traci.vehicle.getPosition(veh)
        if euclidean_distance(vehicle_pos, pos) <= radius:
            count += 1
    return count

def compute_latency(distance, vehicle_pos, data_size):
    tx_delay = compute_transmission_delay(data_size)
    prop_delay = distance * PROPAGATION_FACTOR
    interference = ALPHA_INTERFERENCE * count_vehicles_in_coverage(vehicle_pos, 300)
    noise = random.uniform(0, NOISE_FACTOR)
    return tx_delay + prop_delay + interference + PROCESSING_DELAY + noise

def compute_return_latency(distance, vehicle_pos, result_data_size):
    tx_return = compute_transmission_delay(result_data_size)
    prop_return = distance * PROPAGATION_FACTOR
    noise = random.uniform(0, NOISE_FACTOR / 2)
    return tx_return + prop_return + noise

def compute_energy(latency):
    return latency * 0.2  # Placeholder energy model

def get_nearest_bs(vehicle_pos):
    nearest_bs, min_dist, in_cov = None, float("inf"), False
    for bs in BASE_STATIONS:
        dist = euclidean_distance(vehicle_pos, bs["pos"])
        if dist < min_dist:
            min_dist, nearest_bs = dist, bs["id"]
            in_cov = (dist <= bs["radius"])
    return nearest_bs, min_dist, in_cov

def assign_scenario():
    return random.randint(0, 9)
# Maximum number of tasks a node can run concurrently.
MAX_CONCURRENT_TASKS = 4

IDLE_THRESHOLD = 10            # If a node is idle for more than 10 simulation seconds, it is turned off

NODES_PER_BS = 20
MIN_ACTIVE_NODES = 10  # Minimum nodes that remain active per BS

# (Other constants such as latency model parameters, data sizes, etc. remain unchanged)

# -------------------------------
# Revised Node and BaseStation Classes
# -------------------------------

class Node:
    def __init__(self, node_id, active=True):
        self.node_id = node_id
        self.active_tasks = []  # List of task finish times (in simulation time)
        self.active = active    # True if node is active (online); False if idle/offline
        self.idle_since = None  # Timestamp when node became idle (if no active tasks)

    def update_tasks(self, current_time):
        # Remove finished tasks
        self.active_tasks = [end_time for end_time in self.active_tasks if current_time < end_time]
        # Update idle_since: if no tasks are running and not already marked
        if len(self.active_tasks) == 0 and self.idle_since is None:
            self.idle_since = current_time
        elif len(self.active_tasks) > 0:
            self.idle_since = None

    def available_slots(self, current_time):
        self.update_tasks(current_time)
        return MAX_CONCURRENT_TASKS - len(self.active_tasks)

    def is_available(self, current_time):
        # Node is available if it is active and has at least one free slot.
        self.update_tasks(current_time)
        return self.active and (self.available_slots(current_time) > 0)

    def assign_task(self, current_time, processing_time):
        finish_time = current_time + processing_time
        self.active_tasks.append(finish_time)
        # When a task is assigned, the node is no longer idle.
        self.idle_since = None
        # (Optional) Logging:
        # print(f"Node {self.node_id} now running {len(self.active_tasks)}/{MAX_CONCURRENT_TASKS} tasks.")

class BaseStation:
    def __init__(self, bs_info):
        self.bs_id = bs_info["id"]
        self.pos = bs_info["pos"]
        self.radius = bs_info["radius"]
        # Initialize nodes: first MIN_ACTIVE_NODES active, others are idle (available via Wake-on-LAN)
        self.nodes = [Node(f"{self.bs_id}_Node_{i}", active=(i < MIN_ACTIVE_NODES))
                      for i in range(NODES_PER_BS)]
        self.queue = []  # List of tasks waiting for an active node.
        self.wake_threshold = 1.5  # seconds, for waking up an idle node

    def assign_task(self, task, current_time):
        # If queue is too long, reject immediately.
        if len(self.queue) >= MAX_QUEUE_LENGTH:
            task["status"] = "rejected"
            task["waiting_time"] = 0
            return False

        # Find all active nodes with at least one available slot.
        available_nodes = [node for node in self.nodes if node.is_available(current_time)]
        if available_nodes:
            # Choose the node with the fewest active tasks.
            chosen_node = min(available_nodes, key=lambda n: len(n.active_tasks))
            chosen_node.assign_task(current_time, task["processing_time"])
            task["node_assigned"] = chosen_node.node_id
            task["waiting_time"] = 0
            task["status"] = "assigned"
            return True
        else:
            # No node has available capacity; enqueue the task.
            self.queue.append(task)
            task["status"] = "queued"
            return False

    def wake_idle_node(self, current_time):
        # Find an idle node (one that is turned off)
        idle_nodes = [node for node in self.nodes if not node.active]
        if idle_nodes:
            node_to_wake = idle_nodes[0]
            node_to_wake.active = True
            # Simulate wake-on-LAN delay:
            wake_delay = random.uniform(0.5, 1.5)
            node_to_wake.assign_task(current_time, wake_delay)
            print(f"[{self.bs_id}] Woke node {node_to_wake.node_id} at time {current_time} (wake delay {wake_delay:.2f}s)")
            return node_to_wake.node_id
        return None

    def process_queue(self, current_time, max_tasks_per_step=5):
        processed = 0
        new_queue = []
        for task in self.queue:
            if processed >= max_tasks_per_step:
                new_queue.append(task)
                continue

            waiting_time = current_time - task["arrival_time"]
            if waiting_time > task["deadline"]:
                task["status"] = "dropped"
                task["waiting_time"] = waiting_time
            else:
                if self.assign_task(task, current_time):
                    processed += 1
                else:
                    new_queue.append(task)
        self.queue = new_queue

        # Wake an idle node if average waiting time is too high.
        if self.queue:
            avg_wait = sum(current_time - t["arrival_time"] for t in self.queue) / len(self.queue)
            if avg_wait > self.wake_threshold:
                self.wake_idle_node(current_time)

        # Now, add logic to turn off active nodes that have been idle for too long.
        for node in self.nodes:
            node.update_tasks(current_time)
            # If node is active, has no running tasks, and has been idle longer than threshold,
            # and if it is not one of the minimum required active nodes, turn it off.
            if node.active and len(node.active_tasks) == 0 and node.idle_since is not None:
                idle_time = current_time - node.idle_since
                if idle_time > IDLE_THRESHOLD:
                    # Check how many nodes are still active.
                    active_nodes = [n for n in self.nodes if n.active]
                    if len(active_nodes) > MIN_ACTIVE_NODES:
                        node.active = False
                        print(f"[{self.bs_id}] Turning off node {node.node_id} due to idleness (idle {idle_time:.2f}s)")

# Build BaseStation instances dictionary.
BASE_STATION_INSTANCES = {bs["id"]: BaseStation(bs) for bs in BASE_STATIONS}


# -------------------------------
# Main Simulation Function
# -------------------------------
def run_simulation():
    random.seed(42)
    sumoBinary = sumolib.checkBinary('sumo')  # or 'sumo-gui'
    traci.start([sumoBinary, "--ignore-route-errors", "-c", SUMO_CONFIG])
    
    last_bs_assignment = {}
    data_log = []
    simulation_step = 0

    while traci.simulation.getMinExpectedNumber() > 0 and simulation_step < SIMULATION_DURATION:
        traci.simulationStep()
        vehicle_ids = traci.vehicle.getIDList()
        for veh in vehicle_ids:
            pos = traci.vehicle.getPosition(veh)
            speed = traci.vehicle.getSpeed(veh)
            bs_id, distance, in_coverage = get_nearest_bs(pos)
            bs_instance = BASE_STATION_INSTANCES[bs_id]
            
            scenario_id = assign_scenario()
            data_size = SCENARIO_DATA_SIZES.get(scenario_id, 1000)
            result_size = SCENARIO_RESULT_SIZES.get(scenario_id, 200)
            
            send_latency = compute_latency(distance, pos, data_size)
            return_latency = compute_return_latency(distance, pos, result_size)
            total_latency = send_latency + return_latency
            processing_time = total_latency  # Placeholder processing time
            deadline = generate_deadline()
            
            task = {
                "time": simulation_step,
                "vehicle_id": veh,
                "vehicle_x": pos[0],
                "vehicle_y": pos[1],
                "speed": speed,
                "base_station": bs_id,
                "distance_to_bs": distance,
                "in_coverage": in_coverage,
                "send_latency": send_latency,
                "return_latency": return_latency,
                "total_latency": total_latency,
                "energy": compute_energy(total_latency),
                "scenario_id": scenario_id,
                "data_size": data_size,
                "result_size": result_size,
                "deadline": deadline,
                "arrival_time": simulation_step,
                "processing_time": processing_time,
                "waiting_time": 0,
                "node_assigned": None,
                "status": "new"
            }
            
            # Try to assign the task immediately.
            bs_instance.assign_task(task, simulation_step)
            
            # Handover detection.
            prev_bs = last_bs_assignment.get(veh)
            handover = (prev_bs is not None and prev_bs != bs_id)
            last_bs_assignment[veh] = bs_id
            task["handover"] = handover
            
            data_log.append(task)
        
        # Process queues only every QUEUE_PROCESS_INTERVAL steps.
        if simulation_step % QUEUE_PROCESS_INTERVAL == 0:
            for bs_instance in BASE_STATION_INSTANCES.values():
                bs_instance.process_queue(simulation_step)
        
        simulation_step += TIME_STEP

    traci.close()
    df = pd.DataFrame(data_log)
    output_csv = os.path.join(os.getcwd(), "urban_mobility_advanced_dataset_with_nodes.csv")
    df.to_csv(output_csv, index=False)
    print("Simulation finished. Data saved to:", output_csv)

if __name__ == "__main__":
    run_simulation()
