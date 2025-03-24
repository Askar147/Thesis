import numpy as np
import gym
from gym import spaces
import torch
import random
from collections import deque
import traci
import sumolib
import math
import os
from energy_model import EnergyModel
from latency_model import LatencyModel

class VECEnvironment(gym.Env):
    def __init__(self, 
                 sumo_config="astana.sumocfg",
                 simulation_duration=60,
                 time_step=1,
                 queue_process_interval=5,
                 max_queue_length=50,
                 history_length=10,
                 energy_csv_path=None,
                 energy_weight=0.5,
                 latency_model_params=None,
                 seed=42):
        super(VECEnvironment, self).__init__()
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
        
        self.sumo_config = sumo_config
        self.simulation_duration = simulation_duration
        self.time_step = time_step
        self.queue_process_interval = queue_process_interval
        self.max_queue_length = max_queue_length
        self.history_length = history_length
        self.energy_weight = energy_weight
        
        self.energy_model = EnergyModel(energy_csv_path)
        
        if latency_model_params is None:
            latency_model_params = {
                'frequency_band': 2.4,
                'bandwidth': 20,
                'noise_floor': -95
            }
        self.latency_model = LatencyModel(**latency_model_params)
        
        self.base_stations = [
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
        self.num_base_stations = len(self.base_stations)
        
        self.nodes_per_bs = 20
        self.min_active_nodes = 10  
        self.max_concurrent_tasks = 5  
        self.idle_threshold = 10  
        
        self.action_space = spaces.Discrete(self.nodes_per_bs + 1)
        
        self.observation_space = spaces.Dict({
            # Task information
            'task_size': spaces.Box(low=0, high=10000, shape=(1,), dtype=np.float32),
            'required_cpu_cycles': spaces.Box(low=0, high=1500, shape=(1,), dtype=np.float32),
            'task_deadline': spaces.Box(low=0, high=30, shape=(1,), dtype=np.float32),
            'task_scenario_id': spaces.Box(low=1, high=10, shape=(1,), dtype=np.float32),
            
            # Vehicle information
            'vehicle_pos_x': spaces.Box(low=0, high=5000, shape=(1,), dtype=np.float32),
            'vehicle_pos_y': spaces.Box(low=0, high=8000, shape=(1,), dtype=np.float32),
            'vehicle_speed': spaces.Box(low=0, high=30, shape=(1,), dtype=np.float32),
            
            # Base station information
            'distance_to_bs': spaces.Box(low=0, high=2000, shape=(1,), dtype=np.float32),
            'bs_queue_length': spaces.Box(low=0, high=50, shape=(1,), dtype=np.float32),
            
            # Edge node information
            'active_nodes': spaces.Box(low=0, high=20, shape=(1,), dtype=np.float32),
            'node_loads': spaces.Box(low=0, high=1, shape=(self.nodes_per_bs,), dtype=np.float32),
            'node_active_status': spaces.MultiBinary(self.nodes_per_bs),
            
            # Historical information
            'historical_loads': spaces.Box(low=0, high=1, shape=(self.history_length, self.nodes_per_bs), dtype=np.float32),
            
            # New information from enhanced models
            'link_quality': spaces.Box(low=0, high=5, shape=(1,), dtype=np.float32),
            'energy_efficiency': spaces.Box(low=0, high=10, shape=(1,), dtype=np.float32),
        })
        
        self.base_station_instances = {}
        for bs in self.base_stations:
            self.base_station_instances[bs["id"]] = BaseStation(bs, self.nodes_per_bs, 
                                                               self.min_active_nodes, 
                                                               self.max_concurrent_tasks, 
                                                               self.idle_threshold)
        
        self.sumo_initialized = False
        self.simulation_step = 0
        self.last_bs_assignment = {}
        
        self.task_completion_history = deque(maxlen=100)
        self.task_rejection_history = deque(maxlen=100)
        self.task_drop_history = deque(maxlen=100)
        self.latency_history = deque(maxlen=100)
        self.energy_consumption_history = deque(maxlen=100)
        
        self.current_task = None
        self.current_bs = None
        self.current_vehicle = None
        
        self.load_history = {bs_id: deque(maxlen=self.history_length) for bs_id in [bs["id"] for bs in self.base_stations]}
        for bs_id in self.load_history:
            for _ in range(self.history_length):
                self.load_history[bs_id].append(np.zeros(self.nodes_per_bs))
    
    def initialize_sumo(self):
        if not self.sumo_initialized:
            sumoBinary = sumolib.checkBinary('sumo')  # or 'sumo-gui'
            # Add the --no-warnings flag and other options to reduce console output
            traci.start([sumoBinary, 
                        "--ignore-route-errors", 
                        "--no-step-log",
                        "-c", self.sumo_config])
            self.sumo_initialized = True

    def reset(self):
        if self.sumo_initialized:
            traci.close()
            self.sumo_initialized = False
        
        self.initialize_sumo()
        
        self.simulation_step = 0
        self.last_bs_assignment = {}
        
        for bs_id, bs_instance in self.base_station_instances.items():
            bs_instance.reset()
        
        self.task_completion_history.clear()
        self.task_rejection_history.clear()
        self.task_drop_history.clear()
        self.latency_history.clear()
        self.energy_consumption_history.clear()
        
        for bs_id in self.load_history:
            for i in range(self.history_length):
                self.load_history[bs_id][i] = np.zeros(self.nodes_per_bs)
        
        if hasattr(self, 'latency_model'):
            self.latency_model.reset_history()
        
        obs = self._get_observation()
        
        return obs
    
    def _get_observation(self):
        if self.current_task is None or self.current_bs is None or self.current_vehicle is None:
            return {
                'task_size': np.array([0], dtype=np.float32),
                'required_cpu_cycles': np.array([0], dtype=np.float32),
                'task_deadline': np.array([0], dtype=np.float32),
                'task_scenario_id': np.array([1], dtype=np.float32),
                'vehicle_pos_x': np.array([0], dtype=np.float32),
                'vehicle_pos_y': np.array([0], dtype=np.float32),
                'vehicle_speed': np.array([0], dtype=np.float32),
                'distance_to_bs': np.array([0], dtype=np.float32),
                'bs_queue_length': np.array([0], dtype=np.float32),
                'active_nodes': np.array([self.min_active_nodes], dtype=np.float32),
                'node_loads': np.zeros(self.nodes_per_bs, dtype=np.float32),
                'node_active_status': np.array([1] * self.min_active_nodes + [0] * (self.nodes_per_bs - self.min_active_nodes), dtype=np.int8),
                'historical_loads': np.array(list(self.load_history[self.base_stations[0]["id"]])) if self.base_stations else np.zeros((self.history_length, self.nodes_per_bs), dtype=np.float32),
                'link_quality': np.array([0], dtype=np.float32),
                'energy_efficiency': np.array([0], dtype=np.float32),
            }
        
        bs_instance = self.base_station_instances[self.current_bs.bs_id]
        
        node_loads = np.array([len(node.active_tasks) / self.max_concurrent_tasks for node in bs_instance.nodes], dtype=np.float32)
        
        node_active_status = np.array([1 if node.active else 0 for node in bs_instance.nodes], dtype=np.int8)
        
        historical_loads = np.array(list(self.load_history[self.current_bs.bs_id]))
        
        distance = self.current_task["distance_to_bs"]
        interfering_distances = []
        for bs in self.base_stations:
            if bs["id"] != self.current_bs.bs_id:
                bs_pos = bs["pos"]
                vehicle_pos = (self.current_task["vehicle_x"], self.current_task["vehicle_y"])
                interfering_dist = self._euclidean_distance(vehicle_pos, bs_pos)
                interfering_distances.append(interfering_dist)
        
        _, sinr = self.latency_model.get_link_quality(distance, interfering_distances)
        link_quality = np.array([min(5.0, max(0.0, sinr / 5.0))], dtype=np.float32)  # Normalize to 0-5 range
        
        scenario_id = int(self.current_task["scenario_id"])
        energy_efficiency = np.array([self.energy_model.get_energy_efficiency_ratio(scenario_id)], dtype=np.float32)
        
        return {
            'task_size': np.array([self.current_task["data_size"]], dtype=np.float32),
            'required_cpu_cycles': np.array([self.current_task["required_cpu_cycles"]], dtype=np.float32),
            'task_deadline': np.array([self.current_task["deadline"]], dtype=np.float32),
            'task_scenario_id': np.array([self.current_task["scenario_id"]], dtype=np.float32),
            'vehicle_pos_x': np.array([self.current_task["vehicle_x"]], dtype=np.float32),
            'vehicle_pos_y': np.array([self.current_task["vehicle_y"]], dtype=np.float32),
            'vehicle_speed': np.array([self.current_task["speed"]], dtype=np.float32),
            'distance_to_bs': np.array([self.current_task["distance_to_bs"]], dtype=np.float32),
            'bs_queue_length': np.array([len(bs_instance.queue)], dtype=np.float32),
            'active_nodes': np.array([sum(node_active_status)], dtype=np.float32),
            'node_loads': node_loads,
            'node_active_status': node_active_status,
            'historical_loads': historical_loads,
            'link_quality': link_quality,
            'energy_efficiency': energy_efficiency,
        }
    
    def step(self, action):
        if not self.sumo_initialized:
            self.initialize_sumo()
        
        if self.current_task is None or self.current_bs is None:
            return self._simulate_and_get_new_task()
        
        bs_instance = self.base_station_instances[self.current_bs.bs_id]
        
        reward = 0
        info = {}
        energy_consumption = 0
        
        if action == self.nodes_per_bs:
            woken_node = bs_instance.wake_idle_node(self.simulation_step)
            if woken_node:
                info['woken_node'] = woken_node
                # Energy penalty for waking up a node - use enhanced model
                energy_penalty = self.energy_model.get_wake_up_energy()
                energy_consumption = energy_penalty
                # Small negative reward for waking up a node (energy penalty)
                wake_reward = -self.energy_weight * energy_penalty / 100.0  # Normalize
                reward += wake_reward
            else:
                reward -= 0.5
                
            # Try to process the current task again with the next step
            return self._simulate_and_get_new_task(reward_offset=reward, energy_offset=energy_consumption)
        else:
            # Offload to a specific node
            if action < 0 or action >= self.nodes_per_bs:
                # Invalid action, default to node 0
                action = 0
            
            task = self.current_task.copy()
            task["node_assigned"] = f"{self.current_bs.bs_id}_Node_{action}"
            
            # Check if the selected node is active
            if not bs_instance.nodes[action].active:
                # Trying to offload to an inactive node
                reward -= 1.0  # Penalty
                task["status"] = "rejected"
                task["waiting_time"] = 0
                self.task_rejection_history.append(1)
            else:
                # Calculate the more accurate processing time based on node load
                node = bs_instance.nodes[action]
                node_capacity = 500  # Megacycles per second (adjust based on your node specs)
                node_load = len(node.active_tasks) / node.max_concurrent_tasks
                
                # Update processing time using latency model
                processing_time = self.latency_model.get_processing_latency(
                    task["required_cpu_cycles"], node_capacity, node_load
                )
                task["processing_time"] = processing_time
                
                # Recalculate total latency
                total_latency = task["send_latency"] + processing_time + task["return_latency"]
                task["total_latency"] = total_latency
                
                # Try to assign the task
                success = bs_instance.assign_task(task, self.simulation_step)
                if success:
                    task["status"] = "assigned"
                    
                    # Get processing time and energy consumption for the task
                    scenario_id = int(task["scenario_id"])
                    
                    # Use enhanced energy model to get energy consumption
                    energy_consumption = self.energy_model.get_energy_consumption(
                        scenario_id, 
                        duration=processing_time,
                        node_load=node_load
                    )
                    
                    # Add to history
                    self.energy_consumption_history.append(energy_consumption)
                    self.latency_history.append(total_latency)
                    
                    # Calculate reward components
                    latency_weight = 1.0 - self.energy_weight
                    
                    # Normalize latency between 0 and 1 using deadline
                    normalized_latency = min(1.0, total_latency / task["deadline"])
                    
                    # Latency component: higher reward for lower latency
                    latency_reward = latency_weight * (1.0 - normalized_latency)
                    
                    max_energy = 100.0  # Assuming maximum energy consumption is 100 Joules
                    normalized_energy = min(1.0, energy_consumption / max_energy)
                    energy_reward = self.energy_weight * (1.0 - normalized_energy)
                    
                    # REWARD FUNCTION
                    reward = latency_reward + energy_reward
                    
                    self.task_completion_history.append(1)
                    self.task_rejection_history.append(0)
                else:
                    task["status"] = "queued"
                    # Small negative reward for queueing
                    reward -= 0.2
        
        node_loads = np.array([len(node.active_tasks) / self.max_concurrent_tasks for node in bs_instance.nodes], dtype=np.float32)
        self.load_history[self.current_bs.bs_id].append(node_loads)
        
        # Process the next task
        return self._simulate_and_get_new_task(reward_offset=reward, energy_offset=energy_consumption)
    
    def _simulate_and_get_new_task(self, reward_offset=0.0, energy_offset=0.0):
        """Simulate a step and get a new task for decision making"""
        # Simulate a step in SUMO
        traci.simulationStep()
        self.simulation_step += self.time_step
        
        if self.simulation_step % self.queue_process_interval == 0:
            for bs_instance in self.base_station_instances.values():
                bs_instance.process_queue(self.simulation_step)
        
        # Check if simulation is done
        done = self.simulation_step >= self.simulation_duration or traci.simulation.getMinExpectedNumber() <= 0
        
        # Reset current task
        self.current_task = None
        self.current_bs = None
        self.current_vehicle = None
        
        reward = reward_offset
        
        if not done:
            vehicle_ids = traci.vehicle.getIDList()
            
            if vehicle_ids:
                # Pick a random vehicle for task generation
                vehicle_id = random.choice(vehicle_ids)
                pos = traci.vehicle.getPosition(vehicle_id)
                speed = traci.vehicle.getSpeed(vehicle_id)
                
                # Find nearest base station
                nearest_bs, distance, in_coverage = self._get_nearest_bs(pos)
                
                if in_coverage:
                    scenario_id = random.randint(1, 10) 
                    
                    data_size = self.latency_model.get_task_data_size(scenario_id, is_upstream=True)
                    
                    required_cpu_cycles = self.latency_model.get_required_cpu_cycles(scenario_id)
                    deadline = random.uniform(6, 20)  # seconds
                    
                    # Get interfering distances for SINR calculation
                    interfering_distances = []
                    for bs in self.base_stations:
                        if bs["id"] != nearest_bs:
                            interfering_dist = self._euclidean_distance(pos, bs["pos"])
                            interfering_distances.append(interfering_dist)
                    
                    # Use latency model to calculate transmission latencies
                    uplink_latency = self.latency_model.get_transmission_latency(
                        scenario_id, distance, is_upstream=True, interfering_distances=interfering_distances
                    )
                    downlink_latency = self.latency_model.get_transmission_latency(
                        scenario_id, distance, is_upstream=False, interfering_distances=interfering_distances
                    )
                    
                    base_processing_time = required_cpu_cycles / 500  # Will be adjusted based on actual node
                    
                    total_latency = uplink_latency + downlink_latency + base_processing_time
                    
                    task = {
                        "time": self.simulation_step,
                        "vehicle_id": vehicle_id,
                        "vehicle_x": pos[0],
                        "vehicle_y": pos[1],
                        "speed": speed,
                        "base_station": nearest_bs,
                        "distance_to_bs": distance,
                        "in_coverage": in_coverage,
                        "send_latency": uplink_latency,
                        "return_latency": downlink_latency,
                        "total_latency": total_latency,
                        "scenario_id": scenario_id,
                        "data_size": data_size,
                        "required_cpu_cycles": required_cpu_cycles,
                        "deadline": deadline,
                        "arrival_time": self.simulation_step,
                        "processing_time": base_processing_time,  # Initial estimate, will be refined
                        "waiting_time": 0,
                        "node_assigned": None,
                        "status": "new"
                    }
                    
                    # Set current task and base station
                    self.current_task = task
                    self.current_bs = self.base_station_instances[nearest_bs]
                    self.current_vehicle = vehicle_id
                    
                    # Handle handover detection
                    prev_bs = self.last_bs_assignment.get(vehicle_id)
                    handover = (prev_bs is not None and prev_bs != nearest_bs)
                    self.last_bs_assignment[vehicle_id] = nearest_bs
                    task["handover"] = handover
        
        # Get observation
        obs = self._get_observation()
        
        # Calculate idle energy for all nodes
        idle_energy = 0
        for bs_id, bs_instance in self.base_station_instances.items():
            idle_energy += bs_instance.calculate_idle_energy(self.energy_model, self.time_step)
        
        # Return info
        info = {
            'simulation_step': self.simulation_step,
            'task_completion_rate': np.mean(list(self.task_completion_history)) if self.task_completion_history else 0,
            'task_rejection_rate': np.mean(list(self.task_rejection_history)) if self.task_rejection_history else 0,
            'task_drop_rate': np.mean(list(self.task_drop_history)) if self.task_drop_history else 0,
            'avg_latency': np.mean(list(self.latency_history)) if self.latency_history else 0,
            'avg_energy_consumption': np.mean(list(self.energy_consumption_history)) if self.energy_consumption_history else 0,
            'energy_consumption': energy_offset,
            'idle_energy': idle_energy,
            
            # New metrics
            'avg_uplink_latency': self.latency_model.get_average_latency(window=10) if hasattr(self, 'latency_model') and self.latency_model.latency_history else 0,
            'avg_data_rate': np.mean(list(self.latency_model.datarate_history))/(1e6) if hasattr(self, 'latency_model') and self.latency_model.datarate_history else 0,  # In Mbps
            'avg_idle_power': self.energy_model.get_idle_power() if hasattr(self, 'energy_model') else 0,
        }
        
        return obs, reward, done, info
    
    def _get_nearest_bs(self, vehicle_pos):
        nearest_bs, min_dist, in_cov = None, float("inf"), False
        for bs in self.base_stations:
            dist = self._euclidean_distance(vehicle_pos, bs["pos"])
            if dist < min_dist:
                min_dist, nearest_bs = dist, bs["id"]
                in_cov = (dist <= bs["radius"])
        return nearest_bs, min_dist, in_cov
    
    def _euclidean_distance(self, pos1, pos2):
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    
    def close(self):
        if self.sumo_initialized:
            traci.close()
            self.sumo_initialized = False


class Node:
    """Class representing an edge computing node in a base station"""
    def __init__(self, node_id, active=True, max_concurrent_tasks=4):
        self.node_id = node_id
        self.active_tasks = []  # List of task finish times (in simulation time)
        self.active = active    # True if node is active (online); False if idle/offline
        self.idle_since = None  # Timestamp when node became idle (if no active tasks)
        self.max_concurrent_tasks = max_concurrent_tasks

    def update_tasks(self, current_time):
        """Update tasks based on current time"""
        # Remove finished tasks
        self.active_tasks = [end_time for end_time in self.active_tasks if current_time < end_time]
        # Update idle_since: if no tasks are running and not already marked
        if len(self.active_tasks) == 0 and self.idle_since is None:
            self.idle_since = current_time
        elif len(self.active_tasks) > 0:
            self.idle_since = None

    def available_slots(self, current_time):
        """Get number of available slots for tasks"""
        self.update_tasks(current_time)
        return self.max_concurrent_tasks - len(self.active_tasks)

    def is_available(self, current_time):
        """Check if node is available for new tasks"""
        self.update_tasks(current_time)
        return self.active and (len(self.active_tasks) < self.max_concurrent_tasks)

    def assign_task(self, current_time, processing_time):
        """Assign a task to this node"""
        finish_time = current_time + processing_time
        self.active_tasks.append(finish_time)
        # When a task is assigned, the node is no longer idle
        self.idle_since = None
        return True

    def get_load(self):
        """Get the current load of the node (ratio of used slots to total slots)"""
        return len(self.active_tasks) / self.max_concurrent_tasks


class BaseStation:
    """Class representing a base station with multiple edge computing nodes"""
    def __init__(self, bs_info, nodes_per_bs, min_active_nodes, max_concurrent_tasks, idle_threshold):
        self.bs_id = bs_info["id"]
        self.pos = bs_info["pos"]
        self.radius = bs_info["radius"]
        self.nodes_per_bs = nodes_per_bs
        self.min_active_nodes = min_active_nodes
        self.max_concurrent_tasks = max_concurrent_tasks
        self.idle_threshold = idle_threshold
        
        # Initialize nodes: first min_active_nodes active, others are idle (available via Wake-on-LAN)
        self.nodes = [Node(f"{self.bs_id}_Node_{i}", active=(i < min_active_nodes), max_concurrent_tasks=max_concurrent_tasks)
                      for i in range(nodes_per_bs)]
        self.queue = []  # List of tasks waiting for an active node
        self.wake_threshold = 1.5  # seconds, for waking up an idle node
    
    def reset(self):
        """Reset the base station"""
        for i, node in enumerate(self.nodes):
            node.active = (i < self.min_active_nodes)
            node.active_tasks = []
            node.idle_since = None
        self.queue = []
    
    def assign_task(self, task, current_time):
        """Try to assign a task to an available node"""
        # If queue is too long, reject immediately
        if len(self.queue) >= 50:  # max_queue_length
            task["status"] = "rejected"
            task["waiting_time"] = 0
            return False

        # Find an active node with available slots
        available_nodes = []
        for node in self.nodes:
            if node.active and node.is_available(current_time):
                available_nodes.append(node)
                
        if available_nodes:
            # Choose the node with the fewest active tasks
            chosen_node = min(available_nodes, key=lambda n: len(n.active_tasks))
            chosen_node.assign_task(current_time, task["processing_time"])
            task["node_assigned"] = chosen_node.node_id
            task["waiting_time"] = 0
            task["status"] = "assigned"
            return True
        else:
            # No node has available capacity; enqueue the task
            self.queue.append(task)
            task["status"] = "queued"
            return False
    
    def wake_idle_node(self, current_time):
        """Wake up an idle node if available"""
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
        """Process tasks in the queue"""
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

        # Wake an idle node if average waiting time is too high
        if self.queue:
            avg_wait = sum(current_time - t["arrival_time"] for t in self.queue) / len(self.queue)
            if avg_wait > self.wake_threshold:
                self.wake_idle_node(current_time)

        # Turn off active nodes that have been idle for too long
        for node in self.nodes:
            node.update_tasks(current_time)
            # If node is active, has no running tasks, and has been idle longer than threshold,
            # and if it is not one of the minimum required active nodes, turn it off
            if node.active and len(node.active_tasks) == 0 and node.idle_since is not None:
                idle_time = current_time - node.idle_since
                if idle_time > self.idle_threshold:
                    # Check how many nodes are still active
                    active_nodes = [n for n in self.nodes if n.active]
                    if len(active_nodes) > self.min_active_nodes:
                        node.active = False
                        print(f"[{self.bs_id}] Turning off node {node.node_id} due to idleness (idle {idle_time:.2f}s)")
    
    def get_node_loads(self):
        """Get the current load of all nodes in the base station"""
        return [node.get_load() for node in self.nodes]
    
    def get_active_nodes_count(self):
        """Get the number of active nodes in the base station"""
        return sum(1 for node in self.nodes if node.active)
    
    def calculate_idle_energy(self, energy_model, time_interval):
        """Calculate the idle energy consumption for this base station's nodes"""
        idle_energy = 0
        for node in self.nodes:
            if node.active and len(node.active_tasks) == 0:
                # Node is active but idle - calculate idle energy
                idle_energy += energy_model.get_idle_power() * time_interval
        return idle_energy