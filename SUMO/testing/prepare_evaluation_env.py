#!/usr/bin/env python3
"""
Prepare environment for VEC model evaluation by generating different
traffic patterns and SUMO configurations
"""

import os
import subprocess
import argparse
import shutil
import random
from datetime import datetime

def run_command(command, verbose=True):
    """Run a shell command and optionally print output"""
    if verbose:
        print(f"Running: {command}")
    
    # Use list format for command to properly handle paths with spaces
    if isinstance(command, str):
        import shlex
        command = shlex.split(command)
    
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    if verbose:
        if stdout:
            print(stdout.decode('utf-8'))
        if stderr:
            print(stderr.decode('utf-8'))
    
    return process.returncode

def generate_sumo_config(net_file, route_file, output_dir, name, duration):
    """Generate a SUMO configuration file"""
    config_path = os.path.join(output_dir, f"{name}.sumocfg")
    
    config_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="{net_file}"/>
        <route-files value="{route_file}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{duration}"/>
    </time>
    <processing>
        <ignore-route-errors value="true"/>
        <time-to-teleport value="-1"/>
    </processing>
    <report>
        <verbose value="false"/>
        <duration-log.disable value="true"/>
        <no-step-log value="true"/>
    </report>
</configuration>
"""
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    return config_path

def main():
    parser = argparse.ArgumentParser(description='Prepare environment for VEC model evaluation')
    parser.add_argument('--sumo_home', type=str, default=None, 
                       help='Path to SUMO home directory (default: use SUMO_HOME environment variable)')
    parser.add_argument('--net_file', type=str, default='astana.net.xml',
                       help='Path to network file (default: astana.net.xml)')
    parser.add_argument('--output_dir', type=str, default='eval_scenarios',
                       help='Directory to save generated files')
    parser.add_argument('--duration', type=int, default=600,
                       help='Simulation duration in seconds (default: 600)')
    
    args = parser.parse_args()
    
    # Check SUMO_HOME
    sumo_home = args.sumo_home or os.environ.get('SUMO_HOME')
    if not sumo_home:
        print("Error: SUMO_HOME not set. Please set the SUMO_HOME environment variable or provide --sumo_home")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if net file exists
    if not os.path.exists(args.net_file):
        print(f"Error: Network file {args.net_file} not found")
        return 1
    
    # Copy net file to output directory
    net_file_name = os.path.basename(args.net_file)
    shutil.copy(args.net_file, os.path.join(args.output_dir, net_file_name))
    
    # Define different traffic scenarios
    scenarios = {
        "baseline": {
            "period": 1.5,  # Vehicle insertion period (lower = more vehicles)
            "seed": 42,
            "description": "Normal traffic conditions"
        },
        "high_traffic": {
            "period": 0.5,  # More frequent vehicle insertions
            "seed": 43,
            "description": "Rush hour traffic"
        },
        "low_traffic": {
            "period": 3.0,  # Less frequent vehicle insertions
            "seed": 44,
            "description": "Light traffic conditions"
        },
        "random_traffic": {
            "period": 1.0,
            "seed": random.randint(100, 999),
            "description": "Random traffic pattern"
        },
        "evaluation": {
            "period": 1.0,
            "seed": 100,  # Different from training
            "description": "Evaluation traffic pattern"
        }
    }
    
    # Generate route files and SUMO configs for each scenario
    print("Generating scenarios...")
    
    for scenario_name, params in scenarios.items():
        print(f"\nGenerating {scenario_name} scenario:")
        print(f"  Description: {params['description']}")
        print(f"  Period: {params['period']} (lower = more vehicles)")
        print(f"  Seed: {params['seed']}")
        
        route_file = f"{scenario_name}.rou.xml"
        route_file_path = os.path.join(args.output_dir, route_file)
        
        # Generate route file using randomTrips.py
        random_trips_path = os.path.join(sumo_home, "tools", "randomTrips.py")
        
        # Create command as a list to handle paths with spaces
        command = [
            "python",
            random_trips_path,
            "-n", os.path.join(args.output_dir, net_file_name),
            "-o", route_file_path,
            "-e", str(args.duration),
            "-p", str(params['period']),
            "--seed", str(params['seed'])
        ]
        
        result = run_command(command)
        if result != 0:
            print(f"Error generating routes for {scenario_name}")
            continue
        
        # Generate SUMO config
        config_path = generate_sumo_config(
            net_file_name,
            route_file,
            args.output_dir,
            scenario_name,
            args.duration
        )
        
        print(f"  Created SUMO config: {config_path}")
    
    print("\nAll scenarios generated successfully!")
    print(f"Files saved to: {os.path.abspath(args.output_dir)}")
    print("\nTo use these scenarios in evaluation, point to the specific .sumocfg file when running the evaluation script.")

if __name__ == "__main__":
    main()