import os
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ---------------------------
# Parsing Functions (Keep existing implementations)
# ---------------------------

def parse_docker_stats(file_path):
    """
    Parse a Docker stats log file.
    """
    entries = []
    timestamp_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'
    
    with open(file_path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Check for header line with timestamp and "CONTAINER"
        if re.match(timestamp_pattern, line) and "CONTAINER" in line:
            if i + 1 < len(lines):
                timestamp_match = re.match(timestamp_pattern, line)
                timestamp = datetime.strptime(timestamp_match.group(1), "%Y-%m-%d %H:%M:%S")
                
                data_line = lines[i+1].strip()
                
                # Extract container ID (first word)
                parts = data_line.split(None, 1)  # Split on first whitespace
                if len(parts) < 2:
                    print(f"Warning: Missing container ID in file {file_path} line {i+2}. Skipping entry.")
                    i += 2
                    continue
                    
                container_id = parts[0]
                remaining = parts[1]
                
                # Extract CPU percentage
                cpu_match = re.search(r'(\d+\.\d+|\d+)%', remaining)
                if not cpu_match:
                    print(f"Warning: Could not find CPU percentage in file {file_path} line {i+2}. Skipping entry.")
                    i += 2
                    continue
                
                cpu_percent = float(cpu_match.group(1))
                
                # Extract memory usage
                mem_match = re.search(r'([\d\.]+[MkGiB]+)\s*/\s*([\d\.]+[MkGiB]+)', remaining)
                if not mem_match:
                    print(f"Warning: Memory usage format unexpected in file {file_path} line {i+2}. Trying alternative format.")
                    # Try alternative format without the slash
                    mem_match = re.search(r'([\d\.]+[MkGiB]+)', remaining)
                    if mem_match:
                        mem_usage = mem_match.group(1)
                        mem_limit = "N/A"  # Not available in this format
                    else:
                        print(f"Warning: Could not parse memory usage in file {file_path} line {i+2}. Skipping entry.")
                        i += 2
                        continue
                else:
                    mem_usage = mem_match.group(1)
                    mem_limit = mem_match.group(2)
                
                # Extract network I/O
                net_match = re.search(r'([\d\.]+[kMGiB]*)\s*/\s*([\d\.]+[kMGiB]*)', remaining[remaining.find(mem_match.group(0)) + len(mem_match.group(0)):])
                if net_match:
                    net_in = net_match.group(1)
                    net_out = net_match.group(2)
                else:
                    net_in = "0B"
                    net_out = "0B"
                
                # Extract block I/O
                block_match = re.search(r'([\d\.]+[kMGiB]*)\s*/\s*([\d\.]+[kMGiB]*)', remaining[remaining.find(net_match.group(0) if net_match else mem_match.group(0)) + len(net_match.group(0) if net_match else mem_match.group(0)):])
                if block_match:
                    block_in = block_match.group(1)
                    block_out = block_match.group(2)
                else:
                    block_in = "0B"
                    block_out = "0B"
                
                # Extract PIDs
                pids_match = re.search(r'(\d+)$', data_line)
                if pids_match:
                    pids = int(pids_match.group(1))
                else:
                    pids = 0
                
                entries.append({
                    "timestamp": timestamp,
                    "container_id": container_id,
                    "cpu_percent": cpu_percent,
                    "mem_usage": mem_usage,
                    "mem_limit": mem_limit,
                    "net_in": net_in,
                    "net_out": net_out,
                    "block_in": block_in,
                    "block_out": block_out,
                    "pids": pids
                })
            i += 2
        else:
            i += 1

    df = pd.DataFrame(entries)
    return df

def parse_perf_stat(file_path):
    """
    Parse a perf_stat log file.
    """
    entries = []
    pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+[\d\.]+\s+([\d\s,]+)\s+(\w+)'
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            m = re.match(pattern, line)
            if m:
                ts_str, value_str, event = m.groups()
                value_clean = int(re.sub(r'[,\s]', '', value_str))
                timestamp = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                entries.append({
                    "timestamp": timestamp,
                    "event": event,
                    "value": value_clean
                })
    df = pd.DataFrame(entries)
    if not df.empty:
        # Only keep cycles, instructions, cache-references, and cache-misses
        df = df[df['event'].isin(['cycles', 'instructions', 'cache-references', 'cache-misses'])]
        df['event'] = df['event'].str.replace('-', '_')  # Replace hyphens with underscores for column names
        df = df.pivot_table(index="timestamp", columns="event", values="value", aggfunc="first").reset_index()
        # Rename columns for better readability
        df.rename(columns={
            'cache_references': 'cache',
            'cache_misses': 'cache_miss'
        }, inplace=True)
    return df

def parse_tegrastats(file_path):
    """
    Parse a tegrastats log file.
    """
    entries = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            ts_str = line[:19]  # Extract timestamp part
            try:
                timestamp = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
                
            ram_match = re.search(r'RAM\s+(\d+)/(\d+)MB', line)
            swap_match = re.search(r'SWAP\s+(\d+)/(\d+)MB', line)
            cpu_match = re.search(r'CPU\s+\[([^\]]+)\]', line)
            temp_match = re.search(r'CPU@([\d\.]+)C', line)
            
            if ram_match and swap_match:  # Minimal requirement: RAM and SWAP info
                ram_used = int(ram_match.group(1))
                ram_total = int(ram_match.group(2))
                swap_used = int(swap_match.group(1))
                swap_total = int(swap_match.group(2))
                
                # CPU usage might not be available in all log formats
                avg_cpu = None
                if cpu_match:
                    cpu_str = cpu_match.group(1)
                    cpu_values = []
                    for part in cpu_str.split(','):
                        try:
                            cpu_val = float(part.split('%')[0])
                            cpu_values.append(cpu_val)
                        except Exception:
                            continue
                    avg_cpu = np.mean(cpu_values) if cpu_values else None
                
                # CPU temperature might not be available in all log formats
                cpu_temp = None
                if temp_match:
                    cpu_temp = float(temp_match.group(1))
                
                entries.append({
                    "timestamp": timestamp,
                    "ram_used": ram_used,
                    "ram_total": ram_total,
                    "swap_used": swap_used,
                    "swap_total": swap_total,
                    "avg_cpu": avg_cpu,
                    "cpu_temp": cpu_temp
                })
    
    df = pd.DataFrame(entries)
    return df

# ---------------------------
# New Improved File Mapping Functions
# ---------------------------

def extract_timestamp_from_filename(filename):
    """
    Extract the timestamp (last 6 characters) from a filename.
    Expected format: some_prefix_HHMMSS.log or some_prefix_20250310_HHMMSS.log
    Returns the timestamp as a string in format HHMMSS
    """
    # Try to extract the last 6 digits from the filename
    match = re.search(r'_(\d{6})(?:\.log)?$', filename)
    if match:
        return match.group(1)
    
    # If not found, try finding a date pattern with time
    match = re.search(r'_\d{8}_(\d{6})(?:\.log)?$', filename)
    if match:
        return match.group(1)
    
    # If still not found, return None
    return None

def find_matching_files(file_timestamp, all_files, max_time_diff=1):
    """
    Find files that have timestamps within max_time_diff seconds of the given file_timestamp.
    
    Args:
        file_timestamp: String timestamp from filename (HHMMSS)
        all_files: List of all filenames in the directory
        max_time_diff: Maximum time difference in seconds
    
    Returns:
        List of matching filenames
    """
    if not file_timestamp:
        return []
    
    # Convert timestamp string to datetime object
    try:
        file_time = datetime.strptime(file_timestamp, "%H%M%S").time()
    except ValueError:
        return []
    
    matching_files = []
    
    for other_file in all_files:
        other_timestamp = extract_timestamp_from_filename(other_file)
        if not other_timestamp:
            continue
        
        try:
            other_time = datetime.strptime(other_timestamp, "%H%M%S").time()
            
            # Calculate the time difference in seconds
            # Convert times to seconds
            file_seconds = file_time.hour * 3600 + file_time.minute * 60 + file_time.second
            other_seconds = other_time.hour * 3600 + other_time.minute * 60 + other_time.second
            diff_seconds = abs(file_seconds - other_seconds)
            
            # Handle midnight crossing (if needed)
            if diff_seconds > 12 * 3600:  # More than 12 hours difference
                diff_seconds = 24 * 3600 - diff_seconds
            
            if diff_seconds <= max_time_diff:
                matching_files.append(other_file)
        except ValueError:
            continue
    
    return matching_files

def process_logs_with_filename_matching(docker_log_dir, perf_log_dir, tegrastats_log_dir, max_time_diff=1):
    """
    Process logs by matching files based on the timestamps in their filenames.
    
    Args:
        docker_log_dir: Directory with Docker stats logs
        perf_log_dir: Directory with perf_stat logs
        tegrastats_log_dir: Directory with tegrastats logs
        max_time_diff: Maximum time difference in seconds for matching files
    
    Returns:
        Merged DataFrame with data from all matched files
    """
    # Get lists of files in each directory
    docker_files = os.listdir(docker_log_dir) if os.path.exists(docker_log_dir) else []
    perf_files = os.listdir(perf_log_dir) if os.path.exists(perf_log_dir) else []
    tegrastats_files = os.listdir(tegrastats_log_dir) if os.path.exists(tegrastats_log_dir) else []
    
    # Dictionary to hold all DataFrames
    all_dfs = []
    
    # Process each Docker file as the base
    for docker_file in docker_files:
        docker_timestamp = extract_timestamp_from_filename(docker_file)
        if not docker_timestamp:
            continue
        
        print(f"Processing Docker file: {docker_file} with timestamp {docker_timestamp}")
        
        # Find matching perf files
        matching_perf_files = find_matching_files(docker_timestamp, perf_files, max_time_diff)
        if matching_perf_files:
            print(f"  Found matching perf files: {matching_perf_files}")
        else:
            print(f"  No matching perf files found")
        
        # Find matching tegrastats files
        matching_tegra_files = find_matching_files(docker_timestamp, tegrastats_files, max_time_diff)
        if matching_tegra_files:
            print(f"  Found matching tegrastats files: {matching_tegra_files}")
        else:
            print(f"  No matching tegrastats files found")
        
        # Parse the Docker stats file
        docker_file_path = os.path.join(docker_log_dir, docker_file)
        docker_df = parse_docker_stats(docker_file_path)
        
        if docker_df.empty:
            print(f"  No data found in Docker file. Skipping.")
            continue
        
        docker_df["docker_file"] = docker_file
        
        # Parse matching perf files
        perf_dfs = []
        for perf_file in matching_perf_files:
            perf_file_path = os.path.join(perf_log_dir, perf_file)
            perf_df = parse_perf_stat(perf_file_path)
            if not perf_df.empty:
                perf_df["perf_file"] = perf_file
                perf_dfs.append(perf_df)
        
        combined_perf_df = pd.concat(perf_dfs) if perf_dfs else pd.DataFrame()
        
        # Parse matching tegrastats files
        tegra_dfs = []
        for tegra_file in matching_tegra_files:
            tegra_file_path = os.path.join(tegrastats_log_dir, tegra_file)
            tegra_df = parse_tegrastats(tegra_file_path)
            if not tegra_df.empty:
                tegra_df["tegra_file"] = tegra_file
                tegra_dfs.append(tegra_df)
        
        combined_tegra_df = pd.concat(tegra_dfs) if tegra_dfs else pd.DataFrame()
        
        # Now merge the data with a 2-second window
        merged_df = docker_df.copy()
        
        # If we have perf data, merge it
        if not combined_perf_df.empty:
            # Get matches within 2 seconds
            merged_df = align_timestamps(merged_df, combined_perf_df, tolerance=pd.Timedelta("2s"))
        
        # If we have tegrastats data, merge it
        if not combined_tegra_df.empty:
            # Get matches within 2 seconds
            merged_df = align_timestamps(merged_df, combined_tegra_df, tolerance=pd.Timedelta("2s"))
        
        # Add scenario info
        merged_df["scenario"] = docker_file
        
        # Add to all DataFrames
        all_dfs.append(merged_df)
    
    # Combine all matched data
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        return final_df
    else:
        return pd.DataFrame()

def align_timestamps(left_df, right_df, tolerance=pd.Timedelta("2s")):
    """
    Aligns two DataFrames based on timestamp with the given tolerance.
    
    Args:
        left_df: Left DataFrame with timestamp column
        right_df: Right DataFrame with timestamp column
        tolerance: Time tolerance for merging
    
    Returns:
        Merged DataFrame
    """
    left_sorted = left_df.sort_values("timestamp")
    right_sorted = right_df.sort_values("timestamp")
    
    # Use merge_asof to find the closest timestamp match within tolerance
    merged = pd.merge_asof(
        left_sorted,
        right_sorted,
        on="timestamp",
        direction="nearest",
        tolerance=tolerance
    )
    
    return merged

# ---------------------------
# Main Function
# ---------------------------
def main():
    # Directories for logs (adjust these paths as needed)
    docker_log_dir = "docker_logs"
    perf_log_dir = "perf_logs"
    tegrastats_log_dir = "tegrastats_logs"
    
    # Process logs with filename timestamp matching
    print("Processing logs with timestamp matching...")
    merged_data = process_logs_with_filename_matching(
        docker_log_dir, 
        perf_log_dir, 
        tegrastats_log_dir,
        max_time_diff=1  # 1 second max difference in filenames
    )
    
    if not merged_data.empty:
        print(f"Successfully merged {len(merged_data)} data points.")
        print("Merged Data Sample:")
        print(merged_data.head())
        
        # Save to CSV
        merged_data.to_csv("merged_performance_data.csv", index=False)
        print("Saved merged data to 'merged_performance_data.csv'")
        
        # Provide some statistics
        scenarios = merged_data["scenario"].unique()
        print(f"Data includes {len(scenarios)} unique scenarios:")
        for scenario in scenarios:
            scenario_data = merged_data[merged_data["scenario"] == scenario]
            print(f"  {scenario}: {len(scenario_data)} data points")
    else:
        print("No data could be merged. Check the log files and directory paths.")

if __name__ == "__main__":
    main()