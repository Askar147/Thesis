import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the aggregated CSV file (adjust the filename/path if needed)
csv_file = "aggregated_log_data.csv"
df = pd.read_csv(csv_file, parse_dates=["timestamp"])

# Check the first few rows
print(df.head())

# Convert memory usage fields if necessary (example: remove "MiB" and convert to float)
def parse_mem(mem_str):
    try:
        # Remove non-digit characters and convert to float (assuming unit is MiB)
        return float(re.sub(r'[^\d\.]', '', mem_str))
    except Exception:
        return None

import re
if "mem_usage" in df.columns:
    df["mem_usage_val"] = df["mem_usage"].apply(parse_mem)
if "mem_limit" in df.columns:
    df["mem_limit_val"] = df["mem_limit"].apply(parse_mem)

# ---------------------------
# Plot 1: CPU Usage Over Time
# ---------------------------
plt.figure(figsize=(12, 6))
for container_id in df["container_id"].unique():
    container_id_df = df[df["container_id"] == container_id]
    plt.plot(container_id_df["timestamp"], container_id_df["cpu_percent"], label=container_id)
plt.xlabel("Time")
plt.ylabel("CPU %")
plt.title("CPU Usage Over Time by container_id")
plt.legend()
plt.tight_layout()
plt.savefig("cpu_usage_over_time.png")
plt.show()

# ---------------------------
# Plot 2: Memory Usage Over Time
# ---------------------------
plt.figure(figsize=(12, 6))
for container_id in df["container_id"].unique():
    container_id_df = df[df["container_id"] == container_id]
    plt.plot(container_id_df["timestamp"], container_id_df["mem_usage_val"], label=container_id)
plt.xlabel("Time")
plt.ylabel("Memory Usage (MiB)")
plt.title("Memory Usage Over Time by container_id")
plt.legend()
plt.tight_layout()
plt.savefig("memory_usage_over_time.png")
plt.show()

# ---------------------------
# Plot 3: Scatter Plot: CPU % vs. CPU Temperature (from tegrastats)
# ---------------------------
if "cpu_temp" in df.columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="cpu_percent", y="cpu_temp", hue="container_id")
    plt.xlabel("CPU %")
    plt.ylabel("CPU Temperature (°C)")
    plt.title("CPU % vs. CPU Temperature by container_id")
    plt.tight_layout()
    plt.savefig("cpu_vs_temp.png")
    plt.show()

# ---------------------------
# Plot 4: Scatter Plot: Memory Usage vs. Cycles (from perf_stat if available)
# ---------------------------
if "cycles" in df.columns and "mem_usage_val" in df.columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="mem_usage_val", y="cycles", hue="container_id")
    plt.xlabel("Memory Usage (MiB)")
    plt.ylabel("Cycles")
    plt.title("Memory Usage vs. Cycles by container_id")
    plt.tight_layout()
    plt.savefig("mem_vs_cycles.png")
    plt.show()

# ---------------------------
# Plot 5: Grouped Box Plot: CPU % by container_id
# ---------------------------
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="container_id", y="cpu_percent")
plt.xlabel("container_id")
plt.ylabel("CPU %")
plt.title("Distribution of CPU % by container_id")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("cpu_boxplot_by_container_id.png")
plt.show()

# Additional plots can be added similarly:
# For example, you might plot network I/O over time, block I/O, etc.

print("Plots saved and displayed.")
