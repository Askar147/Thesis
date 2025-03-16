import pandas as pd

# Read the CSV file
df = pd.read_csv('merged_performance_data.csv')

# Group by container_id and sum the cycles
cycles_by_container = df.groupby('container_id')['avg_cpu'].sum().reset_index()

# Display the result
print(cycles_by_container)

# Optional: If you want to sort by total cycles in descending order
cycles_by_container_sorted = cycles_by_container.sort_values('avg_cpu', ascending=False)
print("\nSorted by total instructions (descending):")
print(cycles_by_container_sorted)