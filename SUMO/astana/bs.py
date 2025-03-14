import sumolib
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Path to your Astana network file
net_file = "astana.net.xml"

# Load the SUMO network using sumolib
try:
    net = sumolib.net.readNet(net_file)
except Exception as e:
    print("Error reading network file:", e)
    exit()

# Extract all node coordinates using getCoord()
nodes = []
node_ids = []
for node in net.getNodes():
    x, y = node.getCoord()
    nodes.append([x, y])
    node_ids.append(node.getID())

nodes = np.array(nodes)
print(f"Extracted {len(nodes)} nodes from the network.")

# Decide on the number of base stations (e.g., 3)
num_bs = 64

# Apply k-means clustering to get base station locations
kmeans = KMeans(n_clusters=num_bs, random_state=42).fit(nodes)
centroids = kmeans.cluster_centers_

# Determine the spread for each cluster to set a coverage radius
coverage_radii = []
labels = kmeans.labels_
for i in range(num_bs):
    cluster_points = nodes[labels == i]
    centroid = centroids[i]
    distances = np.linalg.norm(cluster_points - centroid, axis=1)
    radius = distances.max() * 1.2  # add a 20% safety margin
    coverage_radii.append(radius)
    
# Print suggested base station locations and coverage radii
print("Suggested Base Station Locations and Coverage Radii:")
for i, centroid in enumerate(centroids):
    print(f"BS{i+1}: Location = ({centroid[0]:.2f}, {centroid[1]:.2f}), Coverage Radius = {coverage_radii[i]:.2f} meters")

# Visualize the network nodes and base station locations
plt.figure(figsize=(10, 8))
plt.scatter(nodes[:, 0], nodes[:, 1], s=2, label='Network Nodes')
plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='x', s=100, label='Base Station Locations')
for i, centroid in enumerate(centroids):
    circle = plt.Circle((centroid[0], centroid[1]), coverage_radii[i], color='red', fill=False, linestyle='--')
    plt.gca().add_patch(circle)
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Base Station Placement on Astana Network")
plt.legend()
plt.show()
