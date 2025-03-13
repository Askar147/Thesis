import xml.etree.ElementTree as ET
import random

def generate_route_file(output_file, num_vehicles, scenario, simulation_period=3600):
    """
    Generates a SUMO route file with a given number of vehicles and departure pattern.
    
    Parameters:
      output_file (str): The file path to save the route file.
      num_vehicles (int): Number of vehicles to generate.
      scenario (str): Departure pattern, either "uniform" or "rush-hour".
      simulation_period (int): Total simulation duration in seconds (default is 3600).
    """
    # Create the root element <routes>
    routes = ET.Element("routes")

    # Define a vehicle type (this can be expanded as needed)
    ET.SubElement(routes, "vType", id="car", accel="1.0", decel="4.5",
                  length="5", minGap="2.5", maxSpeed="25", guiShape="passenger")
    
    # Define a route.
    # For simplicity, assume that the network contains an edge with id "1" that vehicles will use.
    ET.SubElement(routes, "route", id="route0", edges="1")

    # Generate vehicle elements with departure times.
    for i in range(num_vehicles):
        if scenario == "uniform":
            # Uniformly distribute departures over the entire simulation period.
            depart_time = random.uniform(0, simulation_period)
        elif scenario == "rush-hour":
            # Rush-hour: cluster departures in a short period (e.g., first 600 seconds)
            depart_time = random.uniform(0, 600)
        else:
            # Default to uniform if unknown scenario.
            depart_time = random.uniform(0, simulation_period)
        
        ET.SubElement(routes, "vehicle", id=f"veh{i}", type="car", route="route0",
                      depart=str(round(depart_time, 2)))
    
    # Write the XML tree to a file with an XML declaration.
    tree = ET.ElementTree(routes)
    tree.write(output_file, encoding="UTF-8", xml_declaration=True)
    print(f"Generated route file '{output_file}' with {num_vehicles} vehicles for scenario '{scenario}'.")

if __name__ == "__main__":
    # For reproducibility, set a random seed.
    random.seed(42)

    # Define simulation period (in seconds)
    simulation_period = 600  # 1 hour simulation

    # Generate route files for different densities
    generate_route_file("low_density.rou.xml", 50, "uniform", simulation_period)
    generate_route_file("medium_density.rou.xml", 200, "uniform", simulation_period)
    generate_route_file("high_density.rou.xml", 500, "uniform", simulation_period)

    # Generate a rush-hour scenario route file (e.g., high density but departures clustered in the first 600 sec)
    generate_route_file("rush_hour.rou.xml", 500, "rush-hour", simulation_period)
