from controller import GPS, Supervisor
import math
import random
import os

# Create the Supervisor instance instead of Robot
robot = Supervisor()
GPS = GPS('gps')

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# Get the wooden box node
wooden_box = robot.getSelf()

# Initialize position variables
x = 0.0
y = 0.0
z = 0.74  # Fixed z position
rotation_angle = 0

# Table dimensions
table_width = 1.25
table_length = 0.55
x_min, x_max = -table_width/2, table_width/2
y_min, y_max = -table_length/2, table_length/2

# Remove the file if it already exists
try:
    os.remove('gps_values.txt')
except FileNotFoundError:
    pass

while robot.step(timestep) != -1:
    # Read the GPS values
    gps_values = GPS.getValues()
    
    # Save the GPS values and the rotation angle to a file
    with open('gps_values.txt', 'a') as f:
        f.write(f"{gps_values} {rotation_angle}\n")

    # Generate new random position and rotation every timestep
    x = random.uniform(x_min, x_max)
    y = random.uniform(y_min, y_max)
    rotation_angle = random.uniform(0, 2 * math.pi)
    
    # Set new position and rotation
    wooden_box.getField('translation').setSFVec3f([x, y, z])
    wooden_box.getField('rotation').setSFRotation([0, 0, 1, rotation_angle])

