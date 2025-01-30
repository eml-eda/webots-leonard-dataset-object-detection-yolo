from controller import GPS, Supervisor
import math
import random
import os


def get_object(robot: Supervisor, object_name: str) -> Supervisor:
    """
    Get the object node from the supervisor
    """
    children = robot.getField('children')
    num_children = children.getCount()

    for i in range(num_children):
        node = children.getMFNode(i)
        name = node.getField('name').getSFString()
        if name == object_name:
            return node


if __name__ == "__main__":
    # Create the Supervisor instance instead of Robot
    robot = Supervisor()
    GPS = GPS('gps')

    # get the time step of the current world.
    timestep = int(robot.getBasicTimeStep())

    # Get the wooden box node
    kuka_box_robot = robot.getSelf()

    # Get the height of the kuka box
    kuka_box_height = get_object(kuka_box_robot, 'Kuka box').getField('size').getSFVec3f()[2]

    # Initialize position variables
    x = kuka_box_robot.getField('translation').getSFVec3f()[0]
    y = kuka_box_robot.getField('translation').getSFVec3f()[1]
    z = kuka_box_robot.getField('translation').getSFVec3f()[2]
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
        gps_values[2] = gps_values[2] - kuka_box_height/2  # Needed because the GPS sensor is at the center of the kuka_box, not at the bottom 

        # Save the GPS values and the rotation angle to a file
        with open('gps_values.txt', 'a') as f:
            f.write(f"{gps_values} {rotation_angle}\n")

        # Generate new random position and rotation every timestep
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        rotation_angle = random.uniform(0, 2 * math.pi)
        
        # Set new position and rotation
        kuka_box_robot.getField('translation').setSFVec3f([x, y, z])
        kuka_box_robot.getField('rotation').setSFRotation([0, 0, 1, rotation_angle])

