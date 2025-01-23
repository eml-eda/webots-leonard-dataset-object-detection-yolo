# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, Camera
import os

# create the Robot instance.
camera_robot = Robot()

# get the time step of the current world.
timestep = int(camera_robot.getBasicTimeStep())
camera = Camera('camera')

camera.enable(timestep)
i = 0

if not os.path.exists("all_images"):
    os.makedirs("all_images")

while camera_robot.step(timestep) != -1:
    # Read the camera image
    image = camera.getImage()
    # Display the image
    camera.saveImage(f'all_images/image_{i}.png', 100)
    print(f"Image_{i} saved")
    i += 1

