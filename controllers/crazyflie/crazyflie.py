#  ...........       ____  _ __
#  |  ,-^-,  |      / __ )(_) /_______________ _____  ___
#  | (  O  ) |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
#  | / ,..´  |    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
#     +.......   /_____/_/\__/\___/_/   \__,_/ /___/\___/

# MIT License

# Copyright (c) 2022 Bitcraze

# @file crazyflie_controllers_py.py
# Controls the crazyflie motors in webots in Python

"""crazyflie_controller_py controller."""


import os
import sys
from math import cos, sin

import cv2
import numpy as np
from controller import (
    GPS,
    Camera,
    DistanceSensor,
    Display,
    Gyro,
    InertialUnit,
    Keyboard,
    Motor,
    Robot,
)
from ultralytics import YOLO

controller_dir = os.path.dirname(os.path.abspath(__file__))
shared_python_path = os.path.normpath(
    os.path.join(controller_dir, '..', '..', 'controllers_shared', 'python_based')
)
if shared_python_path not in sys.path:
    sys.path.insert(0, shared_python_path)

try:
    from pid_controller import pid_velocity_fixed_height_controller
except ModuleNotFoundError:
    print(f"[crazyflie] Failed to import pid_controller from: {shared_python_path}")
    print(f"[crazyflie] __file__: {__file__}")
    print(f"[crazyflie] cwd: {os.getcwd()}")
    print(f"[crazyflie] sys.path[0:5]: {sys.path[:5]}")
    raise

FLYING_ATTITUDE = 1

# Initialize YOLO model for detection
# model_weights_path = os.path.normpath(os.path.join(controller_dir, '..', '..', 'weights', 'person_best.pt'))
model_weights_path = os.path.normpath(
    os.path.join(controller_dir, '..', '..', 'weights', 'pedestrian_best.pt')
)
if not os.path.exists(model_weights_path):
    raise FileNotFoundError(f"YOLO weights not found: {model_weights_path}")
model = YOLO(model_weights_path)

def detect_objects(image):
    """Run YOLO object detection on an image."""
    results = model.predict(source=image, verbose=False)
    return results

def draw_detections(image, results, display=None):
    """Draw bounding boxes on detected objects."""
    for detection in results[0].boxes:
        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, detection.xyxy[0])
        conf = detection.conf[0]  # Confidence score
        cls = int(detection.cls[0])  # Class index

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label with confidence score
        label = f"{model.names[cls]} {conf:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if display is not None:
            display.setColor(0x00FF00)
            display.drawRectangle(x1, y1, x2 - x1, y2 - y1)
            display.drawText(label, x1, max(0, y1 - 12))

    return image


def save_camera_image(camera, folder, index):
    # image = camera.getImage()
    file_name = f'image_{str(index).zfill(3)}.png'  # Format the index as a three-digit number
    camera.saveImage(os.path.join(folder, file_name), 100)
    print(f"{file_name} saved")

if __name__ == '__main__':

    robot = Robot()
    timestep = int(robot.getBasicTimeStep())
        
    # Initialize the image index
    image_index = 0
    # Initialize variables for timing
    last_save_time = 0  # Time when the last image was saved
    save_interval = 0.25  # Time interval in seconds between saved images

    ## Initialize motors
    m1_motor = robot.getDevice("m1_motor")
    m1_motor.setPosition(float('inf'))
    m1_motor.setVelocity(-1)
    m2_motor = robot.getDevice("m2_motor")
    m2_motor.setPosition(float('inf'))
    m2_motor.setVelocity(1)
    m3_motor = robot.getDevice("m3_motor")
    m3_motor.setPosition(float('inf'))
    m3_motor.setVelocity(-1)
    m4_motor = robot.getDevice("m4_motor")
    m4_motor.setPosition(float('inf'))
    m4_motor.setVelocity(1)

    ## Initialize Sensors
    imu = robot.getDevice("inertial_unit")
    imu.enable(timestep)
    gps = robot.getDevice("gps")
    gps.enable(timestep)
    gyro = robot.getDevice("gyro")
    gyro.enable(timestep)
    camera = robot.getDevice("camera")
    camera.enable(timestep)
    display = robot.getDevice("display")
    range_front = robot.getDevice("range_front")
    range_front.enable(timestep)
    range_left = robot.getDevice("range_left")
    range_left.enable(timestep)
    range_back = robot.getDevice("range_back")
    range_back.enable(timestep)
    range_right = robot.getDevice("range_right")
    range_right.enable(timestep)
    
    # Get camera parameters
    camera_width = camera.getWidth()
    camera_height = camera.getHeight()

    ## Get keyboard
    keyboard = Keyboard()
    keyboard.enable(timestep)

    ## Initialize variables

    past_x_global = 0
    past_y_global = 0
    past_time = robot.getTime()

    # Crazyflie velocity PID controller
    PID_CF = pid_velocity_fixed_height_controller()
    PID_update_last_time = robot.getTime()
    sensor_read_last_time = robot.getTime()

    height_desired = FLYING_ATTITUDE
    takeoff_tolerance = 0.005
    takeoff_done = False

    print("\n")

    print("====== Controls =======\n\n")

    print(" The Crazyflie can be controlled from your keyboard!\n")
    print(" All controllable movement is in body coordinates\n")
    print("- Use the up, back, right and left button to move in the horizontal plane\n")
    print("- Use Q and E to rotate around yaw ")
    print("- Use W and S to go up and down\n ")

    print("\n====== Crazyflie Drone with YOLOv5 Detection ======\n")
    
    # Main loop:
    while robot.step(timestep) != -1:
        current_time = robot.getTime()
        dt = current_time - past_time
        actual_state = {}

        ## Get sensor data
        roll = imu.getRollPitchYaw()[0]
        pitch = imu.getRollPitchYaw()[1]
        yaw = imu.getRollPitchYaw()[2]
        yaw_rate = gyro.getValues()[2]
        altitude = gps.getValues()[2]
        x_global = gps.getValues()[0]
        v_x_global = (x_global - past_x_global)/dt
        y_global = gps.getValues()[1]
        v_y_global = (y_global - past_y_global)/dt

        ## Get body fixed velocities
        cosyaw = cos(yaw)
        sinyaw = sin(yaw)
        v_x = v_x_global * cosyaw + v_y_global * sinyaw
        v_y = - v_x_global * sinyaw + v_y_global * cosyaw

        ## Initialize values
        desired_state = [0, 0, 0, 0]
        forward_desired = 0
        sideways_desired = 0
        desired_yaw_rate = 0
        height_diff_desired = 0


        if not takeoff_done and abs(altitude - FLYING_ATTITUDE) > takeoff_tolerance:
            height_desired += height_diff_desired * dt
            # print(f"Taking off to {height_desired} m")
        else:           
            takeoff_done = True
            
            # Capture camera image
            raw_image = camera.getImage()
            display_image = display.imageNew(raw_image, Display.BGRA, camera_width, camera_height)
            display.imagePaste(display_image, 0, 0, False)
            display.imageDelete(display_image)
            image = np.frombuffer(raw_image, dtype=np.uint8).reshape((camera_height, camera_width, 4))
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)  # Convert Webots image format to OpenCV BGR

            # Run object detection
            results = detect_objects(image)
            detected_image = draw_detections(image.copy(), results, display)

            # If at least one detection is available, compute the yaw error.
            if results and len(results[0].boxes) > 0:
                # Select the first detection (or choose the one with highest confidence).
                detection = results[0].boxes[0]
                x1, y1, x2, y2 = map(int, detection.xyxy[0])
                bbox_center_x = (x1 + x2) / 2

                # Compute horizontal error relative to the camera center.
                image_center_x = camera_width / 2
                error_x = image_center_x - bbox_center_x

                # Compute desired yaw rate using a proportional controller.
                # (Adjust Kp_yaw as necessary for smooth behavior.)
                Kp_yaw = 0.005
                desired_yaw_rate = Kp_yaw * error_x

                # (Optional) Save the annotated output image.
                if current_time - last_save_time >= save_interval:
                    cv2.imwrite("detected_output.png", detected_image)
                    last_save_time = current_time
            else:
                # If no object is detected, you can keep desired_yaw_rate at 0
                # or use the previous value if you want to hold the last heading.
                desired_yaw_rate = 0
        
        ## PID velocity controller with fixed height
        motor_power = PID_CF.pid(dt, forward_desired, sideways_desired,
                                desired_yaw_rate, height_desired,
                                roll, pitch, yaw_rate,
                                altitude, v_x, v_y)

        m1_motor.setVelocity(-motor_power[0])
        m2_motor.setVelocity(motor_power[1])
        m3_motor.setVelocity(-motor_power[2])
        m4_motor.setVelocity(motor_power[3])

        past_time = current_time
        past_x_global = x_global
        past_y_global = y_global
