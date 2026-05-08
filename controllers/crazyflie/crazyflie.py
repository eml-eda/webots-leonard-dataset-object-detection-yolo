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
import json
import base64
import threading
import urllib.error
import urllib.request
from math import atan, cos, sin, tan

import cv2
import numpy as np
from controller import (
    GPS,
    Camera,
    Display,
    DistanceSensor,
    Gyro,
    InertialUnit,
    Keyboard,
    Motor,
    Robot,
)

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
DETECTION_URL = os.getenv('CF_DETECTION_URL', 'http://127.0.0.1:8765/detect')
DETECTION_INPUT_SIZE = max(8, int(os.getenv('CF_DETECTION_INPUT_SIZE', '128')))
DETECTION_TIMEOUT_S = max(0.01, float(os.getenv('CF_DETECTION_TIMEOUT_S', '0.25')))
DETECTION_RETRY_S = max(0.05, float(os.getenv('CF_DETECTION_RETRY_S', '0.5')))
CAMERA_PERIOD_MS = 200
OPEN_LOOP_YAW_RATE = max(0.05, float(os.getenv('CF_OPEN_LOOP_YAW_RATE', '0.6')))
MAX_OPEN_LOOP_YAW_DELTA = max(0.05, float(os.getenv('CF_MAX_OPEN_LOOP_YAW_DELTA', '1.2')))
YAW_STOP_TOLERANCE = max(0.001, float(os.getenv('CF_YAW_STOP_TOLERANCE', '0.01')))

def _scale_detections(detections, src_width, src_height, dst_width, dst_height):
    scale_x = dst_width / max(src_width, 1)
    scale_y = dst_height / max(src_height, 1)
    scaled = []
    for det in detections:
        scaled.append({
            'x1': int(det['x1'] * scale_x),
            'y1': int(det['y1'] * scale_y),
            'x2': int(det['x2'] * scale_x),
            'y2': int(det['y2'] * scale_y),
            'conf': float(det.get('conf', 0.0)),
            'cls': int(det.get('cls', -1)),
            'label': str(det.get('label', 'object')),
        })
    return scaled


def _wrap_angle(angle):
    """Wrap an angle in radians to [-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def detect_objects_remote(image, frame_id):
    """Send a single RGB frame to a remote service and receive detections."""
    resized_bgr = cv2.resize(image, (DETECTION_INPUT_SIZE, DETECTION_INPUT_SIZE))
    resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

    payload = {
        'frame_id': frame_id,
        'encoding': 'raw_rgb24',
        'width': DETECTION_INPUT_SIZE,
        'height': DETECTION_INPUT_SIZE,
        'channels': 3,
        'image_b64': base64.b64encode(resized_rgb.tobytes()).decode('ascii'),
    }
    request_data = json.dumps(payload).encode('utf-8')
    request = urllib.request.Request(
        DETECTION_URL,
        data=request_data,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )

    with urllib.request.urlopen(request, timeout=DETECTION_TIMEOUT_S) as response:
        response_payload = json.loads(response.read().decode('utf-8'))

    detections = response_payload.get('detections', [])
    width = int(response_payload.get('image_width', DETECTION_INPUT_SIZE))
    height = int(response_payload.get('image_height', DETECTION_INPUT_SIZE))
    return detections, width, height


class RemoteDetectionClient:
    """Background HTTP client so controller loop is never blocked by network calls."""

    def __init__(self):
        self._lock = threading.Lock()
        self._request_in_flight = False
        self._pending_result = None
        self._last_error = None

    def _worker(self, image_bgr, frame_id):
        try:
            detections, width, height = detect_objects_remote(image_bgr, frame_id)
            with self._lock:
                self._pending_result = (detections, width, height)
                self._last_error = None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError) as exc:
            with self._lock:
                self._last_error = str(exc)
        finally:
            with self._lock:
                self._request_in_flight = False

    def submit(self, image_bgr, frame_id):
        with self._lock:
            if self._request_in_flight:
                return False
            self._request_in_flight = True

        worker = threading.Thread(
            target=self._worker,
            args=(image_bgr.copy(), frame_id),
            daemon=True,
        )
        worker.start()
        return True

    def pop_result(self):
        with self._lock:
            result = self._pending_result
            self._pending_result = None
            return result

    def pop_error(self):
        with self._lock:
            error = self._last_error
            self._last_error = None
            return error

    def in_flight(self):
        with self._lock:
            return self._request_in_flight

if __name__ == '__main__':

    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    camera_period_ms = max(timestep, CAMERA_PERIOD_MS)
    # image_process_interval = max(
    #     0.0,
    #     float(os.getenv('CF_IMAGE_PROCESS_INTERVAL', '0.1'))
    # )
        
    # Initialize the image index
    image_index = 0
    # Initialize variables for timing
    last_save_time = 0  # Time when the last image was saved
    save_interval = 0.25  # Time interval in seconds between saved images
    desired_yaw_rate_cmd = 0.0
    fixed_target_detection = None
    planned_yaw_delta = None
    yaw_target = None
    one_shot_completed = False
    remote_client = RemoteDetectionClient()
    next_remote_retry_time = 0.0

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
    camera.enable(camera_period_ms)
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
    camera_fov = camera.getFov()
    focal_length_px = camera_width / (2.0 * tan(camera_fov / 2.0))

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

    print("\n====== Crazyflie Drone with Externalizable YOLO Detection ======\n")
    print("[crazyflie] Detection backend: remote")
    print("[crazyflie] Effective config:")
    print(f"  CF_DETECTION_URL={DETECTION_URL}")
    print(f"  CF_DETECTION_INPUT_SIZE={DETECTION_INPUT_SIZE}")
    print(f"  CF_DETECTION_TIMEOUT_S={DETECTION_TIMEOUT_S:.3f}")
    print(f"  CF_DETECTION_RETRY_S={DETECTION_RETRY_S:.3f}")
    print(f"  CF_CAMERA_PERIOD_MS={camera_period_ms}")
    print(f"  CF_OPEN_LOOP_YAW_RATE={OPEN_LOOP_YAW_RATE:.3f}")
    print(f"  CF_MAX_OPEN_LOOP_YAW_DELTA={MAX_OPEN_LOOP_YAW_DELTA:.3f}")
    print(f"  CF_YAW_STOP_TOLERANCE={YAW_STOP_TOLERANCE:.4f}")
    
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
        desired_yaw_rate = desired_yaw_rate_cmd
        height_diff_desired = 0


        if not takeoff_done and abs(altitude - FLYING_ATTITUDE) > takeoff_tolerance:
            height_desired += height_diff_desired * dt
            # print(f"Taking off to {height_desired} m")
        else:           
            takeoff_done = True

            # # Run camera rendering and YOLO at a lower, configurable rate.
            # Capture camera image
            raw_image = camera.getImage()
            display_image = display.imageNew(raw_image, Display.BGRA, camera_width, camera_height)
            display.imagePaste(display_image, 0, 0, False)
            display.imageDelete(display_image)
            image = np.frombuffer(raw_image, dtype=np.uint8).reshape((camera_height, camera_width, 4))
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)  # Convert Webots image format to OpenCV BGR

            if (
                not one_shot_completed
                and planned_yaw_delta is None
                and current_time >= next_remote_retry_time
            ):
                submitted = remote_client.submit(image, image_index)
                if submitted:
                    next_remote_retry_time = current_time + DETECTION_RETRY_S
                    image_index += 1

            remote_error = remote_client.pop_error()
            if remote_error:
                print(f"[crazyflie] remote detection failed: {remote_error}")

            remote_result = remote_client.pop_result()
            if remote_result is not None:
                detections, infer_width, infer_height = remote_result
                scaled = _scale_detections(
                    detections,
                    infer_width,
                    infer_height,
                    camera_width,
                    camera_height,
                )
                if scaled:
                    fixed_target_detection = scaled[0]
                    bbox_center_x = (fixed_target_detection['x1'] + fixed_target_detection['x2']) / 2.0
                    image_center_x = camera_width / 2.0
                    # Positive angle means rotate left, negative means rotate right.
                    planned_yaw_delta = atan((image_center_x - bbox_center_x) / max(focal_length_px, 1e-6))
                    if planned_yaw_delta > MAX_OPEN_LOOP_YAW_DELTA:
                        planned_yaw_delta = MAX_OPEN_LOOP_YAW_DELTA
                    elif planned_yaw_delta < -MAX_OPEN_LOOP_YAW_DELTA:
                        planned_yaw_delta = -MAX_OPEN_LOOP_YAW_DELTA
                    yaw_target = _wrap_angle(yaw + planned_yaw_delta)
                    print(
                        f"[crazyflie] Planned single-shot yaw delta: {planned_yaw_delta:.3f} rad "
                        f"({planned_yaw_delta * 180.0 / np.pi:.1f} deg)"
                    )
                elif planned_yaw_delta is None:
                    # Keep retrying if no box has ever been received yet.
                    next_remote_retry_time = current_time + DETECTION_RETRY_S

            # Execute the pre-planned single-shot yaw in open loop.
            if yaw_target is not None:
                yaw_error = _wrap_angle(yaw_target - yaw)
                safe_dt = max(dt, 1e-6)
                if abs(yaw_error) > YAW_STOP_TOLERANCE:
                    raw_yaw_rate = yaw_error / safe_dt
                    if raw_yaw_rate > OPEN_LOOP_YAW_RATE:
                        desired_yaw_rate_cmd = OPEN_LOOP_YAW_RATE
                    elif raw_yaw_rate < -OPEN_LOOP_YAW_RATE:
                        desired_yaw_rate_cmd = -OPEN_LOOP_YAW_RATE
                    else:
                        desired_yaw_rate_cmd = raw_yaw_rate
                else:
                    desired_yaw_rate_cmd = 0.0
                    yaw_target = None
                    planned_yaw_delta = 0.0
                    fixed_target_detection = None
                    one_shot_completed = True
                    print('[crazyflie] Open-loop yaw plan completed.')
            else:
                desired_yaw_rate_cmd = 0.0

            # Optional: save the raw camera image periodically (without bounding boxes).
            if current_time - last_save_time >= save_interval:
                cv2.imwrite("captured_output.png", image)
                last_save_time = current_time

            desired_yaw_rate = desired_yaw_rate_cmd
        
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
