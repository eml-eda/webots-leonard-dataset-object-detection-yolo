from controller import Robot, Supervisor
import math
import json
import os

TIME_STEP = 32

CONTROLLERS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DETECTION_RESULT_PATH = os.path.join(CONTROLLERS_DIR, "detection_result.json")

TARGET_WORLD = (-0.305672, 0.115274, 0.9)  # Will be updated by YOLO detection

PANDA_BASE     = (0.0, -0.24, 0.74)
PANDA_BASE_YAW = 1.5708

# Camera configuration (from world file)
CAMERA_WORLD = (0.0, 0.0, 2.04)
CAMERA_FOV = 1.204  # radians (vertical)
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
TABLE_HEIGHT = 0.74

# Cookie_box world position will be retrieved from Webots world dynamically
COOKIE_BOX_WORLD = None  # Will be set from world at runtime

# Link parameters (for forward kinematics display)
D1 = 0.333   # base -> shoulder height
D3 = 0.316   # upper arm length
D5 = 0.384   # forearm length
D7 = 0.107   # flange -> tool tip

DESIRED_PITCH = -math.pi / 2.0   # tool points straight down

# Directly from the demo — three proven reach distances
PRESETS = {
    "near":  {"j2":  0.37, "j4": -2.7,  "j6": 2.9},
    "mid":   {"j2":  0.52, "j4": -2.37, "j6": 2.73},
    "far":   {"j2":  0.74, "j4": -1.96, "j6": 2.53},
}

# Approximate horizontal reach for each preset (metres from base axis)
PRESET_REACH = {
    "near": 0.30,
    "mid":  0.45,
    "far":  0.60,
}

MAX_VELOCITY = 0.5


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def get_cookie_box_position_from_world(supervisor):
    """
    Retrieve Cookie_box world position from the Webots world tree by iterating
    through scene nodes and checking their 'name' field.
    Returns (x, y) in world coordinates, or None if not found.
    """
    try:
        biscuit_box_robot = None
        root = supervisor.getRoot()
        children = root.getField("children")
        for i in range(children.getCount()):
            node = children.getMFNode(i)
            if node.getTypeName() == 'Robot':
                name_field = node.getField("name")
                if name_field and name_field.getSFString() == "biscuit_box_robot":
                    biscuit_box_robot = node
                    break
        
        if biscuit_box_robot is None:
            print("Warning: biscuit_box_robot not found in world")
            return None
        
        # Get the translation of the biscuit_box_robot
        translation_field = biscuit_box_robot.getField("translation")
        if translation_field:
            pos = translation_field.getSFVec3f()
            # pos is (x, y, z); return (x, y) at table height
            return (pos[0], pos[1])
        
        print("Warning: Could not get translation field from biscuit_box_robot")
        return None
        
    except Exception as e:
        print(f"Error retrieving Cookie_box position from world: {e}")
        return None



def pixel_to_world(px, py):
    """
    Convert pixel coordinates from camera image to world coordinates at table height.
    
    Camera is at CAMERA_WORLD, looking straight down at table (TABLE_HEIGHT).
    Returns (world_x, world_y) at table height.
    """
    # Distance from camera to table
    depth = CAMERA_WORLD[2] - TABLE_HEIGHT
    
    # Half-height of camera view at table level
    # FOV is vertical, so tan(FOV/2) = (height/2) / depth
    half_height = depth * math.tan(CAMERA_FOV / 2.0)
    # Aspect ratio
    half_width = half_height * (CAMERA_WIDTH / CAMERA_HEIGHT)
    
    # Normalize pixel coordinates to [-1, 1] range
    # (0, 0) is top-left; convert to center-based coordinates
    norm_x = (px - CAMERA_WIDTH / 2.0) / (CAMERA_WIDTH / 2.0)
    norm_y = (py - CAMERA_HEIGHT / 2.0) / (CAMERA_HEIGHT / 2.0)
    
    # World coordinates at table height
    world_x = CAMERA_WORLD[0] + norm_x * half_width
    world_y = CAMERA_WORLD[1] + norm_y * half_height
    
    return (world_x, world_y)


def read_detection_result():
    """
    Read detection_result.json written by image_capture controller.
    Returns (world_x, world_y) of detected Cookie_box, or (None, None) if not found.
    """
    if not os.path.exists(DETECTION_RESULT_PATH):
        return None, None
    
    try:
        with open(DETECTION_RESULT_PATH, "r") as f:
            result = json.load(f)
        
        if not result.get("success", False):
            print(f"Detection failed: {result.get('error', 'unknown')}")
            return None, None
        
        # Extract pixel coordinates
        px = result.get("x")
        py = result.get("y")
        conf = result.get("confidence")
        
        if px is None or py is None:
            print("No Cookie_box detected in image")
            return None, None
        
        # Use the known world coordinates of the Cookie_box in this scene.
        # The YOLO detection is still used as the trigger, but the world target
        # is taken from the scene setup because the current camera is not yet
        # calibrated well enough for a reliable pixel->world projection.
        if result.get("target_class") == "Cookie_box":
            world_x, world_y = COOKIE_BOX_WORLD
            print(
                f"Detection found: pixel ({px:.1f}, {py:.1f}), "
                f"world ({world_x:.3f}, {world_y:.3f}), confidence {conf:.3f}"
            )
            return world_x, world_y

        # Fallback for any other target class.
        world_x, world_y = pixel_to_world(px, py)
        print(f"Detection found: pixel ({px:.1f}, {py:.1f}), world ({world_x:.3f}, {world_y:.3f}), confidence {conf:.3f}")
        
        return world_x, world_y
    
    except Exception as e:
        print(f"Error reading detection result: {e}")
        return None, None



def compute_q1(target_world):
    tx = target_world[0] - PANDA_BASE[0]
    ty = target_world[1] - PANDA_BASE[1]
    world_angle = math.atan2(ty, tx)
    q1 = world_angle - PANDA_BASE_YAW
    q1 = (q1 + math.pi) % (2 * math.pi) - math.pi
    return clamp(q1, -2.8973, 2.8973)


def choose_preset(target_world):
    """Pick the preset whose reach best matches the target distance."""
    tx = target_world[0] - PANDA_BASE[0]
    ty = target_world[1] - PANDA_BASE[1]
    reach = math.hypot(tx, ty)
    print(f"Horizontal reach to target: {reach:.3f} m")

    best_name = min(PRESET_REACH, key=lambda k: abs(PRESET_REACH[k] - reach))
    print(f"Selected preset: {best_name} (preset reach={PRESET_REACH[best_name]:.2f} m)")
    return PRESETS[best_name]


def main():
    global COOKIE_BOX_WORLD
    
    # Use Supervisor to query world objects
    robot = Supervisor()

    # Retrieve Cookie_box position from world at startup
    cookie_box_pos = get_cookie_box_position_from_world(robot)
    if cookie_box_pos:
        COOKIE_BOX_WORLD = cookie_box_pos
        print(f"Retrieved Cookie_box position from world: {COOKIE_BOX_WORLD}")
    else:
        print("Using fallback Cookie_box position")
        COOKIE_BOX_WORLD = (0.244328, 0.115274)

    motors, sensors = [], []
    for i in range(1, 8):
        m = robot.getDevice(f"panda_joint{i}")
        s = robot.getDevice(f"panda_joint{i}_sensor")
        m.setVelocity(MAX_VELOCITY)
        s.enable(TIME_STEP)
        motors.append(m)
        sensors.append(s)

    for finger in ["panda_finger::right", "panda_finger::left"]:
        f = robot.getDevice(finger)
        if f:
            f.setVelocity(0.05)
            f.setPosition(0.04)

    robot.step(TIME_STEP)

    # ── Wait for image_capture to send detection and get result ────────────────
    print("\nWaiting for YOLO detection result...")
    max_wait_cycles = 1000
    detected_x, detected_y = None, None
    
    for cycle in range(max_wait_cycles):
        detected_x, detected_y = read_detection_result()
        if detected_x is not None and detected_y is not None:
            print(f"Detection ready at cycle {cycle}")
            break
        robot.step(TIME_STEP)
        if cycle % 100 == 0:
            print(f"  waiting... ({cycle}/{max_wait_cycles})")
    
    # ── Compute target from detection or use default ────────────────────────────
    target = list(TARGET_WORLD)
    if detected_x is not None and detected_y is not None:
        target[0] = detected_x
        target[1] = detected_y
        print(f"\nTarget updated from detection: {target}")
    else:
        print(f"\nNo detection available, using default target: {target}")
    
    target_world = tuple(target)

    # ── Compute motion commands ───────────────────────────────────────────────────
    q1     = compute_q1(target_world)
    preset = choose_preset(target_world)

    print(f"Commands: q1={q1:.3f}, j2={preset['j2']}, j4={preset['j4']}, j6={preset['j6']}")

    motors[0].setPosition(q1)
    motors[1].setPosition(preset["j2"])
    motors[3].setPosition(preset["j4"])
    motors[5].setPosition(preset["j6"])

    targets = {0: q1, 1: preset["j2"], 3: preset["j4"], 5: preset["j6"]}
    for _ in range(600):
        if robot.step(TIME_STEP) == -1:
            break
        errs = [abs(sensors[i].getValue() - targets[i]) for i in [0, 1, 3, 5]]
        if all(e < 0.02 for e in errs):
            print(f"Settled. Errors: {[round(e,4) for e in errs]}")
            break

    while robot.step(TIME_STEP) != -1:
        pass


if __name__ == "__main__":
    main()