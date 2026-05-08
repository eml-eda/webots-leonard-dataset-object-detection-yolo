from controller import Robot
import math

TIME_STEP = 32

TARGET_WORLD = (-0.305672, 0.115274, 0.9)

PANDA_BASE     = (0.0, -0.24, 0.74)
PANDA_BASE_YAW = 1.5708

# Directly from the demo — three proven reach distances
PRESETS = {
    "near":  {"j2":  0.37, "j4": -2.7,  "j6": 2.9},
    "mid":   {"j2":  0.52, "j4": -2.37, "j6": 2.73},
    "far":   {"j2":  0.74, "j4": -1.96, "j6": 2.53},
}

# Approximate horizontal reach for each preset (metres from base axis)
# Measure these empirically in Webots by checking where the gripper lands
PRESET_REACH = {
    "near": 0.30,
    "mid":  0.45,
    "far":  0.60,
}

MAX_VELOCITY = 0.5


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


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
    robot = Robot()

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


    q1     = compute_q1(TARGET_WORLD)
    preset = choose_preset(TARGET_WORLD)

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