#!/usr/bin/env python3

from controller import Supervisor
import math
import optparse


def parse_trajectory(trajectory_option):
    points = []
    for token in trajectory_option.split(','):
        coords = token.split()
        if len(coords) != 2:
            continue
        points.append((float(coords[0]), float(coords[1])))
    return points


def build_cumulative_distances(points):
    cumulative = [0.0]
    for i in range(len(points) - 1):
        dx = points[i + 1][0] - points[i][0]
        dy = points[i + 1][1] - points[i][1]
        cumulative.append(cumulative[-1] + math.hypot(dx, dy))
    return cumulative


def interpolate_position_and_heading(points, cumulative, traveled):
    if len(points) < 2:
        return points[0][0], points[0][1], 0.0

    total_length = cumulative[-1]
    if traveled <= 0.0:
        heading = math.atan2(points[1][1] - points[0][1], points[1][0] - points[0][0])
        return points[0][0], points[0][1], heading

    if traveled >= total_length:
        heading = math.atan2(points[-1][1] - points[-2][1], points[-1][0] - points[-2][0])
        return points[-1][0], points[-1][1], heading

    segment_index = 0
    for i in range(len(cumulative) - 1):
        if cumulative[i + 1] >= traveled:
            segment_index = i
            break

    segment_start = points[segment_index]
    segment_end = points[segment_index + 1]
    segment_length = cumulative[segment_index + 1] - cumulative[segment_index]
    if segment_length <= 0.0:
        ratio = 0.0
    else:
        ratio = (traveled - cumulative[segment_index]) / segment_length

    x = segment_start[0] + ratio * (segment_end[0] - segment_start[0])
    y = segment_start[1] + ratio * (segment_end[1] - segment_start[1])
    heading = math.atan2(segment_end[1] - segment_start[1], segment_end[0] - segment_start[0])
    return x, y, heading


class PedestrianStop(Supervisor):
    BODY_PARTS_NUMBER = 13
    WALK_SEQUENCES_NUMBER = 8
    ROOT_HEIGHT = 1.27
    CYCLE_TO_DISTANCE_RATIO = 0.22

    JOINT_NAMES = [
        "leftArmAngle",
        "leftLowerArmAngle",
        "leftHandAngle",
        "rightArmAngle",
        "rightLowerArmAngle",
        "rightHandAngle",
        "leftLegAngle",
        "leftLowerLegAngle",
        "leftFootAngle",
        "rightLegAngle",
        "rightLowerLegAngle",
        "rightFootAngle",
        "headAngle",
    ]

    HEIGHT_OFFSETS = [
        -0.02, 0.04, 0.08, -0.03, -0.02, 0.04, 0.08, -0.03
    ]

    ANGLES = [
        [-0.52, -0.15, 0.58, 0.7, 0.52, 0.17, -0.36, -0.74],
        [0.0, -0.16, -0.7, -0.38, -0.47, -0.3, -0.58, -0.21],
        [0.12, 0.0, 0.12, 0.2, 0.0, -0.17, -0.25, 0.0],
        [0.52, 0.17, -0.36, -0.74, -0.52, -0.15, 0.58, 0.7],
        [0.0, -0.16, -0.7, -0.38, -0.47, -0.3, -0.58, -0.21],
        [0.0, -0.17, -0.25, 0.0, 0.12, 0.0, 0.12, 0.2],
        [-0.55, -0.85, -1.14, -0.7, -0.56, 0.12, 0.24, 0.4],
        [1.4, 1.58, 1.71, 0.49, 0.84, 0.0, 0.14, 0.26],
        [0.07, 0.07, -0.07, -0.36, 0.0, 0.0, 0.32, -0.07],
        [-0.56, 0.12, 0.24, 0.4, -0.55, -0.85, -1.14, -0.7],
        [0.84, 0.0, 0.14, 0.26, 1.4, 1.58, 1.71, 0.49],
        [0.0, 0.0, 0.42, -0.07, 0.07, 0.07, -0.07, -0.36],
        [0.18, 0.09, 0.0, 0.09, 0.18, 0.09, 0.0, 0.09],
    ]

    @staticmethod
    def _parse_trajectory(trajectory_option):
        points = []
        for token in trajectory_option.split(','):
            coords = token.split()
            if len(coords) != 2:
                continue
            points.append((float(coords[0]), float(coords[1])))
        return points

    @staticmethod
    def _build_cumulative_distances(points):
        cumulative = [0.0]
        for i in range(len(points) - 1):
            dx = points[i + 1][0] - points[i][0]
            dy = points[i + 1][1] - points[i][1]
            cumulative.append(cumulative[-1] + math.hypot(dx, dy))
        return cumulative

    @staticmethod
    def _interpolate_position_and_heading(points, cumulative, traveled):
        if len(points) < 2:
            return points[0][0], points[0][1], 0.0

        total_length = cumulative[-1]
        if traveled <= 0.0:
            heading = math.atan2(points[1][1] - points[0][1], points[1][0] - points[0][0])
            return points[0][0], points[0][1], heading

        if traveled >= total_length:
            heading = math.atan2(points[-1][1] - points[-2][1], points[-1][0] - points[-2][0])
            return points[-1][0], points[-1][1], heading

        segment_index = 0
        for i in range(len(cumulative) - 1):
            if cumulative[i + 1] >= traveled:
                segment_index = i
                break

        segment_start = points[segment_index]
        segment_end = points[segment_index + 1]
        segment_length = cumulative[segment_index + 1] - cumulative[segment_index]
        if segment_length <= 0.0:
            ratio = 0.0
        else:
            ratio = (traveled - cumulative[segment_index]) / segment_length

        x = segment_start[0] + ratio * (segment_end[0] - segment_start[0])
        y = segment_start[1] + ratio * (segment_end[1] - segment_start[1])
        heading = math.atan2(segment_end[1] - segment_start[1], segment_end[0] - segment_start[0])
        return x, y, heading

    def run(self):
        parser = optparse.OptionParser()
        parser.add_option("--trajectory", default="", help="Trajectory format: x1 y1, x2 y2, ...")
        parser.add_option("--speed", type=float, default=0.5, help="Walking speed in m/s")
        parser.add_option("--max-distance", type=float, default=3.0, help="Maximum traveled distance in meters")
        parser.add_option("--start-on-takeoff", action="store_true", default=False,
                          help="Wait for Crazyflie to reach takeoff height before starting to walk")
        parser.add_option("--takeoff-height", type=float, default=1.0,
                          help="Crazyflie altitude threshold (m) that triggers pedestrian walking")
        parser.add_option("--takeoff-tolerance", type=float, default=0.01,
                          help="Altitude tolerance (m) when checking the takeoff threshold")
        parser.add_option("--crazyflie-def", default="CRAZYFLIE",
                          help="DEF name of the Crazyflie node in the world")
        parser.add_option("--step", type=int, help="Controller time step (optional)")
        options, _ = parser.parse_args()

        points = parse_trajectory(options.trajectory)
        if len(points) < 2:
            print("[pedestrian_stop] Invalid trajectory. Expected at least 2 points.")
            return

        speed = options.speed if options.speed and options.speed > 0.0 else 0.5
        max_distance = options.max_distance if options.max_distance and options.max_distance > 0.0 else 0.0
        time_step = options.step if options.step and options.step > 0 else int(self.getBasicTimeStep())

        root_node = self.getSelf()
        translation_field = root_node.getField("translation")
        rotation_field = root_node.getField("rotation")
        joint_fields = [root_node.getField(name) for name in self.JOINT_NAMES]
        root_height = self.ROOT_HEIGHT

        cumulative = self._build_cumulative_distances(points)
        path_length = cumulative[-1]
        if path_length <= 0.0:
            print("[pedestrian_stop] Degenerate trajectory length.")
            return

        capped_distance = min(max_distance, path_length)

        start_on_takeoff = options.start_on_takeoff
        takeoff_height = options.takeoff_height
        takeoff_tolerance = options.takeoff_tolerance
        crazyflie_node = None
        if start_on_takeoff:
            crazyflie_node = self.getFromDef(options.crazyflie_def)
            if crazyflie_node is None:
                print(
                    f"[pedestrian_stop] Could not find Crazyflie DEF '{options.crazyflie_def}'. "
                    "Starting walk immediately."
                )
                start_on_takeoff = False

        # Keep pedestrian still at trajectory start while waiting for takeoff trigger.
        start_x, start_y, start_heading = self._interpolate_position_and_heading(points, cumulative, 0.0)
        translation_field.setSFVec3f([start_x, start_y, root_height + self.HEIGHT_OFFSETS[0]])
        rotation_field.setSFRotation([0, 0, 1, start_heading])
        for index in range(self.BODY_PARTS_NUMBER):
            joint_fields[index].setSFFloat(self.ANGLES[index][0])

        walking_started = not start_on_takeoff
        walking_time = 0.0
        last_time = self.getTime()

        while self.step(time_step) != -1:
            current_time = self.getTime()
            dt = max(0.0, current_time - last_time)
            last_time = current_time

            if not walking_started:
                altitude = crazyflie_node.getPosition()[2]
                if altitude >= (takeoff_height - takeoff_tolerance):
                    walking_started = True
                    print(
                        f"[pedestrian_stop] Crazyflie reached takeoff altitude ({altitude:.3f} m). "
                        "Starting pedestrian movement."
                    )
                else:
                    continue

            walking_time += dt
            traveled = min(walking_time * speed, capped_distance)
            x, y, heading = self._interpolate_position_and_heading(points, cumulative, traveled)

            current_sequence = int(((walking_time * speed) / self.CYCLE_TO_DISTANCE_RATIO) % self.WALK_SEQUENCES_NUMBER)
            ratio = (walking_time * speed) / self.CYCLE_TO_DISTANCE_RATIO
            ratio -= int(ratio)

            for index in range(self.BODY_PARTS_NUMBER):
                current_angle = self.ANGLES[index][current_sequence] * (1 - ratio) + \
                    self.ANGLES[index][(current_sequence + 1) % self.WALK_SEQUENCES_NUMBER] * ratio
                joint_fields[index].setSFFloat(current_angle)

            current_height_offset = self.HEIGHT_OFFSETS[current_sequence] * (1 - ratio) + \
                self.HEIGHT_OFFSETS[(current_sequence + 1) % self.WALK_SEQUENCES_NUMBER] * ratio

            translation_field.setSFVec3f([x, y, root_height + current_height_offset])
            rotation_field.setSFRotation([0, 0, 1, heading])

            if traveled >= capped_distance:
                break


controller = PedestrianStop()
controller.run()
