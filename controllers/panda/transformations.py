import math


def calculate_viewable_area(height, hfov, vfov):
    """
    Calculate the corners of the viewable area on the ground.

    Parameters:
        height (float): Height of the camera in meters.
        hfov (float): Horizontal field of view in degrees.
        vfov (float): Vertical field of view in degrees.

    Returns:
        dict: Coordinates of the four corners on the ground.
    """
    # Convert FOV angles to radians
    hfov_rad = math.radians(hfov / 2)
    vfov_rad = math.radians(vfov / 2)

    # Calculate distances to the edges of the viewable area
    d_h = height * math.tan(hfov_rad)
    d_v = height * math.tan(vfov_rad)

    # Calculate the corners
    corners = [[d_h, d_v], [-d_h, d_v], [-d_h, -d_v], [d_h, -d_v]]

    # Invert x and y coordinates
    corners = [[y, x] for x, y in corners]

    return corners


def compute_points_world_from_relative(image_corners, relative_points):
    """
    Convert relative coordinates back to world coordinates using image corners
    Args:
        image_corners: List of numpy arrays containing image corner vertices
        relative_points: List of arrays containing relative coordinates [x_rel, y_rel, z]
    Returns:
        List of arrays containing world coordinates [x, y, z]
    """
    # Get image corner references
    top_right = image_corners[1]
    top_left = image_corners[0]
    bottom_left = image_corners[3]
    bottom_right = image_corners[2]

    # Get image dimensions in world coordinates
    y_image_difference = abs(top_right[1] - top_left[1])
    # print("y_image_difference", y_image_difference)
    x_image_difference = abs(top_right[0] - bottom_right[0])
    # print("x_image_difference", x_image_difference)

    world_points = []
    for point in relative_points:
        y_rel, x_rel, z = point
        x_world = (x_rel * x_image_difference) - top_left[0]
        y_world = top_left[1] - (y_rel * y_image_difference)
        world_points.append([float(x_world), float(y_world), float(z)])

    return world_points


def parse_bbox(bbox_unprocessed, obj_id):
    # TODO: generalize this
    # Extract object infos
    x_center, y_center = bbox_unprocessed[obj_id][2:4]
    size = bbox_unprocessed[obj_id][
        4:6
    ]  # FIXME: need to tranform just as the center position

    # Transform the coordinates
    y_center = y_center - 140
    x_center = x_center / 640
    y_center = y_center / 360
    viewable_area = calculate_viewable_area(1.3, 69, 42)
    points_world = compute_points_world_from_relative(
        viewable_area, [[x_center, y_center, 0.76]]
    )[0]

    # Quickfix: Invert x-axis
    position = [-points_world[0], points_world[1], points_world[2]]
    return position, size
