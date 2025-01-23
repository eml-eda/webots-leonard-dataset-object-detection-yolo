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
    corners = [
        [d_h, d_v],  # top-right
        [-d_h, d_v],  # top-left
        [-d_h, -d_v],  # bottom-left
        [d_h, -d_v]  # bottom-right
    ]

    return corners

if __name__ == "__main__":
    # Example parameters
    camera_height = 1.30  # in meters
    horizontal_fov = 69  # in degrees
    vertical_fov = 42    # in degrees

    # Calculate corners
    corners = calculate_viewable_area(camera_height, horizontal_fov, vertical_fov)

    # Output results
    print("Viewable Area Corners (in meters):")
    for corner, coordinates in corners:
        print(f"{corner}: {coordinates}")
