import numpy as np
from shapely.geometry import Polygon
from itertools import combinations
import matplotlib.pyplot as plt
import shutil
import os
from check_overlap import check_overlaps
from camera_viewable_area import calculate_viewable_area

def plot_polygons_over_image(polygons, image):
    """
    Plot multiple polygons using matplotlib
    Args:
        polygons: List of numpy arrays containing polygon vertices
        titles: List of strings for legend
    """
    plt.figure(figsize=(10, 10))
    colors = ['r', 'r', 'r']
    # image_height, image_width, _ = image.shape

    for poly, color in zip(polygons, colors):
        # Connect back to first point to close polygon
        points = np.vstack((poly, poly[0]))
        points[:, 0] = points[:, 0] * image.shape[1]
        points[:, 1] = points[:, 1] * image.shape[0]
        print("Points: ", points)
        plt.plot(points[:, 0], points[:, 1], color)
    
    plt.imshow(image)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Polygon Overlap Visualization')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def compute_points_relative_to_image(image_corners, points_corners):
    """
    Compute the points relative to the image corners
    Args:
        image_corners: List of numpy arrays containing image corner vertices
        point_corners: List of numpy arrays containing point corner vertices
    Returns:
        List of numpy arrays containing the points relative to the image corners
    """
    # Compute the transformation matrix
    top_right = image_corners[0]
    top_left = image_corners[1]
    bottom_left = image_corners[2]
    bottom_right = image_corners[3]

    x_image_difference = top_right[0] - top_left[0]
    y_image_difference = top_right[1] - bottom_right[1]

    return_points = []
    for corners in points_corners:
        # print("Corners: ", corners)
        partial_return_points = []
        for corner in corners:
            # print("Corner: ", corner)
            x_point, y_point, _ = corner
            x_difference = x_point - top_left[0]
            y_difference = y_point - top_left[1]
            x_relative = abs(x_difference / x_image_difference)
            y_relative = abs(y_difference / y_image_difference)
            partial_return_points.append([float(x_relative), float(y_relative), float(corner[2])])
            # breakpoint()
            # print("Corner: ", corner)
            # print("Partial return points: ", partial_return_points)
        # print("Corners: ", corners)
        # print("Partial return points: ", partial_return_points)
        return_points.append(partial_return_points)
    # print("Return points: ", return_points)
    return return_points

def main():
    # Load the .npy files
    poly1 = np.load('kuka_box/corners.npy')
    poly2 = np.load('biscuit_box/corners.npy')
    poly3 = np.load('wooden_box/corners.npy')
    
    poly_len1 = len(poly1)
    poly_len2 = len(poly2)
    poly_len3 = len(poly3)

    # Check that the len of the three polygons is the same
    try:
        assert poly_len1 == poly_len2 == poly_len3
    except AssertionError:
        print("The number of polygons is not the same")
        return
    
    for i in range(poly_len1):
        polygons = [poly1[i], poly2[i], poly3[i]]
        shapely_polygons = [Polygon(p) for p in polygons]
        if not check_overlaps(shapely_polygons):
            print(f"Frame {i} has no overlaps")
            image = plt.imread(f'image_capture/all_images/image_{i}.png')
            viewable_area = calculate_viewable_area(1.30, 69, 42)
            # plot_polygons_over_image(polygons, image)
            polygons_relative_to_image = compute_points_relative_to_image(viewable_area, polygons)
            # print("polygons_relative_to_image: ", polygons_relative_to_image)
            plot_polygons_over_image(polygons_relative_to_image, image)

if __name__ == "__main__":
    main()