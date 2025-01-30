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
    colors = ["r", "r", "r"]
    # image_height, image_width, _ = image.shape

    for poly, color in zip(polygons, colors):
        # Connect back to first point to close polygon
        points = np.vstack((poly, poly[0]))
        points[:, 0] = points[:, 0] * image.shape[1]
        points[:, 1] = points[:, 1] * image.shape[0]
        print("Points: ", points)
        plt.plot(points[:, 0], points[:, 1], color)

    plt.imshow(image)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Polygon Overlap Visualization")
    plt.grid(True)
    plt.axis("equal")
    plt.show()

def plot_polygon_over_image(polygon, image):
    """
    Plot a single polygon using matplotlib
    Args:
        polygon: List of numpy arrays containing polygon vertices
        image: Image to plot the polygon over
    """
    plt.figure(figsize=(10, 10))
    colors = ["r"]
    # image_height, image_width, _ = image.shape

    # Connect back to first point to close polygon
    points = np.vstack((polygon, polygon[0]))
    points[:, 0] = points[:, 0] * image.shape[1]
    points[:, 1] = points[:, 1] * image.shape[0]
    plt.plot(points[:, 0], points[:, 1], colors[0])

    plt.imshow(image)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Polygon Overlap Visualization")
    plt.grid(True)
    plt.axis("equal")
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
            partial_return_points.append(
                [float(x_relative), float(y_relative), float(corner[2])]
            )
            # breakpoint()
            # print("Corner: ", corner)
            # print("Partial return points: ", partial_return_points)
        # print("Corners: ", corners)
        # print("Partial return points: ", partial_return_points)
        return_points.append(partial_return_points)
    # print("Return points: ", return_points)
    return return_points

def compute_points_relative_to_image_single(image_corners, points_corners):
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
    for corner in points_corners:
        x_point, y_point, _ = corner
        x_difference = x_point - top_left[0]
        y_difference = y_point - top_left[1]
        x_relative = abs(x_difference / x_image_difference)
        y_relative = abs(y_difference / y_image_difference)
        return_points.append([float(x_relative), float(y_relative), float(corner[2])])
    return return_points


def main():
    # Load the .npy files
    kuka_box_polygon = np.load("kuka_box/corners.npy")
    biscuit_box_polygon = np.load("biscuit_box/corners.npy")
    wooden_box_polygon = np.load("wooden_box/corners.npy")

    kuka_box_polygon_len = len(kuka_box_polygon)
    biscuit_box_polygon_len = len(biscuit_box_polygon)
    wooden_box_polygon_len = len(wooden_box_polygon)

    # Check that the len of the three polygons is the same
    try:
        assert kuka_box_polygon_len == biscuit_box_polygon_len == wooden_box_polygon_len
    except AssertionError:
        print("The number of polygons is not the same")
        return

    for i in range(kuka_box_polygon_len):
        polygons = [
            kuka_box_polygon[i],
            biscuit_box_polygon[i],
            wooden_box_polygon[i],
        ]
        shapely_polygons = [Polygon(p) for p in polygons]
        if not check_overlaps(shapely_polygons):
            for j in range(3):
                print(f"Frame {i} has no overlaps")
                image = plt.imread(f"image_capture/all_images/image_{i}.png")
                bottom_face_viewable_area = calculate_viewable_area(1.30, 69, 42)
                # plot_polygons_over_image(polygons, image)
                # polygons_relative_to_image = compute_points_relative_to_image(
                #     viewable_area, polygons[j]
                # )
                bottom_polygons_relative_to_image = compute_points_relative_to_image_single(
                    bottom_face_viewable_area, polygons[j]
                )
                # print("polygons_relative_to_image: ", polygons_relative_to_image)
                # plot_polygons_over_image(polygons_relative_to_image, image)
                plot_polygon_over_image(bottom_polygons_relative_to_image, image)

                if j == 1:
                    top_face_viewable_area = calculate_viewable_area(1.26, 69, 42)
                else:
                    top_face_viewable_area = calculate_viewable_area(1.24, 69, 42)
                    
                top_polygons_relative_to_image = compute_points_relative_to_image_single(
                    top_face_viewable_area, polygons[j]
                )
                plot_polygon_over_image(top_polygons_relative_to_image, image)


if __name__ == "__main__":
    main()
