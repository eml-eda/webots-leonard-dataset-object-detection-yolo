import numpy as np
from shapely.geometry import Polygon
from itertools import combinations
import matplotlib.pyplot as plt
import shutil
import os

def plot_polygons(polygons, titles=None):
    """
    Plot multiple polygons using matplotlib
    Args:
        polygons: List of numpy arrays containing polygon vertices
        titles: List of strings for legend
    """
    plt.figure(figsize=(10, 10))
    colors = ['r', 'g', 'b']
    if titles is None:
        titles = [f'Polygon {i+1}' for i in range(len(polygons))]
    
    for poly, color, title in zip(polygons, colors, titles):
        # Connect back to first point to close polygon
        points = np.vstack((poly, poly[0]))
        plt.plot(points[:, 0], points[:, 1], color, label=title)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Polygon Overlap Visualization')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def check_overlaps(polygons):
    """
    Check if any of the polygons overlap with each other
    Args:
        polygons: List of numpy arrays, each containing 4 corner points
    Returns:
        bool: True if any overlap exists, False otherwise
    """
    
    # Check all combinations of polygons
    for poly1, poly2 in combinations(polygons, 2):
        if poly1.intersects(poly2):
            return True
            
    return False

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

    if os.path.exists("image_capture/overlaps"):
        shutil.rmtree("image_capture/overlaps")
    os.makedirs("image_capture/overlaps")

    if os.path.exists("image_capture/no_overlaps"):
        shutil.rmtree("image_capture/no_overlaps")
    os.makedirs("image_capture/no_overlaps")

    for i in range(poly_len1):
        polygons = [poly1[i], poly2[i], poly3[i]]
        shapely_polygons = [Polygon(p) for p in polygons]
        if check_overlaps(shapely_polygons):
            shutil.copy(f"image_capture/all_images/image_{i}.png", "image_capture/overlaps")
        else:
            shutil.copy(f"image_capture/all_images/image_{i}.png", "image_capture/no_overlaps")

        # Plot the polygons for visual verification
        # plot_polygons(polygons, titles=['Kuka Box', 'Biscuit Box', 'Wooden Box'])

if __name__ == "__main__":
    main()