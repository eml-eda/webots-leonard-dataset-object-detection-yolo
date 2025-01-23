import numpy as np
import matplotlib.pyplot as plt

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
    plt.title('Polygon Visualization')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def main():
    # Load single .npy file
    corners = np.load('corners.npy')
    
    # Plot each set of corners
    for i, polygon in enumerate(corners):
        plot_polygons([polygon], titles=[f'Frame {i}'])

if __name__ == "__main__":
    main()