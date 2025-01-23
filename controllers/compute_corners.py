import numpy as np

def compute_corners(position, rotation, length, width):
    x, y, z = position
    
    # Create rotation matrix
    cos_theta = np.cos(rotation)
    sin_theta = np.sin(rotation)
    
    # Define corners relative to center (counter-clockwise from top-right)
    half_length = length / 2
    half_width = width / 2
    
    corners_local = [
        [half_length, half_width],    # top-right
        [-half_length, half_width],   # top-left
        [-half_length, -half_width],  # bottom-left
        [half_length, -half_width]    # bottom-right
    ]
    
    # Apply rotation and translation to each corner
    corners_world = []
    for corner in corners_local:
        rotated_x = corner[0] * cos_theta - corner[1] * sin_theta + x
        rotated_y = corner[0] * sin_theta + corner[1] * cos_theta + y
        corners_world.append([rotated_x, rotated_y, z])
    
    return corners_world

def read_positions_file(filename):
    positions = []
    rotations = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                # Parse "[x, y, z] theta" format
                pos_str, rot_str = line.strip().split(']')
                pos = eval(pos_str + ']')  # Safe since we control the input format
                rot = float(rot_str)
                positions.append(pos)
                rotations.append(rot)
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None
    return positions, rotations

def main(input_file, output_file, length, width):
    positions, rotations = read_positions_file(input_file)
    if positions is None:
        return
        
    all_corners = []
    for pos, rot in zip(positions, rotations):
        corners = compute_corners(pos, rot, length, width)
        all_corners.append(corners)

    # Save the corners to a .npy file
    np.save(output_file, all_corners)

    print(f"Saved corners to {output_file}")
    
    return

if __name__ == "__main__":
    main("biscuit_box/gps_values.txt", "biscuit_box/corners.npy", 0.08, 0.24)
    main("kuka_box/gps_values.txt", "kuka_box/corners.npy", 0.1, 0.2)
    main("wooden_box/gps_values.txt", "wooden_box/corners.npy", 0.2, 0.2)