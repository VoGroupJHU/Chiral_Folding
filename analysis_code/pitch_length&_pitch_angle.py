import numpy as np
import math
import gsd.hoomd
import os
from scipy.optimize import least_squares

# Step 1: Calculate the principal axis of the pentagon to determine its orientation.
# Step 2: Rotate the pentagon core such that its principal axis aligns with the z-axis for easier analysis.
# Step 3: Use a parametric function to fit the helix structure.
# Step 4: Calculate the pitch angle and pitch length based on the parametric function.

# folder to perform the calculation
size = 1.1
path = "/data/tvo12/xzhan357/project_9_single_chain_chiral/hollow_tube/batch_run/BP_size=" + str(size) + "/"
chain_length = 200
# if size == 0.8:
    # set_folders = [0.1, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 20.0, 24.0, 28.0, 32.0, 40.0, 48.0] # size = 0.8
# if size == 0.9:
    # set_folders = [0.1, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 30.0, 34.0, 38.0, 42.0, 44.0, 48.0]
# if size == 1.0:
    # set_folders = [0.1, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 25.0, 30.0, 34.0, 38.0, 42.0, 44.0, 48.0]   # size = 1.0    
# if size == 1.1:
    # set_folders = [0.1, 4.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 36.0, 42.0, 48.0, 54.0, 60.0]

# pentagon type
penta_type = 4

# read all folders in the directory 
folders_dis = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
set_folders = sorted(folders_dis, key=lambda folder_name: float(folder_name.split('=')[-1]))
print(set_folders)

# output file name
name_backup = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen"]
output_file = open(path + "BP_" + str(size) + "_pitch.txt", 'w')
output_file.close()
def calculate_principal_axis(points):
    """
    Calculate the first principal axis of a set of points in 3D.
    
    Parameters:
        points (numpy.ndarray): An Nx3 array of 3D points.
        
    Returns:
        numpy.ndarray: The first principal axis (a 3D unit vector).
    """
    # Compute the centroid and center the points
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    
    # Compute the covariance matrix
    covariance_matrix = np.cov(centered_points, rowvar=False)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # Extract the eigenvector corresponding to the largest eigenvalue
    first_principal_axis = eigenvectors[:, -1]
    
    return first_principal_axis, centroid

def calculate_quaternion_to_align(v1, v2):
    """
    Calculate the quaternion to rotate vector v1 to align with vector v2.
    
    Parameters:
        v1 (numpy.ndarray): A 3D vector to rotate from.
        v2 (numpy.ndarray): A 3D vector to rotate to.
        
    Returns:
        numpy.ndarray: Quaternion [w, x, y, z] representing the rotation.
    """
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    cross = np.cross(v1, v2)
    dot = np.dot(v1, v2)
    if dot >= 1.0:
        # No rotation needed
        return np.array([1, 0, 0, 0])
    elif dot <= -1.0:
        # 180-degree rotation about any orthogonal axis
        orthogonal = np.array([1, 0, 0]) if np.abs(v1[0]) < 0.99 else np.array([0, 1, 0])
        axis = np.cross(v1, orthogonal)
        axis = axis / np.linalg.norm(axis)
        return np.array([0, *axis])
    else:
        w = np.sqrt((1.0 + dot) * 0.5)
        xyz = cross / np.linalg.norm(cross) * np.sqrt((1.0 - dot) * 0.5)
        return np.array([w, xyz[0], xyz[1], xyz[2]])

def apply_quaternion_rotation(points, quaternion, centroid):
    """
    Rotate a set of points using a quaternion.
    
    Parameters:
        points (numpy.ndarray): An Nx3 array of 3D points.
        quaternion (numpy.ndarray): Quaternion [w, x, y, z].
        centroid (numpy.ndarray): Centroid of the points to shift them back after rotation.
        
    Returns:
        numpy.ndarray: The rotated Nx3 array of points.
    """
    w, x, y, z = quaternion
    rotation_matrix = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]
    ])  # Convert quaternion to rotation matrix
    
    # Subtract the centroid, rotate, and then add the centroid back
    centered_points = points - centroid
    rotated_points = centered_points @ rotation_matrix.T
    return rotated_points   # align with (0,0,1)

def calculate_helix_turns(points):
    """
    Calculate the number of turns and the period of a helical structure.

    Parameters:
    - points: ndarray of shape (N, 3), where N is the number of 3D points (x, y, z).

    Returns:
    - num_turns: Number of helix turns.
    - period: The distance along the z-axis per turn.
    """
    # Extract x, y, z
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Calculate the angular position of each point in the x-y plane
    angles = np.arctan2(y, x)  # Angle in radians (-π, π]

    # Unwrap the angles to avoid discontinuities (e.g., from π to -π)
    unwrapped_angles = np.unwrap(angles)

    # Calculate the total change in angle (in radians)
    total_angle_change = unwrapped_angles[-1] - unwrapped_angles[0]

    # Number of turns is the total angle change divided by 2π
    num_turns = total_angle_change / (2 * np.pi)

    # Calculate the z-distance covered by the helix
    z_min, z_max = np.min(z), np.max(z)
    total_z_distance = z_max - z_min

    # Calculate the period (z-distance per turn)
    period = total_z_distance / num_turns

    return num_turns, period

def project_to_xy_and_calculate_mean(points_3d):
    """
    Project points in 3D onto the x-y plane and calculate the mean center.

    Parameters:
    points_3d (numpy.ndarray): A Nx3 array where each row represents a point in 3D space.

    Returns:
    tuple: The mean center of the projected points on the x-y plane as (mean_x, mean_y).
    """
    # Project the points onto the x-y plane by ignoring the z-coordinate
    points_xy = points_3d[:, :2]
    
    # Calculate the mean center of the projected points
    mean_x = np.mean(points_xy[:, 0])
    mean_y = np.mean(points_xy[:, 1])
    
    return mean_x, mean_y
# perform the calculation for all of folders in the directory
for folder in set_folders:
    # read bond length value
    bond_length = folder.split('=')[-1]
    bond_length = f"{float(bond_length):.1f}"
    folder = folder + "/"
    start_name = "_chiral_a_" + str(bond_length)
    filename = start_name
    for name in name_backup:
        test_name = filename + "_" + name
        if os.path.isfile(path + folder + test_name + ".gsd"):
            filename = test_name
        else:
            break
    file_temp = path + folder + filename + '.gsd'
    print("a", file_temp)
    # open the gsd file
    f = gsd.hoomd.open(name = file_temp, mode = 'rb')
    frame_temp = f[len(f)-1]
    f.close() 
    # read number of particles and typeid array in the system
    N = frame_temp.particles.N
    typeid = frame_temp.particles.typeid
    posi = []
    posi_penta = []
    count = 0
    count_5 = 0
    threshhold = 0
    
    # use above functions to perform the calculation
    for i in range(N):
        if (frame_temp.particles.typeid[i] == 0):
            count = count + 1
            if (count > threshhold):       # <500 all of particles.
                posi.append(frame_temp.particles.position[i])
        if (frame_temp.particles.typeid[i] >= penta_type):
            count_5 = count_5 + 1
            if (count_5 > threshhold):
                posi_penta.append(frame_temp.particles.position[i])
    posi = np.array(posi) 
    posi_penta = np.array(posi_penta)
    first_principal_axis, centroid = calculate_principal_axis(posi_penta)
    target_axis = np.array([0, 0, 1])
    quaternion = calculate_quaternion_to_align(first_principal_axis, target_axis)
    rotated_points = apply_quaternion_rotation(posi, quaternion, centroid)
    N_T = len(rotated_points)
    # pitch length
    num_turns, period = calculate_helix_turns(rotated_points)
    num_turns = np.abs(num_turns)
    period = np.abs(period)
    # center position on the x-y plane
    mean_center = project_to_xy_and_calculate_mean(rotated_points)
    # average radius of the projected points on the x-y plane
    R_average = 0.0
    for i in range(N_T):
        R_average = R_average + np.sqrt( (rotated_points[i][0] - mean_center[0])**2 + (rotated_points[i][1] - mean_center[1])**2)

    R_average = R_average/N_T
    # pitch angle
    pitch_angle = np.degrees(np.arctan(period/(2.0*np.pi*R_average)))
    # output results
    output_file = open(path + "BP_" + str(size) + "_pitch.txt", 'a')
    output_file.write(f"{bond_length} {period:.6f} {pitch_angle:.6f} {R_average:.6f}  {num_turns:.6f} {mean_center[0]:.6f} {mean_center[1]:.6f}\n")
    # for i in range(len(rotated_points)):
        # output_file.write(f"{rotated_points[i][0]} {rotated_points[i][1]} {rotated_points[i][2]}\n")
    output_file.close()
    