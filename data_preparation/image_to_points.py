import numpy as np
import struct
import os
from multiprocessing import Pool
from tqdm import tqdm
import time
import argparse

def set_dataset_parameters(dataset_type):
    if dataset_type == 'WeatherKITTI':
        fov_up = 3.0
        fov_down = -25.0
        proj_H = 64
        proj_W = 1920
        depth_range = 80.0
        intensity_range = 1.0
    elif dataset_type == 'WeatherNCLT':
        fov_up = 30.67
        fov_down = -10.67
        proj_H = 32
        proj_W = 1440
        depth_range = 60.0
        intensity_range = 1.0
    else:
        raise ValueError("Invalid dataset type. Allowed types are 'WeatherKITTI' and 'WeatherNCLT'.")

    return fov_up, fov_down, proj_H, proj_W, depth_range, intensity_range

def save_point_cloud(out_path, bin_file_name, point_cloud):

    file_name = os.path.basename(bin_file_name)
    number_str = file_name.split('.')[0]
    padded_number = number_str.zfill(6)  # Fill it to six digits.
    new_file_name = f"{padded_number}.bin"  # Generate a new file name.
    output_filename = os.path.join(out_path, new_file_name)
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    """Save point cloud data to a .bin file."""
    point_cloud = point_cloud.astype(np.float32)
    # Use Numpy's tofile method to directly write to a binary file.
    point_cloud.tofile(output_filename)

def normalize_custom_data(data, depth_range, intensity_range):
    # Copy the original data to prevent the modification of the original array.
    result_data = data.copy()
    new_normalized_data = np.zeros(data.shape)

    # Extract two channels, with depth in the first channel and intensity in the second channel.
    depth = result_data[:, :, 0]
    intensity = result_data[:, :, 1]

    # The processing of the depth channel
    depth_mask = (depth == -1)  # Get the positions of the outliers.
    # Normalize it to the range of the depth and exclude the outliers.
    depth_min, depth_max = 0, 255
    depth_normalized = (depth - depth_min) / (depth_max - depth_min) * depth_range
    depth_normalized = np.clip(depth_normalized, 0, depth_range)
    # Restore the outliers
    depth_normalized[depth_mask] = -1

    # The processing of the intensity channel
    intensity_mask = (intensity == -1)
    # Normalize it to the range of the intensity and exclude the outliers.
    intensity_min, intensity_max = 0, 255
    intensity_normalized = (intensity - intensity_min) / (intensity_max - intensity_min) * intensity_range
    intensity_normalized = np.clip(intensity_normalized, 0, intensity_range)
    # Restore the outliers
    intensity_normalized[intensity_mask] = -1

    # Save the normalized result back to the result array.
    new_normalized_data[:, :, 0] = depth_normalized
    new_normalized_data[:, :, 1] = intensity_normalized
    return new_normalized_data

def restore_point_cloud(proj_range_and_intensity, dataset_type):
    """ Convert to a point cloud from depth and intensity image.
        Args:
            proj_range_and_intensity: projected depth and intensity image
            dataset: the type of the dataset
                fov_up: upper bound of vertical fov
                fov_down: lower bound of vertical fov
                proj_H: the length parameter of the image
                proj_W: the width parameter of the image
                depth_range: the maximum depth value
                intensity_range: the maximum intensity value
        Returns:
            converted_point_cloud: the point cloud that has been completed with the conversion
    """
    fov_up, fov_down, proj_H, proj_W, depth_range, intensity_range = set_dataset_parameters(dataset_type)

    fov_up = fov_up / 180.0 * np.pi  # field of view up in radians
    fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians

    proj_depth_and_intensity_norm = normalize_custom_data(proj_range_and_intensity, depth_range, intensity_range)

    depth = proj_depth_and_intensity_norm[:, :, 0]
    intensity = proj_depth_and_intensity_norm[:, :, 1]

    # Ignore invalid points
    valid_mask = (depth > 0) & (intensity > 0)

    # Compute pitch and yaw
    y_indices, x_indices = np.indices((proj_H, proj_W))
    pitch = (1.0 - y_indices / proj_H) * fov - abs(fov_down)
    yaw = -(x_indices / proj_W - 0.5) * 2 * np.pi

    # Calculate the coordinates
    x_coord = depth * np.cos(pitch) * np.cos(yaw)
    y_coord = depth * np.cos(pitch) * np.sin(yaw)
    z_coord = depth * np.sin(pitch)

    # Apply valid mask
    x_coord = x_coord[valid_mask]
    y_coord = y_coord[valid_mask]
    z_coord = z_coord[valid_mask]
    intensity = intensity[valid_mask]

    converted_point_cloud = np.stack((x_coord, y_coord, z_coord, intensity), axis=-1)

    return converted_point_cloud

def process_file(args):
    file_path_npy, output_bin_path_selected, dataset_type = args
    print("Processing file:", file_path_npy)
    proj_range_and_intensity = np.load(file_path_npy)
    # Restore the point cloud
    restored_point_cloud = restore_point_cloud(proj_range_and_intensity, dataset_type=dataset_type)
    # Save the restored point cloud to a .bin file
    save_point_cloud(output_bin_path_selected, file_path_npy, restored_point_cloud)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate depth and intensity data from point cloud files.')

    parser.add_argument('--npy_folder', '--nf', type=str, required=True,
                        help='The path to the restored npy file for weatherKITTI.')

    parser.add_argument('--dst_folder', '--df', type=str, required=True,
                        help='The path to save the converted point cloud file.')

    parser.add_argument('--dataset_type', '--dt', type=str, required=True,
                        help='The type of the dataset being processed, WeatherKITTI or WeatherNCLT.')

    args = parser.parse_args()

    npy_folder = args.npy_folder
    dst_folder = args.dst_folder
    dataset_type = args.dataset_type
    os.makedirs(dst_folder, exist_ok=True)

    target_path_list = []
    # Traverse all the contents in the npy_folder.
    for item in os.listdir(npy_folder):
        item_path = os.path.join(npy_folder, item)
        # Check whether it is a folder.
        if os.path.isdir(item_path):
            # Add the sub folder names to the list.
            target_path_list.append(item)

    target_path_list.sort()

    for target_path in target_path_list:
        input_npy_path_selected = os.path.join(npy_folder, target_path)
        output_bin_path_selected = os.path.join(dst_folder, target_path)

        # Load the range image and intensity image from files
        # Get the absolute paths of all .npy files in the folder.
        npy_files = [os.path.join(input_npy_path_selected, f) for f in os.listdir(input_npy_path_selected) if
                     f.endswith('.npy')]

        sorted_npy_files = sorted(npy_files)

        # Create a list of parameters for use in the process pool.
        args_list = [(file_path_npy, output_bin_path_selected, dataset_type) for file_path_npy in sorted_npy_files]

        # Perform parallel processing using a process pool and display the progress.
        with Pool() as pool:
            list(tqdm(pool.imap(process_file, args_list), total=len(args_list)))