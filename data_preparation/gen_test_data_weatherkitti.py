#!/usr/bin/env python3
# Developed by Wenqing Kuang and Xiongwei Zhao
# This file is covered by the LICENSE file in the root of the project ResLPR:
# https://github.com/nubot-nudt/ResLPR
# Brief: a script to generate depth and intensity data
import os
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.range_projection import range_projection
from multiprocessing import Pool
import glob
import random
import argparse

def get_file_paths(folder):
    paths = glob.glob(os.path.join(folder, '*.bin'))
    paths.sort()
    return paths

def extract_file_name(file_path):
    """
    Extracts the file name from a given file path and converts it to an integer if possible.
    """
    base_name = os.path.basename(file_path)
    name, _ = os.path.splitext(base_name)
    return int(name)

def sample_files_from_folders(folders):

    all_files = []
    # Step 1: Load files from each folder and check file counts
    folder_files = [get_file_paths(folder) for folder in folders]
    file_counts = [len(files) for files in folder_files]

    if len(set(file_counts)) != 1:
        raise ValueError("All folders must have the same number of files.")

    print(f"There are files in each folder: {file_counts}")

    num_files = file_counts[0]

    # Step 2: Extract unique file names from the first folder
    all_files_set = {os.path.basename(file) for file in folder_files[0]}

    # Step 3: Randomly shuffle the file names and distribute them across folders
    file_names = list(all_files_set)
    random.seed(42)  # Set random seed
    random.shuffle(file_names)

    sampled_files = {folder: [] for folder in folders}

    for i, file_name in enumerate(file_names):
        folder_index = i % 3  # Distribute files evenly among the folders
        folder = folders[folder_index]
        file_path = os.path.join(folder, file_name)
        sampled_files[folder].append(file_path)

    # Verify that each file was selected only once and that the total number matches
    selected_file_names = set()
    for files in sampled_files.values():
        for file in files:
            file_name = os.path.basename(file)
            if file_name in selected_file_names:
                raise ValueError(f"Duplicate file selected: {file_name}")
            selected_file_names.add(file_name)

    if len(selected_file_names) != num_files:
        raise ValueError("The total number of sampled files does not match the number in one folder.")

    # Step 4: Collect all sampled files and sort them by the integer value of the file name
    for folder, files in sampled_files.items():
        all_files.extend(files)

    all_files.sort(key=extract_file_name)

    print(f"Total files to sample: {len(all_files)}")
    return all_files

def normalize_data(data, data_range, target_range=(0, 255)):
    # Obtain a mask to mark the positions of abnormal values
    mask = data == -1
    # Temporarily set anomaly values to the minimum value of the data range for normalization processing.
    data[mask] = data_range[0]

    # Normalization
    min_val, max_val = data_range
    target_min, target_max = target_range
    normalized_data = (data - min_val) / (max_val - min_val) * (target_max - target_min) + target_min

    normalized_data = np.clip(normalized_data, target_min, target_max)

    # Restore anomaly values
    normalized_data[mask] = -1

    normalized_data = normalized_data.astype(np.int16)
    normalized_data[mask] = -1

    return normalized_data

def process_scan(scan_path, dst_folder, clean_flag, severity):
    """
    Process a single point cloud file to generate a depth-intensity map and save it.
    :param scan_path: point cloud file path
    :param dst_folder: depth and intensity image save path
    :param clean_flag: he original point cloud of KITTI or the corrupted point cloud
    :param severity: the severity of the corruption, only used when clean_flag is False.
    """
    # Load point cloud data
    current_vertex = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)

    # Generate depth and intensity data
    # The size of the training data used in WeatherKITTI is 64*1920*2.
    proj_range, _, proj_intensity, _ = range_projection(current_vertex, fov_up=3.0, fov_down=-25.0, proj_H=64, proj_W=1920, max_range=80)

    # Normalize proj_range data
    proj_range_normalized = normalize_data(proj_range, data_range=(0, 80))

    # Normalize proj_intensity data
    proj_intensity_normalized = normalize_data(proj_intensity, data_range=(0, 1))

    # expand dimensions
    proj_range_normalized = np.expand_dims(proj_range_normalized, axis=-1)
    proj_intensity_normalized = np.expand_dims(proj_intensity_normalized, axis=-1)

    range_int_image = np.concatenate((proj_range_normalized, proj_intensity_normalized), axis=-1)

    # Generate the save path and save the image
    idx = os.path.basename(scan_path).split('.')[0]
    if clean_flag:
        path_range_and_intensity = os.path.join(dst_folder, 'clean')
        os.makedirs(path_range_and_intensity, exist_ok=True)
        file_name_range_and_intensity = os.path.join(path_range_and_intensity, f"{str(idx).zfill(6)}.npy")
    else:
        severity_dict = {0: 'light', 1: 'mod', 2: 'heavy'}
        path_range_and_intensity = os.path.join(dst_folder, severity_dict[severity])
        os.makedirs(path_range_and_intensity, exist_ok=True)
        file_name_range_and_intensity = os.path.join(path_range_and_intensity, f"{str(idx).zfill(6)}.npy")

    np.save(file_name_range_and_intensity, range_int_image)

def gen_depth_intensity_data(scan_folder, dst_folder, clean_flag, corruption_type):
    """ Generate depth and intensity data for the given folders. """
    for i in range(0, 1):

        if clean_flag:
            scan_path = os.path.join(scan_folder, str(i).zfill(2), 'velodyne')
            folders = [scan_path]
        else:
            scan_path_light = os.path.join(scan_folder, 'light', str(i).zfill(2), f'{corruption_type}_velodyne')
            scan_path_mod = os.path.join(scan_folder, 'mod', str(i).zfill(2), f'{corruption_type}_velodyne')
            scan_path_heavy = os.path.join(scan_folder, 'heavy', str(i).zfill(2), f'{corruption_type}_velodyne')
            folders = [scan_path_light, scan_path_mod, scan_path_heavy]

        for i, folder in enumerate(folders):
            file_paths = get_file_paths(folder)
            print("###############Processing path############### ", folder)
            print("len of sampled files: ", len(file_paths))
            # Process point cloud files using a process pool
            with Pool() as pool:
                pool.starmap(process_scan, [(path, dst_folder, clean_flag, i) for path in file_paths])

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate depth and intensity data from point cloud files.')

    parser.add_argument('--scan_folder', '--sf', type=str, required=True,
                        help='The path to weatherKITTI snow/fog/rain train data or clean kitti data')

    parser.add_argument('--dst_folder', '--df', type=str, required=True,
                        help='The path to save depth and intensity image data')

    parser.add_argument('--clean_flag', '--cf', action='store_false', default=True,
                        help='Choose whether to generate clean data or corrupted data. Default is True.')

    parser.add_argument('--corruption_type', '--ct', type=str, required=False,
                        help='Select a corresponding type of corruption.')

    args = parser.parse_args()

    scan_folder = args.scan_folder
    dst_folder = args.dst_folder
    clean_flag = args.clean_flag
    corruption_type = args.corruption_type
    os.makedirs(dst_folder, exist_ok=True)

    gen_depth_intensity_data(scan_folder, dst_folder, clean_flag, corruption_type)