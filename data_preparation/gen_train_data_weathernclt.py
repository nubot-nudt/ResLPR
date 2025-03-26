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
import struct

def load_pc_nclt(file_pathname: str):
    # Load point cloud from file
    hits = []
    with open(file_pathname,'rb') as f_bin:
        while True:
            x_str = f_bin.read(2)
            if x_str == b"":  # eof
                break
            x = struct.unpack('<H', x_str)[0]
            y = struct.unpack('<H', f_bin.read(2))[0]
            z = struct.unpack('<H', f_bin.read(2))[0]
            i = struct.unpack('B', f_bin.read(1))[0]
            l = struct.unpack('B', f_bin.read(1))[0]
            x, y, z = convert(x, y, z)
            s = "%5.3f, %5.3f, %5.3f, %d, %d" % (x, y, z, i, l)
            hits += [[x, y, -z, i/255]]  # flip z axis
            # Only retain the first four features, and normalize the intensity values to the range between 0 and 1.
        hits = np.asarray(hits)
    return hits

def convert(x_s, y_s, z_s):
    scaling = 0.005
    offset = -100.0

    x = x_s * scaling + offset
    y = y_s * scaling + offset
    z = z_s * scaling + offset

    return x, y, z


def generate_file_path_list(folder_path, txt_path):
    try:
        file_path_list = []
        with open(txt_path, 'r', encoding='utf-8') as file:
            for line in file:
                file_name = line.strip()
                if file_name:
                    full_path = os.path.join(folder_path, file_name.replace('.npy', '.bin'))
                    file_path_list.append(full_path)
        return file_path_list
    except FileNotFoundError:
        print("Error: The specified file or folder was not found!")
    except Exception as e:
        print(f"Error: An unknown error has occurred： {e}")
    return []

def get_file_paths(folder):
    return glob.glob(os.path.join(folder, '*.bin'))

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

def process_scan(scan_path, dst_folder, clean_flag):
    """
    Process a single point cloud file to generate a depth-intensity map and save it.
    :param scan_path: point cloud file path
    :param dst_folder: depth and intensity image save path
    :param clean_flag: The original point cloud of NCLT or the damaged point cloud
    """
    # Load point cloud data
    if clean_flag:
        current_vertex = load_pc_nclt(scan_path)
        current_vertex = current_vertex.reshape((-1, 4))
    else:
        current_vertex = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)

    # Generate depth and intensity data
    # The size of the training data used in WeatherNCLT is 32*1440*2.
    proj_range, _, proj_intensity, _ = range_projection(current_vertex, fov_up=30.67, fov_down=-10.67, proj_H=32, proj_W=1440, max_range=60)

    # Normalize proj_range data
    proj_range_normalized = normalize_data(proj_range, data_range=(0, 60))

    # Normalize proj_intensity data
    proj_intensity_normalized = normalize_data(proj_intensity, data_range=(0, 1))

    # expand dimensions
    proj_range_normalized = np.expand_dims(proj_range_normalized, axis=-1)
    proj_intensity_normalized = np.expand_dims(proj_intensity_normalized, axis=-1)

    range_int_image = np.concatenate((proj_range_normalized, proj_intensity_normalized), axis=-1)

    # Generate the save path and save the image
    idx = os.path.basename(scan_path).split('.')[0]
    file_name_range_and_intensity = os.path.join(dst_folder,  f"{idx}.npy")

    np.save(file_name_range_and_intensity, range_int_image)

def gen_depth_intensity_data(scan_folder, dst_folder, clean_flag, corruption_type):
    """ Generate depth and intensity data for the given folders. """

    if clean_flag:
        scan_path = os.path.join(scan_folder, 'velodyne_sync')
    else:
        scan_path = os.path.join(scan_folder, f'{corruption_type}_velodyne')

    current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory where the current script is located.
    file_list_dir = os.path.join(current_dir, '..', 'data_dir', f'nclt_{corruption_type}.txt')
    sampled_file_paths = generate_file_path_list(scan_path, file_list_dir)

    print("len of sampled files: ", len(sampled_file_paths))

    # Process point cloud files using a process pool
    with Pool() as pool:
        pool.starmap(process_scan, [(path, dst_folder, clean_flag) for path in sampled_file_paths])

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate depth and intensity data from point cloud files.')

    parser.add_argument('--scan_folder', '--sf', type=str, required=True,
                        help='The path to weatherNCLT snow/fog/rain train data or clean NCLT data')

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