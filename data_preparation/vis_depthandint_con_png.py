import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import matplotlib.pyplot as plt
from skimage import exposure
import argparse

def visualize_depth(file_path, save_dir):
    print("file path: ", file_path)
    depthandint = np.load(file_path)

    # Print the dimensional information of the .npy file.
    print(f"Loaded data shape: {depthandint.shape}")

    proj_H, proj_W, C = depthandint.shape

    image_channel = depthandint[:, :, 0]

    # Perform histogram equalization
    equalized_image = exposure.equalize_hist(image_channel)

    # Display the enhanced image.
    plt.figure(figsize=(6, 6))
    plt.imshow(equalized_image)

    # Obtain depth image data
    first_depth_view = depthandint[:, :, 0]

    zero_count_depth_0 = np.count_nonzero(first_depth_view == 0)
    zero_count_depth_1 = np.count_nonzero(first_depth_view < 0)

    print("The number of elements with a depth of 0:", zero_count_depth_0)
    print("The number of elements with a depth less than 0:", zero_count_depth_1)

    max_depth_value = np.max(first_depth_view)
    min_depth_value = np.min(first_depth_view)
    print("The maximum depth value:", max_depth_value)
    print("The minimum depth value:", min_depth_value)

    bins = np.arange(0, max_depth_value + 25, 25)  # From 0 to the maximum value, there is an interval of every 25.
    hist, edges = np.histogram(first_depth_view, bins=bins)

    # Calculate the total number.
    total_count = first_depth_view.size

    print("The proportion of elements less than 0 is", (zero_count_depth_1/total_count)*100)
    # Print the interval statistics and proportions.
    for i in range(len(hist)):
        count = hist[i]
        proportion = (count / total_count) * 100 if total_count > 0 else 0
        print(f"interval {edges[i]:.1f} - {edges[i + 1]:.1f} number: {count}, The proportion of the total number: {proportion:.2f}%")

    plt.axis('off')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    depth_image_path = os.path.join(save_dir, os.path.basename(file_path).replace('.npy', '_ri.png'))
    plt.savefig(depth_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Depth image saved to {depth_image_path}")

def visualize_int(file_path, save_dir):
    depthandint = np.load(file_path)

    # Print the dimensional information of the .npy file.
    print(f"Loaded data shape: {depthandint.shape}")

    proj_H, proj_W, C = depthandint.shape

    image_channel = depthandint[:, :, 1]

    # Perform histogram equalization
    equalized_image = exposure.equalize_hist(image_channel)

    # Display the enhanced image.
    plt.figure(figsize=(6, 6))
    plt.imshow(equalized_image)

    # Obtain intensity image data
    first_int_view = depthandint[:, :, 1]

    zero_count_int_0 = np.count_nonzero(first_int_view == 0)
    zero_count_int_1 = np.count_nonzero(first_int_view < 0)

    print("The number of elements with a intensity of 0:", zero_count_int_0)
    print("The number of elements with a intensity less than 0:", zero_count_int_1)

    # 计算并打印最大深度值
    max_int_value = np.max(first_int_view)
    min_int_value = np.min(first_int_view)
    print("The maximum intensity value:", max_int_value)
    print("The minimum intensity value:", min_int_value)

    bins = np.arange(0, max_int_value + 25, 25)  # 从0开始到最大值，每0.1一个区间
    hist, edges = np.histogram(first_int_view, bins=bins)

    total_count = first_int_view.size

    print("The proportion of elements less than 0 is", (zero_count_int_1/total_count)*100)
    for i in range(len(hist)):
        count = hist[i]
        proportion = (count / total_count) * 100 if total_count > 0 else 0
        print(f"interval {edges[i]:.1f} - {edges[i + 1]:.1f} number: {count}, The proportion of the total number: {proportion:.2f}%")

    plt.axis('off')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    int_image_path = os.path.join(save_dir, os.path.basename(file_path).replace('.npy', '_int.png'))
    plt.savefig(int_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Intensity image saved to {int_image_path}")

def visualize_selected_depthandint(file_path, save_dir):
    """
    This function is used to visualize a specified .npy file and save the generated image as a PNG file.
    Each type of image is saved as a separate PNG file.

    Parameters:
    file_paths (str or list of str): The path to the .npy file to be visualized
    save_dir (str): The folder path to save PNG images

    return:
    None
    """

    # Ensure the save path exists, if not, create it.
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Traverse each specified .npy file, visualize it, and save the visualization result.
    visualize_depth(file_path, save_dir)
    visualize_int(file_path, save_dir)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Visualize depth and intensity image data')

    parser.add_argument('--npy_folder', '--nf', type=str, required=True,
                        help='Specify the path of a certain depth and intensity image.')

    parser.add_argument('--dst_folder', '--df', type=str, required=True,
                        help='The save path for visualization results.')

    args = parser.parse_args()

    npy_folder = args.npy_folder
    dst_folder = args.dst_folder

    visualize_selected_depthandint(npy_folder, dst_folder)





