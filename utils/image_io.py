import glob
import os
import torch
import torchvision
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# import skvideo.io

matplotlib.use('agg')

def process_channels(arr):
    """
    Process a NumPy array with the shape of 64*1920*2 or 32*1440*2.

    Perform the following operations on the data of each channel:

        Set the numbers less than 0 to -1.
        Set the numbers greater than 255 to 255.
        Change the data type of all elements to int16.

    Parameters:
    arr (numpy.ndarray): The input array with the shape of 64*1920*2 or 32*1440*2.

    Returns:
    numpy.ndarray: The processed array.
    """
    # 确保输入数组的形状是 64*1920*2
    if arr.shape != (64, 1920, 2) and arr.shape != (32, 1440, 2):
        raise ValueError("The shape of the input array must be either 64*1920*2 or 32*1440*2")

    # Create a new array to store the processed data.
    processed_arr = np.empty_like(arr, dtype=np.int16)

    # Process the data for each channel.
    for i in range(2):
        channel_data = arr[:, :, i]
        channel_data = np.where(channel_data < 0, -1, channel_data)
        channel_data = np.where(channel_data > 255, 255, channel_data)
        processed_arr[:, :, i] = channel_data.astype(np.int16)
    return processed_arr


def save_npy(image_tensor, output_path="output/"):
    image_np = torch_to_np(image_tensor)
    restored_npy = image_np.transpose(1, 2, 0)
    restored_npy_handled = process_channels(restored_npy)

    print(restored_npy_handled.shape)
    print("save path:", output_path)
    np.save(output_path, restored_npy_handled)
    return restored_npy_handled


def np_to_torch(img_np):
    """
    Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]

    :param img_np:
    :return:
    """
    return torch.from_numpy(img_np)[None, :]


def torch_to_np(img_var):
    """
    Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    :param img_var:
    :return:
    """
    return img_var.detach().cpu().numpy()[0]
