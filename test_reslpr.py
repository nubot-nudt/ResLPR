import argparse
import subprocess
from tqdm import tqdm
import numpy as np
import time

import torch
from torch.utils.data import DataLoader
import os
import torch.nn as nn 

from utils.dataset_utils import DeweatherDataset
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_npy, torch_to_np
from utils.schedulers import LinearWarmupCosineAnnealingLR
from net.model import ResLPR

import lightning.pytorch as pl
import torch.optim as optim
import re

def get_sorted_file_names(path):
    try:
        # Get the names of all sub-files in the specified path
        file_names = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        # Sort the file names in lexicographical order
        sorted_file_names = sorted(file_names)
        return sorted_file_names
    except FileNotFoundError:
        print(f"Error: The specified path {path} was not found.")
    except PermissionError:
        print(f"Error: You do not have permission to access the path {path}.")
    except Exception as e:
        print(f"An unknown error occurred: {e}")


class ResLPRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = ResLPR(decoder=True)
        self.loss_fn = nn.L1Loss()
    
    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = self.loss_fn(restored,clean_patch)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=150)
        return [optimizer],[scheduler]


def test_Deweather(net, dataset, output_path, which_epoch, task="deweather"):
    subprocess.check_output(['mkdir', '-p', testopt.output_path])
    dataset.set_dataset(task)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored = net(degrad_patch)
            save_path = os.path.join(output_path, degraded_name[0] + '.npy')
            restored_npy = save_npy(restored, save_path)
            clean_npy = torch_to_np(clean_patch)
            # restored_npy = image_np.detach().cpu().numpy()
            clean_npy = clean_npy.transpose(1, 2, 0)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored_npy, clean_npy)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

        # 指定输出文件路径
        output_file = os.path.join(output_path, 'deweather_results.txt')

        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))
        with open(output_file, 'a') as f:
            f.write("Epoch %d: PSNR: %.2f, SSIM: %.4f\n" % (which_epoch, psnr.avg, ssim.avg))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--mode', type=int, default=0, help='0 for deweather')
    parser.add_argument('--unrestored_folder', '--unf', type=str, default="/path_to/kitti_unrestored/", help='save path of unrestored data')
    parser.add_argument('--dst_folder', '--df', type=str, default="/path_to/kitti_restored/", help='output save path')
    parser.add_argument('--ckpt_folder', '--cf', type=str, default="/path_to/ResLPR/weights/", help='checkpoint save path')

    testopt = parser.parse_args()
    target_path_list = get_sorted_file_names(testopt.unrestored_folder)

    for target_path in target_path_list:
        print("Testing on target path: {}".format(target_path))
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.set_device(testopt.cuda)
        ckpt_path_list = testopt.ckpt_path
        # Read all the files in the folder.
        files = os.listdir(ckpt_path_list)
        # Filter out the files that end with '.ckpt' and extract the numbers of 'epoch'.
        weight_files = []
        for file in files:
            if file.endswith('.ckpt'):
                match = re.search(r'epoch=(\d+)', file)
                if match:
                    epoch = int(match.group(1))  # Convert the extracted epoch to an integer.
                    weight_files.append((epoch, os.path.join(ckpt_path_list, file)))

        # Sort according to the epoch number.
        weight_files.sort(key=lambda x: x[0])

        # Only keep the sorted file paths.
        sorted_weight_paths = [file_path for _, file_path in weight_files]

        for index, ckpt_name in enumerate(sorted_weight_paths):

            print("CKPT name : {}".format(ckpt_name))

            net = ResLPRModel.load_from_checkpoint(ckpt_name).cuda()
            net.eval()

            output_path = os.path.join(testopt.dst_folder, target_path)
            os.makedirs(output_path, exist_ok=True)

            if testopt.mode == 0:
                print('Start testing weather corruptions ')
                derain_set = DeweatherDataset(testopt, target_path=target_path)
                test_Deweather(net, derain_set, output_path, index, task="deweather")