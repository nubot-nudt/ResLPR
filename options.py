import argparse
parser = argparse.ArgumentParser()
# Input Parameters
parser.add_argument('--cuda', type=int, default=0)

parser.add_argument('--epochs', type=int, default=120, help='maximum number of epochs to train the total model.')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size to use per GPU')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate of encoder.')
parser.add_argument('--de_type', nargs='+', default=['defog_kitti', 'desnow_kitti', 'derain_kitti', 'defog_nclt', 'desnow_nclt', 'derain_nclt'],
                    help='which type of degradations is training and testing for.')
parser.add_argument('--patch_height', type=int, default=32)
parser.add_argument('--patch_width', type=int, default=480)

# Make changes according to the number of CPU cores, which determines the data loading speed.
parser.add_argument('--num_workers', type=int, default=16, help='number of workers.')

parser.add_argument('--resume', type=bool, default=False, help='weather to resume training or not')
parser.add_argument('--resume_ckpt', type=str, default='weights/', help='path to load the checkpoint.')

parser.add_argument('--data_file_dir', type=str, default='/path_to/ResLPR/data_dir/',  help='where clean images of denoising saves.')

parser.add_argument('--defog_kitti_dir', type=str, default='/path_to/train_kitti_data/fog/',
                    help='where clean images of defog_kitti saves.')
parser.add_argument('--desnow_kitti_dir', type=str, default='/path_to/train_kitti_data/snow/',
                    help='where training images of desnow_kitti saves.')
parser.add_argument('--derain_kitti_dir', type=str, default='/path_to/train_kitti_data/rainy/',
                    help='where training images of derain_kitti saves.')
parser.add_argument('--defog_nclt_dir', type=str, default='/path_to/train_nclt_data/fog/',
                    help='where clean images of defog_nclt saves.')
parser.add_argument('--desnow_nclt_dir', type=str, default='/path_to/train_nclt_data/snow/',
                    help='where training images of desnow_nclt saves.')
parser.add_argument('--derain_nclt_dir', type=str, default='/path_to/train_nclt_data/rainy/',
                    help='where training images of derain_nclt saves.')

parser.add_argument('--output_path', type=str, default="/path_to/train_results/output_dir/", help='output save path')
parser.add_argument('--ckpt_path', type=str, default="/path_to/train_results/ckpt_path/", help='checkpoint save path')
parser.add_argument('--wblogger', type=str, default="ResLPR", help='Determine to log to wandb or not and the project name')
parser.add_argument('--ckpt_dir', type=str, default="/path_to/train_results/ckpt_dir/", help='Name of the Directory where the checkpoint is to be saved')
parser.add_argument('--num_gpus', type=int, default=4, help='Number of GPUs to use for training')
options = parser.parse_args()