import os
import random
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from utils.image_utils import random_augmentation, crop_img

class ResLPRTrainDataset(Dataset):
    def __init__(self, args):
        super(ResLPRTrainDataset, self).__init__()
        self.args = args
        self.fog_kitti_ids = []
        self.snow_kitti_ids = []
        self.rain_kitti_ids = []

        self.fog_nclt_ids = []
        self.snow_nclt_ids = []
        self.rain_nclt_ids = []

        self.de_temp = 0
        self.de_type = self.args.de_type
        print(self.de_type)
        # Define a dictionary for indexing the corresponding types of snow, fog, and rain.
        self.de_dict = {'defog_kitti': 0, 'desnow_kitti': 1, 'derain_kitti': 2, 'defog_nclt': 3, 'desnow_nclt': 4, 'derain_nclt': 5}
        self._init_ids()  # Call the _init_ids() method to get the list of file names.
        self._merge_ids()  # Call the _merge_ids() method to merge different lists of file names into one list.

        self.toTensor = ToTensor()

    def _init_ids(self):
        if 'defog_kitti' in self.de_type:
            self._init_fog_kitti_ids()
        if 'desnow_kitti' in self.de_type:
            self._init_snow_kitti_ids()
        if 'derain_kitti' in self.de_type:
            self._init_rain_kitti_ids()
        if 'defog_nclt' in self.de_type:
            self._init_fog_nclt_ids()
        if 'desnow_nclt' in self.de_type:
            self._init_snow_nclt_ids()
        if 'derain_nclt' in self.de_type:
            self._init_rain_nclt_ids()

        random.shuffle(self.de_type)

    def _init_fog_kitti_ids(self):
        temp_ids = []
        fog_kitti_file_list = self.args.data_file_dir + "/kitti_fog.txt"
        temp_ids += [self.args.defog_kitti_dir + id_.strip() for id_ in open(fog_kitti_file_list)]
        self.fog_kitti_ids = [{"clean_id": x, "de_type": 0} for x in temp_ids]

        self.fog_kitti_counter = 0

        self.num_kitti_fog = len(self.fog_kitti_ids)
        print("Total Fog KITTI Ids : {}".format(self.num_kitti_fog))

    def _init_snow_kitti_ids(self):
        temp_ids = []
        snow_kitti_file_list = self.args.data_file_dir + "/kitti_snow.txt"
        temp_ids += [self.args.desnow_kitti_dir + id_.strip() for id_ in open(snow_kitti_file_list)]
        self.snow_kitti_ids = [{"clean_id": x, "de_type": 1} for x in temp_ids]

        self.snow_kitti_counter = 0

        self.num_kitti_snow = len(self.snow_kitti_ids)
        print("Total Snow KITTI Ids : {}".format(self.num_kitti_snow))

    def _init_rain_kitti_ids(self):
        temp_ids = []
        rain_kitti_file_list = self.args.data_file_dir + "/kitti_rain.txt"
        temp_ids += [self.args.derain_kitti_dir + id_.strip() for id_ in open(rain_kitti_file_list)]
        self.rain_kitti_ids = [{"clean_id": x, "de_type": 2} for x in temp_ids]

        self.rain_kitti_counter = 0

        self.num_kitti_rain = len(self.rain_kitti_ids)
        print("Total Rain KITTIIds : {}".format(self.num_kitti_rain))

    def _init_fog_nclt_ids(self):
        temp_ids = []
        fog_nclt_file_list = self.args.data_file_dir + "/nclt_fog.txt"
        temp_ids += [self.args.defog_nclt_dir + id_.strip() for id_ in open(fog_nclt_file_list)]
        self.fog_nclt_ids = [{"clean_id": x, "de_type": 3} for x in temp_ids]

        self.fog_nclt_counter = 0

        self.num_nclt_fog = len(self.fog_nclt_ids)
        print("Total Fog NCLT Ids : {}".format(self.num_nclt_fog))

    def _init_snow_nclt_ids(self):
        temp_ids = []
        snow_nclt_file_list = self.args.data_file_dir + "/nclt_snow.txt"
        temp_ids += [self.args.desnow_nclt_dir + id_.strip() for id_ in open(snow_nclt_file_list)]
        self.snow_nclt_ids = [{"clean_id": x, "de_type": 4} for x in temp_ids]

        self.snow_nclt_counter = 0

        self.num_nclt_snow = len(self.snow_nclt_ids)
        print("Total Snow NCLT Ids : {}".format(self.num_nclt_snow))

    def _init_rain_nclt_ids(self):
        temp_ids = []
        rain_nclt_file_list = self.args.data_file_dir + "/nclt_rain.txt"
        temp_ids += [self.args.derain_nclt_dir + id_.strip() for id_ in open(rain_nclt_file_list)]
        self.rain_nclt_ids = [{"clean_id": x, "de_type": 5} for x in temp_ids]

        self.rain_nclt_counter = 0

        self.num_nclt_rain = len(self.rain_nclt_ids)
        print("Total rain NCLT Ids : {}".format(self.num_nclt_rain))

    def load_and_crop_npy(self, file_path, base=16):
        """
        Read the .npy file and crop the image
        """
        image = np.load(file_path)  # Load the .npy file
        cropped_image = crop_img(image, base)
        return cropped_image

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]

        # Ensure the patch size is within the image dimensions
        if H < self.args.patch_height or W < self.args.patch_width:
            raise ValueError("Patch size is larger than the image dimensions.")

        ind_H = random.randint(0, H - self.args.patch_height)
        ind_W = random.randint(0, W - self.args.patch_width)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_height, ind_W:ind_W + self.args.patch_width]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_height, ind_W:ind_W + self.args.patch_width]

        return patch_1, patch_2

    def _get_fog_gt_name(self, fog_name):
        gt_name = fog_name.split("fog")[0] + 'clean' + fog_name.split('fog')[-1]
        return gt_name

    def _get_snow_gt_name(self, snow_name):
        gt_name = snow_name.split("snow")[0] + 'clean' + snow_name.split('snow')[-1]
        return gt_name

    def _get_rain_gt_name(self, rain_name):
        gt_name = rain_name.split("rainy")[0] + 'clean' + rain_name.split('rainy')[-1]
        return gt_name

    def _merge_ids(self):
        self.sample_ids = []
        if "defog_kitti" in self.de_type:
            self.sample_ids += self.fog_kitti_ids
        if "desnow_kitti" in self.de_type:
            self.sample_ids += self.snow_kitti_ids
        if "derain_kitti" in self.de_type:
            self.sample_ids += self.rain_kitti_ids
        if "defog_nclt" in self.de_type:
            self.sample_ids += self.fog_nclt_ids
        if "desnow_nclt" in self.de_type:
            self.sample_ids += self.snow_nclt_ids
        if "derain_nclt" in self.de_type:
            self.sample_ids += self.rain_nclt_ids
        print(len(self.sample_ids))  # The number of corrupted samples after merging

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        de_id = sample["de_type"]

        if de_id == 0:
            # kitti_fog
            degrad_img = self.load_and_crop_npy(sample["clean_id"], base=16)
            clean_name = self._get_fog_gt_name(sample["clean_id"])  # Obtain the file name of the corresponding clean image by means of replacement.
            clean_img = self.load_and_crop_npy(clean_name, base=16)
        elif de_id == 1:
            # kitti_snow
            degrad_img = self.load_and_crop_npy(sample["clean_id"], base=16)
            clean_name = self._get_snow_gt_name(sample["clean_id"])
            clean_img = self.load_and_crop_npy(clean_name, base=16)
        elif de_id == 2:
            # kitti_rain
            degrad_img = self.load_and_crop_npy(sample["clean_id"], base=16)
            clean_name = self._get_rain_gt_name(sample["clean_id"])
            clean_img = self.load_and_crop_npy(clean_name, base=16)
        elif de_id == 3:
            # nclt_fog
            degrad_img = self.load_and_crop_npy(sample["clean_id"], base=16)
            clean_name = self._get_fog_gt_name(sample["clean_id"])
            clean_img = self.load_and_crop_npy(clean_name, base=16)
        elif de_id == 4:
            # nclt_snow
            degrad_img = self.load_and_crop_npy(sample["clean_id"], base=16)
            clean_name = self._get_snow_gt_name(sample["clean_id"])
            clean_img = self.load_and_crop_npy(clean_name, base=16)
        elif de_id == 5:
            # nclt_rain
            degrad_img = self.load_and_crop_npy(sample["clean_id"], base=16)
            clean_name = self._get_rain_gt_name(sample["clean_id"])
            clean_img = self.load_and_crop_npy(clean_name, base=16)

        degrad_patch, clean_patch = random_augmentation(*self._crop_patch(degrad_img, clean_img))

        # Convert the data into tensors.
        clean_patch = self.toTensor(clean_patch).float()  # Convert it to a FloatTensor and place it on the GPU.
        degrad_patch = self.toTensor(degrad_patch).float()

        return [clean_name, de_id], degrad_patch, clean_patch

    def __len__(self):
        return len(self.sample_ids)


class DeweatherDataset(Dataset):
    def __init__(self, args, task="deweather", target_path=None):
        super(DeweatherDataset, self).__init__()
        self.ids = []
        self.task_idx = 0
        self.args = args

        self.task_dict = {'deweather': 0}
        self.toTensor = ToTensor()
        self.target_path = target_path

        self.set_dataset(task)


    def load_and_crop_npy(self, file_path, base=16):
        """
        Read the .npy file and crop the image.
        """
        image = np.load(file_path)
        cropped_image = crop_img(image, base)
        return cropped_image

    def _init_input_ids(self):
        if self.task_idx == 0:
            self.ids = []
            files_path = os.path.join(self.args.deweather_path, self.target_path)
            name_list = os.listdir(files_path)
            self.ids += [files_path + f"/{id_}" for id_ in name_list]
        self.length = len(self.ids)

    def _get_gt_path(self, degraded_name):
        gt_name = degraded_name.replace(self.target_path, "clean")
        return gt_name

    def set_dataset(self, task):
        self.task_idx = self.task_dict[task]
        self._init_input_ids()

    def __getitem__(self, idx):
        degraded_path = self.ids[idx]
        clean_path = self._get_gt_path(degraded_path)

        degraded_img = self.load_and_crop_npy(degraded_path, base=16)
        clean_img = self.load_and_crop_npy(clean_path, base=16)

        clean_img, degraded_img = self.toTensor(clean_img).float(), self.toTensor(degraded_img).float()
        degraded_name = degraded_path.split('/')[-1][:-4]

        return [degraded_name], degraded_img, clean_img

    def __len__(self):
        return self.length