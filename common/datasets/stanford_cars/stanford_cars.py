import logging
import os
import pickle
from subprocess import call

import numpy as np
import scipy.io
from imageio import imread
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

train_transform = transforms.Compose(
    [
        transforms.Resize(250),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199)),
    ]
)
test_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.46905602, 0.45872932, 0.4539325), (0.26603131, 0.26460057, 0.26935185)),
    ]
)


larger_train_transform = transforms.Compose(
    [
        transforms.Resize((400, 400)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
larger_test_transform = transforms.Compose(
    [transforms.Resize((400, 400)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
class_num = 196


def is_greyscale(img_path):
    image = imread(img_path)
    if len(image.shape) < 3:
        return True
    elif len(image.shape) == 3:
        return False
    else:
        raise AssertionError()


def build_folder_is_greyscale(fold_path):
    all_fnames = os.listdir(fold_path)
    files_is_grescale = {
        fname for fname in tqdm(all_fnames, desc="scanning folder") if not is_greyscale(f"{fold_path}/{fname}")
    }
    return files_is_grescale


class StanfordCarsDataset(Dataset):
    """
    Stanford Cars fine-grained image classification dataset.
    http://ai.stanford.edu/~jkrause/cars/car_dataset.html
    """

    def __init__(self, root, train=True, only_rgb=True, download=True, transform=None):
        """

        Args:
            root: folder that holds the data. Actual data will be downloaded to root/stanford-cars
            train: Whether this is a train/test split
            only_rgb: Whether to filter images only to RGB
            download: Whether to download the files if the do not exists in $root
            transform: Optional transform to be applied on a sample.
        """
        root = os.path.join(root, "stanford-cars")
        if download:
            if not os.path.exists(f"{root}/devkit/"):
                os.makedirs(root, exist_ok=True)
                status = call(["bash", "./download.sh", root])
                assert status == 0, "download failed"
            else:
                logging.info(f"root folder {root} exists, to download again delete it.")

        if train:
            mat_anno = os.path.join(root, "devkit/cars_train_annos.mat")
        else:
            mat_anno = os.path.join(root, "devkit/cars_test_annos_withlabels.mat")
        split_str = "train" if train else "test"
        data_dir = os.path.join(root, f"cars_{split_str}")
        car_names = os.path.join(root, "devkit/cars_meta.mat")
        self.full_data_set = scipy.io.loadmat(mat_anno)
        self.car_annotations = self.full_data_set["annotations"]
        self.car_annotations = self.car_annotations[0]

        if only_rgb:
            cleaned_annos = []
            logging.info("Cleaning up data set (only take pics with rgb channels)...")
            filter_path = f"{root}/devkit/rgb_filter_{split_str}.pkl"
            if os.path.exists(filter_path):
                with open(filter_path, "rb") as f:
                    rgb_files_set = pickle.load(f)
            else:
                rgb_files_set = build_folder_is_greyscale(data_dir)
                with open(filter_path, "wb") as f:
                    pickle.dump(rgb_files_set, f)

            for c in self.car_annotations:
                if c[-1][0] in rgb_files_set:
                    cleaned_annos.append(c)
            self.car_annotations = cleaned_annos

        self.car_names = scipy.io.loadmat(car_names)["class_names"]
        self.car_names = np.array(self.car_names[0])

        self.classes = [name[0] for name in self.car_names]
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.car_annotations)

    def __getitem__(self, idx):
        # an improvement can be to move all to hdf5 which make data loading much faster
        img_name = os.path.join(self.data_dir, self.car_annotations[idx][-1][0])
        image = Image.open(img_name)
        car_class = int(self.car_annotations[idx][-2][0][0]) - 1

        if self.transform:
            image = self.transform(image)

        return image, car_class

    def map_class(self, id):
        id = np.ravel(id)
        ret = self.car_names[id - 1][0][0]
        return ret
