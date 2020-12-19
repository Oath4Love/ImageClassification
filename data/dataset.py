import os
import random
import glob
import numpy as np
import cv2
import torch
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader


class DogCat(Dataset):
    def __init__(self, root_path, training=False):
        self.image_file_list = glob.glob(os.path.join(root_path, "*.jpg"))

    def __getitem__(self, index):
        image_path = self.image_file_list[index]
        label = "0" if os.path.basename(image_path).split(".")[0] == "cat" else "1"
        image = io.imread(image_path)
        image = transform.resize(image, (256, 256, 3))
        image = torch.from_numpy(image)
        return image, label

    def __len__(self):
        return len(self.image_file_list)


if __name__ == "__main__":
    root_path = r"E:\LiHui\datasets\cat_dog\train"
    dog_cat_data = DogCat(root_path, training=True)
    train_loader = DataLoader(dog_cat_data, batch_size=4, shuffle=True, num_workers=4)
    for idx, (image, label) in enumerate(train_loader):
        print(image.shape)

