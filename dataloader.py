import pickle
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2

class UPBDataset(Dataset):
    """UPB dataset"""

    def __init__(self, root_dir, width=100, height=80):
        self.root_dir = root_dir
        self.width = width
        self.height=height
        self.files = [os.path.join(root_dir, file) for file in os.listdir(root_dir)]


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # read file
        file = self.files[idx]
        file = open(file, 'rb')
        content = pickle.load(file)
        file.close()

        # normalize and transpose image
        content['img'] = cv2.resize(content['img'], (self.width, self.height))
        content['img'] = content['img'].transpose(2, 0, 1).astype(dtype=np.float32)
        content['img'] = (content['img'] - 128.) / 128.

        return {
            "steer_coord": np.array([content['x_steer'], content['y_steer']], dtype=np.float32),
            "utm_coord": np.array([content['x_utm'], content['y_utm']], dtype=np.float32),
            "img": content['img']
        }

if __name__ == "__main__":
    dataset = UPBDataset("dataset")
    for i in range(len(dataset)):
        img = dataset[i]['img']
        cv2.imshow("IMG", img.transpose(1, 2, 0))
        cv2.waitKey(0)
