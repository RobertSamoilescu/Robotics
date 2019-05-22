import pickle
import os
import numpy as np
import cv2
import imgaug as ig
from imgaug import augmenters as iga
from torch.utils.data import Dataset, DataLoader

class UPBDataset(Dataset):
    """UPB dataset"""

    def __init__(self, root_dir, augmentation=True, img_size=128, test=False):
        self.root_dir = root_dir
        
        if test:
            self.files = [os.path.join(root_dir, file) for file in sorted(os.listdir(root_dir), key=int)]
        else:
            self.files = [os.path.join(root_dir, file) for file in os.listdir(root_dir)]
        
        self.augmentation = augmentation
        self.img_size = img_size

        if augmentation:    
            st = lambda aug: iga.Sometimes(0.4, aug)
            oc = lambda aug: iga.Sometimes(0.3, aug)
            rl = lambda aug: iga.Sometimes(0.09, aug)
            self.seq = iga.Sequential(
                [
                    rl(iga.GaussianBlur(
                        (0, 1.5))),  # blur images with a sigma between 0 and 1.5
                    rl(
                        iga.AdditiveGaussianNoise(
                            loc=0, scale=(0.0, 0.05),
                            per_channel=0.5)),  # add gaussian noise to images
                    oc(iga.Dropout((0.0, 0.10), per_channel=0.5)
                       ),  # randomly remove up to X% of the pixels
                    oc(
                        iga.CoarseDropout(
                            (0.0, 0.10), size_percent=(0.08, 0.2), per_channel=0.5)
                    ),  # randomly remove up to X% of the pixels
                    oc(
                        iga.Add((-40, 40), per_channel=0.5)
                    ),  # change brightness of images (by -X to Y of original value)
                    st(iga.Multiply((0.10, 2.5), per_channel=0.2)
                       ),  # change brightness of images (X-Y% of original value)
                    rl(iga.ContrastNormalization(
                        (0.5, 1.5),
                        per_channel=0.5)),  # improve or worsen the contrast
                ],
                random_order=True)

        # get map information
        self.data_map = pickle.load(open("map_dir/map.pkl", "rb"))


    def __construct_guassian(self, *coords):
        mu = np.array(coords)
        sigma = 10 * np.ones((2, ))

        # add offset
        mu -= np.array([self.data_map['x_min'], self.data_map['y_min']])
        mu[1] = self.data_map['height'] - mu[1]

        # construct gaussian
        x, y = np.meshgrid(
            np.linspace(0, self.data_map['height'], self.data_map['width']), 
            np.linspace(0, self.data_map['width'], self.data_map['height'])
        )
        
        # compute gaussian
        g = np.exp(-(x - mu[0])**2 / (2 * sigma[0]**2) - (y - mu[1])**2 / (2 * sigma[1]**2))
        
        # scale to img_size corresponding to network output
        g = cv2.resize(g, (self.img_size, self.img_size))

        # normalize
        g = g / g.sum()

        # Visualize       
        # location = cv2.resize(self.data_map['img'], g.shape)
        # location[:,:, 2] += 255 * g
        # cv2.imshow("MAP", location)
        # cv2.waitKey(0)

        return g


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # read file
        file = self.files[idx]
        file = open(file, 'rb')
        content = pickle.load(file)
        file.close()

        # resize image to make it square
        content['img'] = cv2.resize(content['img'], (self.img_size, self.img_size))
        
        # normalize and transpose image
        if self.augmentation:
            content['img'] = self.seq.augment_image(content['img'])
        
        content['img'] = content['img'].transpose(2, 0, 1).astype(dtype=np.float32)
        content['img'] = (content['img'] - 128.) / 128.

        # construct output gaussian
        gaussian_map = self.__construct_guassian(content['x_steer'], content['y_steer'])

        return {
            "steer_coord": np.array([content['x_steer'], content['y_steer']], dtype=np.float32),
            "utm_coord": np.array([content['x_utm'], content['y_utm']], dtype=np.float32),
            "img": content['img'],
            "gaussian": gaussian_map
        }

if __name__ == "__main__":
    dataset = UPBDataset("train")
    for i in range(len(dataset)):
        img = dataset[i]['img']
        img = (img + 1) * 0.5

        cv2.imshow("IMG", img.transpose(1, 2, 0))
        cv2.waitKey(0)
