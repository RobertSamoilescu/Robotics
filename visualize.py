from model import *
from copy import deepcopy
from dataloader import *
import argparse
import cv2
import model
import torch
import torch.nn as nn
import torch.nn.functional as F


argparser = argparse.ArgumentParser()

argparser.add_argument("--src", type=str, required=True,
                       help="source directory")
args = argparser.parse_args()

# load dataset
validation_dataset = UPBDataset(args.src, augmentation=False)
validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=1)

# get map information
data_map = pickle.load(open("map_dir/map.pkl", "rb"))

# initialize model
# net = model.CAE().cuda()
net = torch.load("./checkpoints/best_model")


def construct_gaussian(coords):
    mu = np.array(coords)
    sigma = 10 * np.ones((2, ))

    # add offset
    mu -= np.array([data_map['x_min'], data_map['y_min']])
    mu[1] = data_map['height'] - mu[1]

    # construct gaussian
    x, y = np.meshgrid(
        np.linspace(0, data_map['height'], data_map['width']), 
        np.linspace(0, data_map['width'], data_map['height'])
    )

    # compute gaussian
    g = np.exp(-(x - mu[0])**2 / (2 * sigma[0]**2) - (y - mu[1])**2 / (2 * sigma[1]**2))

    # normalize for visual effect
    g = g / g.max()
    g = (g > 0.01).astype(np.float)

    return g



def display(img, map_img, distribution, fx=2.75, fy=2.5):
    img = img.transpose(1, 2, 0)
    img = 0.5 * (img + 1)


    # add dot in the map
    map_img = deepcopy(map_img)
    map_img[:,:, 2] = np.clip(map_img[:, :, 2] + distribution, 0, 1)

    # resize map_img
    img = cv2.resize(img, None, fx=fx, fy=fy)
    map_img = cv2.resize(map_img, (img.shape[1], img.shape[0]))

    # concat images
    full_view = np.hstack((map_img, img))

    cv2.imshow("MAP", full_view)
    cv2.waitKey(0)


def evaluate():
    net.eval()

    for i, data in enumerate(validation_dataloader, 0):
        # get inputs
        X = data['img']

        # send to gpu
        X = X.cuda()

        # forward, backward & optimize
        with torch.no_grad():
            Y = net(X)

        # transform to numpy
        X = X.cpu().numpy()[0]
        Y = Y.cpu().numpy()[0]

        display(X, data_map["img"], construct_gaussian(data["steer_coord"].numpy()[0]))            
        # display(X, data_map["img"], construct_gaussian(Y))

if __name__ == "__main__":
    evaluate()