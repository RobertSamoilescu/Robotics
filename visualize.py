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
net = torch.load("./checkpoints/distribution_std10/best_model")


def display(img, map_img, distribution, fx=2.75, fy=2.5):
    img = img.transpose(1, 2, 0)
    img = 0.5 * (img + 1)

    # resize distribution to map size
    distribution = cv2.resize(distribution, (map_img.shape[1], map_img.shape[0]))

    # scale for visual effect
    distribution /= distribution.max()
    distribution = (distribution > 0.01).astype(np.float)

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
            Y = F.softmax(net(X).reshape(X.shape[0], -1), dim=1)
            Y = Y.reshape(Y.shape[0], data["gaussian"].shape[1], -1)

        # transform to numpy
        X = X.cpu().numpy()[0]
        Y = Y.cpu().numpy()[0]

        # display(X, data_map["img"], data["gaussian"].numpy()[0])            
        display(X, data_map["img"], Y)

if __name__ == "__main__":
    evaluate()