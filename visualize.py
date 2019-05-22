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
argparser.add_argument("--project", action="store_true",
    help="project predicted point onto the map")
args = argparser.parse_args()

# load dataset
validation_dataset = UPBDataset(args.src, augmentation=False, test=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=8)

# get map information
data_map = pickle.load(open("map_dir/map.pkl", "rb"))

# read coodinates
coords = np.array(pickle.load(open("map_dir/coords.pkl", "rb")))

# initialize model
# net = model.CAE().cuda()
net = torch.load("./checkpoints/best_model")


def project_point(p):
    p = p[0].cpu().numpy()

    distances = np.array([np.linalg.norm(coord - p) for coord in coords])
    index = distances.argmin()

    return torch.from_numpy(coords[index]).unsqueeze(dim=0).float().cuda()


def corrected_coords(coords):
    mu = np.array(coords)

    # add offset
    mu -= np.array([data_map['x_min'], data_map['y_min']])
    mu[1] = data_map['height'] - mu[1]

    return mu


def display(img, map_img, coords, fx=2.75, fy=2.5):
    img = img.transpose(1, 2, 0)
    img = 0.5 * (img + 1)

    # add dot in the map
    map_img = deepcopy(map_img)
    cv2.circle(map_img, tuple(coords), 30, (0,255,0), -1)

    # resize map_img
    img = cv2.resize(img, None, fx=fx, fy=fy)
    map_img = cv2.resize(map_img, (img.shape[1], img.shape[0]))

    # concat images
    full_view = np.hstack((map_img, img))

    cv2.imshow("MAP", full_view)
    cv2.waitKey(33)


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

        if args.project:
        	Y = project_point(Y)

        # transform to numpy
        X = X.cpu().numpy()[0]
        Y = Y.cpu().numpy()[0]

        # display only central camera
        if i % 3 == 0:
	        # display(X, data_map["img"], corrected_coords(data["steer_coord"].numpy()[0]))            
	        display(X, data_map["img"], corrected_coords(Y))

if __name__ == "__main__":
    evaluate()