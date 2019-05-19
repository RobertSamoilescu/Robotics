import torch
import torch.nn as nn
import torch.nn.functional as F
import model
import matplotlib.pyplot as plt
import pprint
import argparse


from logger import Logger
from model import *

argparser = argparse.ArgumentParser()

argparser.add_argument("--src", type=str, required=True,
                       help="source directory")
args = argparser.parse_args()


# constants
batchsize = 1
num_workers = 8

# read all data
dataset = UPBDataset(args.src)
dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=num_workers)

# read data map
data_map = pickle.load(open("map_dir/map.pkl", "rb"))

# initialize model
net = torch.load("./checkpoints/best_model")
net = net.eval()
print(net)

pp = pprint.PrettyPrinter(indent=4)


def KLDiv():
    # criterion & total loss
    criterion = nn.KLDivLoss(reduction='batchmean')
    losses = [] 

    for i, data in enumerate(dataloader, 0):
        # get inputs
        X, Y_gt = data['img'], data['gaussian']

        # send to gpu
        X = X.cuda()
        Y_gt = Y_gt.cuda()

        # forward, backward & optimize
        with torch.no_grad():
            Y = net(X)

        # reshape
        Y = Y.reshape(Y.shape[0], -1)
        Y = F.log_softmax(Y, dim=1)
        
        Y_gt = Y_gt.reshape(Y_gt.shape[0], -1)
        
        # compute loss
        losses.append(criterion(Y, Y_gt.float()).item())


    # plot
    plt.bar(np.arange(len(losses)), losses, align='center', alpha=0.5)
    plt.title("KL Divergence")
    plt.xlabel("frame")
    plt.ylabel("value")
    plt.show()

    mean_KL = np.mean(losses)
    max_KL = np.max(losses)
    min_KL = np.min(losses)

    ret = dict()  
    ret["mean_KL"] = np.mean(losses)
    ret["max_KL"] = np.max(losses)
    ret["min_KL"] = np.min(losses)
    return ret


def euclidian_distance(distribution, coords):
    # resize distribution to map size
    distribution = cv2.resize(distribution, (data_map["img"].shape[1], data_map["img"].shape[0]))
    idx = distribution.argmax()
    predicted_coords = np.unravel_index(idx, distribution.shape)

    # correct coords
    coords -= np.array([data_map['x_min'], data_map['y_min']])
    coords[1] = data_map['height'] - coords[1]
    coords = np.flip(coords)

    return np.linalg.norm(coords - predicted_coords)


def RMSE():
    # criterion & total loss
    criterion = nn.MSELoss()
    losses = [] 

    for i, data in enumerate(dataloader, 0):
        # get inputs
        X, Y_gt = data['img'], data['steer_coord']

        # send to gpu
        X = X.cuda()

        # forward, backward & optimize
        with torch.no_grad():
            Y = net(X)

        # compute loss
        losses.append(euclidian_distance(Y.cpu().numpy()[0, 0], Y_gt.numpy()[0]))

    # plot 
    plt.bar(np.arange(len(losses)), losses, align='center', alpha=0.5)
    plt.title("Euclidian distance")
    plt.xlabel("frame")
    plt.ylabel("distance[m]")
    plt.show()


    mean_KL = np.mean(losses)
    max_KL = np.max(losses)
    min_KL = np.min(losses)

    ret = dict()  
    ret["mean_distance"] = np.mean(losses)
    ret["max_distance"] = np.max(losses)
    ret["min_distance"] = np.min(losses)
    return ret


def main():
    print(" * Computing KL Divergence")
    kl_results = KLDiv()
    pp.pprint(kl_results)
    

    print(" * Computing Euclidian Distance")
    rmse_results = RMSE()
    pp.pprint(rmse_results)

    
if __name__ == "__main__":
    main()