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
net = net.cuda()
net = net.eval()
print(net)

pp = pprint.PrettyPrinter(indent=4)

def RMSE():
    # criterion & total loss
    criterion = nn.MSELoss()
    losses = [] 

    for i, data in enumerate(dataloader, 0):
        # get inputs
        X, Y_gt = data['img'], data['steer_coord']

        # send to gpu
        X = X.cuda()
        Y_gt = Y_gt.cuda()

        # forward, backward & optimize
        with torch.no_grad():
            Y = net(X)

        # compute loss
        loss = criterion(Y, Y_gt)
        losses.append(loss.item())

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
    print(" * Computing Euclidian Distance")
    rmse_results = RMSE()
    pp.pprint(rmse_results)

    
if __name__ == "__main__":
    main()