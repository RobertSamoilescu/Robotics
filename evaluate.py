import torch
import torch.nn as nn
import torch.nn.functional as F
import model
import matplotlib.pyplot as plt
import pprint
import argparse


from tqdm import tqdm
from logger import Logger
from model import *

argparser = argparse.ArgumentParser()

argparser.add_argument("--src", type=str, required=True,
    help="source directory")
argparser.add_argument("--project", action="store_true",
    help="project predicted point onto the map")
argparser.add_argument("--outliers", action="store_true",
    help="remove outilers")
args = argparser.parse_args()

# constants
batchsize = 1
num_workers = 8

# read all data
dataset = UPBDataset(args.src)
dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=num_workers)

# read data map
data_map = pickle.load(open("map_dir/map.pkl", "rb"))

# read coodinates
coords = np.array(pickle.load(open("map_dir/coords.pkl", "rb")))

# initialize model
net = torch.load("./checkpoints/best_model").cuda()
net = net.eval()
print(net)

pp = pprint.PrettyPrinter(indent=4)


def project_point(p):
    p = p[0].cpu().numpy()

    distances = np.array([np.linalg.norm(coord - p) for coord in coords])
    index = distances.argmin()

    return torch.from_numpy(coords[index]).unsqueeze(dim=0).float().cuda()


def RMSE():
    # criterion & total loss
    criterion = nn.MSELoss()
    losses = []
    points = []

    for i, data in tqdm(enumerate(dataloader, 0), total=len(dataloader)):
        # get inputs
        X, Y_gt = data['img'], data['steer_coord']

        # send to gpu
        X = X.cuda()
        Y_gt = Y_gt.cuda()

        # forward, backward & optimize
        with torch.no_grad():
            Y = net(X)

        # project point
        if args.project:
            Y = project_point(Y)

        # compute loss
        loss = criterion(Y, Y_gt)
        losses.append(np.sqrt(loss.item()))
        points.append(Y.cpu().numpy())


    # plot 
    plt.bar(np.arange(len(losses)), losses, align='center', alpha=0.5, linewidth=10)
    plt.title("Euclidian distance")
    plt.xlabel("frame")
    plt.ylabel("distance[m]")
    plt.show()


    mean_RMSE = np.mean(losses)
    max_RMSE = np.max(losses)
    min_RMSE = np.min(losses)

    ret = dict()
    ret["losses"] = losses
    ret["mean_distance"] = np.mean(losses)
    ret["max_distance"] = np.max(losses)
    ret["min_distance"] = np.min(losses)
    return ret


def plot_histograms(rmse_results):
    bins = np.arange(0, int(rmse_results["max_distance"]) + 1, 5)
    
    # plot distribution
    distribution = plt.hist(rmse_results["losses"], bins, density=True)
    plt.title("Distance distribution")
    plt.xlabel("distance[m]")
    plt.ylabel("fraction")
    plt.show()

    # plot cumulative distribution
    cumulative = plt.hist(rmse_results["losses"], bins, 
        histtype='step', density=True, cumulative=True)
    plt.title("Cumulative distance distribution")
    plt.xlabel("distance [m]")
    plt.ylabel("fraction")    
    plt.show()

    ret = dict()
    ret["distribution"] = distribution
    ret["cumulative"] = cumulative

    return ret



def RMSE_denoise(rmse_results, threshold=50):
    losses = np.array(rmse_results["losses"])
    mask = losses < 50

    losses = losses[mask]

    mean_RMSE = np.mean(losses)
    max_RMSE = np.max(losses)
    min_RMSE = np.min(losses)

    ret = dict()
    ret["losses"] = losses
    ret["mean_distance"] = np.mean(losses)
    ret["max_distance"] = np.max(losses)
    ret["min_distance"] = np.min(losses)
    ret["percentage"] = np.mean(mask)
    return ret


def main():
    # compute raw results
    print(" * Computing Euclidian Distance")
    rmse_results = RMSE()

    if not args.outliers:
        pp.pprint(rmse_results) 

    # plot distribution an cumulative distnaces    
    print(" * Plot histogram of distances ")
    plot_histograms(rmse_results)

    # compute results without outliers
    if args.outliers:
        print(" *  Compute Euclidian Distance without outliers")
        rmse_results = RMSE_denoise(rmse_results)

        pp.pprint(rmse_results)

    
if __name__ == "__main__":
    main()