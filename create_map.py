import cv2
import pickle as pkl
import numpy as np
import argparse
import os

from tqdm import tqdm


argparser = argparse.ArgumentParser()

argparser.add_argument("--src", type=str, required=True,
                       help="source directory")
argparser.add_argument("--dst", type=str, required=True,
						help="destination directory")
args = argparser.parse_args()

def read_files():
	# get list of files
	files = os.listdir(args.src)
	files = [os.path.join(args.src, file) for file in files]

	# allocate memory
	coords = np.zeros((2, len(files)))

	# read files and add coordinates
	for i in tqdm(range(len(files)), desc="Loading files"):
		data = pkl.load(open(files[i], "rb"))
		coords[0][i], coords[1][i] = data["x_steer"], data["y_steer"]

	# convert to int
	coords = np.round(coords).astype(np.int)
	
	# construct set
	coords_set = set()

	for i in range(len(files)):
		coord = tuple(coords[:, i])
		coords_set.add(coord)

	# export as pkl
	coords = list(coords_set)

	fout = open(os.path.join(args.dst, "coords.pkl"), "wb")
	pkl.dump(coords, fout)

	return coords


def draw_map(coords, padding=50, verbose=True):
	np_coords = np.array(coords)

	# find limits
	x_min, x_max = np_coords[:, 0].min(), np_coords[:, 0].max()
	y_min, y_max = np_coords[:, 1].min(), np_coords[:, 1].max()

	# add padding
	x_min, y_min = x_min - padding, y_min - padding
	x_max, y_max = x_max + padding, y_max + padding

	# compute matrix width, height
	width = x_max - x_min + 1
	height = y_max - y_min + 1

	# allocate memory for the map
	m = np.ones((height, width, 3)) 

	# add offset
	np_coords -= np.array([x_min, y_min])	

	# draw points
	for i in range(np_coords.shape[0]):
		cv2.circle(m, (np_coords[i][0], height - np_coords[i][1]), 5, (255, 0, 0), -1)

	# show map
	if verbose:
		cv2.imshow("Map", m)
		cv2.waitKey(0)


	# save as dictionary
	dict_map = dict()

	dict_map["img"] = m
	dict_map["width"] = width
	dict_map["height"] = height
	dict_map["x_min"], dict_map["x_max"] = x_min, x_max
	dict_map["y_min"], dict_map["y_max"] = y_min, y_max

	# export as pkl
	fout = open(os.path.join(args.dst, "map.pkl"), "wb")
	pkl.dump(dict_map, fout)


def main():
	coords = read_files()
	draw_map(coords)


if __name__ == "__main__":
    main()