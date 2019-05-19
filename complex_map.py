import simple_map
import matplotlib.pyplot as plt
import cv2
import pickle as pkl
import os
import argparse
import numpy as np


# Create argparser
argparser = argparse.ArgumentParser()

argparser.add_argument("--src", type=str, required=True,
                       help="source directory")
argparser.add_argument("--dst", type=str, required=True,
                       help="destination directory")
args = argparser.parse_args()


class ComplexMap():
    def __init__(self, jsons):
        self.complex_map = []

        # get first map as reference
        smap = simple_map.SimpleMap(jsons[0])
        self.complex_map.append(smap.get_route())
        ref_globals = (smap.northing, smap.easting)
        print(" * Video 0 done")
       
        for i in range(1, len(jsons)):
            smap = simple_map.SimpleMap(jsons[i], ref_globals)
            self.complex_map.append(smap.get_route())
            print(" * Video", i, "done")

    def draw_map(self):
        xs = []; ys =[]

        for smap in self.complex_map:
            (x, y), (x_real, y_real), _ = smap

            # append coordinates
            xs += x; ys += y

            # plot simple map
            plt.scatter(x, y, c="#3399ff")
            # plt.scatter(x_real, y_real, c='#3399ff')

        plt.axis('equal')
        plt.show()

        # construct complex map
        cmap = np.array([xs, ys])
        return cmap

    def export_pkl(self):
        cnt = 0

        for i in range(len(self.complex_map)):
            print(" * Exporting file", i)
            (x, y), (x_real, y_real), imgs = self.complex_map[i]

            for j in range(len(imgs)):
                for k in range(3):
                    cnt += 1

                    path = os.path.join(args.dst, str(cnt))
                    output_file = open(path, 'wb')
                    obj = {
                        "x_steer": x[j], "y_steer": y[j],
                        "x_utm": x_real[j], "y_utm": y_real[j],
                        "img": imgs[j][k]
                    }
                    pkl.dump(obj, output_file)
                    output_file.close()


def main():
    # get json files
    jsons = [os.path.join(args.src, file) for file in os.listdir(args.src) if file.endswith(".json")][:10]

    # create & draw complex map
    cmap = ComplexMap(jsons)
    points = cmap.draw_map()
    
    # export points to pkl
    file = open("map.pkl", "wb")
    pkl.dump(points, file)


if __name__ == "__main__":
    main()
