import simple_map
import pickle
import os
import argparse
import cv2

argparser = argparse.ArgumentParser()

argparser.add_argument("--src", type=str, required=True,
                    help="source directory")
argparser.add_argument("--dst", type=str, required=True,
                    help="destination directory")
argparser.add_argument("--ref", type=str, required=False, default="train_raw", 
                    help="global reference directory (default: train_raw)")
args = argparser.parse_args()


def get_reference():
    json = sorted([os.path.join(args.ref, file) for file in os.listdir(args.ref) if file.endswith(".json")])[0]
    smap = simple_map.SimpleMap(json)
    return smap.northing, smap.easting

def construct_maps(jsons):
    cnt = 0

    # get first map as reference
    ref_globals = get_reference()
   
    for i in range(len(jsons)):
        smap = simple_map.SimpleMap(jsons[i], ref_globals)
        (x, y), (x_real, y_real), imgs = smap.get_route()

        # resize image
        imgs = [tuple(map(lambda x: cv2.resize(x, None, fx=0.2, fy=0.2), img)) for img in imgs]

        for j in range(0, len(imgs), 10):
            for k in range(3):
                cnt += 1
                path = os.path.join(args.dst, str(cnt))
                output_file = open(path, 'wb')
                obj = {"x_steer": x[j], "y_steer": y[j],
                       "x_utm": x_real[j], "y_utm": y_real[j],
                       "img": imgs[j][k]}
                pickle.dump(obj, output_file)
                output_file.close()

        print("* Video %d done, %s" %( i, jsons[i]))


def main():
    jsons = sorted([os.path.join(args.src, file) for file in os.listdir(args.src) if file.endswith(".json")])
    construct_maps(jsons)


if __name__ == "__main__":
    main()
