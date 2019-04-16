import simple_map
import matplotlib.pyplot as plt
import cv2
import pickle
import os
   

class ComplexMap():
    def __init__(self, jsons):
        self.complex_map = []

        # get first map as reference
        smap = simple_map.SimpleMap(jsons[0])
        self.complex_map.append(smap.get_route())
        ref_globals = (smap.northing, smap.easting)
       
        for i in range(1, len(jsons)):
            smap = simple_map.SimpleMap(jsons[i], ref_globals)
            self.complex_map.append(smap.get_route())

    def draw_map(self):
        for smap in self.complex_map:
            (x, y), (x_real, y_real), _ = smap
            plt.scatter(x, y, c="#3399ff")
            # plt.scatter(x_real, y_real, c='#3399ff')

        plt.axis('equal')
        plt.show()

    def export_pkl(self, folder='dataset'):
        cnt = 0

        for i in range(len(self.complex_map)):
            print("Exporting file", i)
            (x, y), (x_real, y_real), imgs = self.complex_map[i]

            for j in range(len(imgs)):
                cnt += 1
                path = os.path.join(folder, str(cnt))
                output_file = open(path, 'wb')
                obj = {"x_steer": x[j], "y_steer": y[j], "x_utm": x_real[j], "y_utm": y_real[j], "img": imgs[j]}
                pickle.dump(obj, output_file)
                output_file.close()

if __name__ == "__main__":
    jsons = [
        "./test_data/0ef581bf4a424ef1.json",
        "./test_data/0ba94a1ed2e0449c.json",
        "./test_data/1c820d64b4af4c85.json"
    ]
    cmap = ComplexMap(jsons)
    cmap.draw_map()
    cmap.export_pkl()
