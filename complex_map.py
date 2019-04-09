import simple_map
import matplotlib.pyplot as plt
   

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
            x, y = smap
            plt.scatter(x, y, c="#3399ff")
        plt.show()

if __name__ == "__main__":
    jsons = [
        "./test_data/0ba94a1ed2e0449c.json",
       	"./test_data/0ef581bf4a424ef1.json",
        "./test_data/1c820d64b4af4c85.json"
    ]
    cmap = ComplexMap(jsons)
    cmap.draw_map()