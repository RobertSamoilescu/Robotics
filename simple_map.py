import json
import cv2
import numpy as np
import steering

class Convertor(object):
    @staticmethod
    def kmperh2mpers(speed):
        return speed / 3.6

class SimpleMap():
    def __init__(self, json, ref_globals=None):
        self.json = json
        self.frame_index = 0

        self.deasting = 0
        self.dnorthing = 0

        self._read_json()
        self._set_transformation(ref_globals)

    def _set_transformation(self, ref_globals):
        alpha = np.deg2rad(self.orientation)
        self.T = np.array([
            [np.cos(alpha), np.sin(alpha), 0],
            [-np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 1]
            ])

        if ref_globals:
            self.dnorthing = self.northing - ref_globals[0]
            self.deasting = self.easting - ref_globals[1]

    def _read_json(self):
        # get data from json
        with open(self.json) as f:
            self.data = json.load(f)

        # get cameras
        self.center_camera = self.data['cameras'][0]
        self.left_camera = self.data['cameras'][1]
        self.right_camera = self.data['cameras'][2]

        # open videos
        self.center_capture = cv2.VideoCapture(self.center_camera['video_path'])
        self.left_capture = cv2.VideoCapture(self.left_camera['video_path'])
        self.right_camera = cv2.VideoCapture(self.right_camera['video_path'])
        
        # read locations list
        self.locations = self.data['locations']
        self.easting = self.locations[0]['easting']
        self.northing = self.locations[0]['northing']
        self.orientation = self.locations[1]['course']

    def _get_position(self, course, speed, dt, eps=1e-12):
        sgn = 1 if course >= 0 else -1
        dist = sgn * speed * dt
        rad_course = np.deg2rad(course)
        R = dist / (abs(rad_course) + eps)
        position = np.array([R * (1 - np.cos(rad_course)), R * np.sin(rad_course), 1])
        return position

    def _get_relative_course(self, prev_course, crt_course):
        a = crt_course - prev_course
        a = (a + 180) % 360 - 180
        return a

    def _get_closest_location(self, tp):
        return min(self.locations, key=lambda x:abs(x['timestamp']-tp))

    def _get_rotation_matrix(self, course):
        rad_course = -np.deg2rad(course)
        R = np.array([
            [np.cos(rad_course), -np.sin(rad_course), 0], 
            [np.sin(rad_course), np.cos(rad_course), 0],
            [0, 0, 1]
            ])
        return R

    def _get_translation_matrix(self, position):
        T = np.eye(3)
        T[0, 2] = position[0]
        T[1, 2] = position[1]
        return T

    def _next_image_position(self):
        """
        :param predicted_course: predicted course by nn in degrees
        :return: image and coresponding position [x, y, 1] 
        """
        ret, frame = self.center_capture.read()
        dt = 1. / self.center_capture.get(cv2.CAP_PROP_FPS)

        if not ret:
            return None, np.array([])

        # for the first frame return
        if self.frame_index == 0:
            self.prev_course = self.locations[self.frame_index]['course']
            self.frame_index += 1
            return frame, np.array([0, 0, 1])

        # read course and speed for previous frame
        location = self._get_closest_location(1000 * dt * (self.frame_index - 1) + self.locations[0]['timestamp'])
        course = location['course']
        speed = location['speed']

        # compute relative course and save current course
        rel_course = self._get_relative_course(self.prev_course, course)
        self.prev_course = course

        # compute position from course, speed, dt
        position = self._get_position(rel_course, speed, dt)
        ret = (frame, np.dot(self.T, position))

        # increase the frame index
        self.frame_index += 1

        # update T matrix
        R = self._get_rotation_matrix(rel_course)
        T = self._get_translation_matrix(position)
        self.T = np.matmul(self.T, np.matmul(T, R))

        return ret 

    def get_route(self):
        frame, position = self._next_image_position()
        x = []; y = []

        while position.size > 0:
            x.append(position[0] + self.deasting)
            y.append(position[1] + self.dnorthing)
            frame, position = self._next_image_position()
    
            ## VISUAL        
            # import matplotlib.pyplot as plt
            # plt.clf()
            # plt.scatter(x, y)
            # plt.draw()
            # plt.pause(0.001)

            # import cv2
            # cv2.imshow("FRAME", frame)
            # cv2.waitKey(33)


        return x, y

if __name__ == "__main__":
    smap = SimpleMap("./test_data/0ef581bf4a424ef1.json")
    x, y= smap.get_route()

    # abs_x = [abs(elem) for elem in x]
    # abs_y = [abs(elem) for elem in y]
    # print(max(abs_x), max(abs_y))

    import matplotlib.pyplot as plt
    plt.axis('equal')
    plt.scatter(x, y)
    plt.show()
