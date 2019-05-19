import json
import cv2
import numpy as np
import steering
import re


class Convertor(object):
    @staticmethod
    def kmperh2mpers(speed):
        return speed / 3.6


class SimpleMap(object):
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

        suffix_len = len(".json")

        # get center camera
        video_path = self.json[:-suffix_len] + "-0.mov"
        self.center_capture = cv2.VideoCapture(video_path)

        # get left camera
        video_path = self.json[:-suffix_len] + "-1.mov"
        self.left_capture = cv2.VideoCapture(video_path)

        # get right camera
        video_path = self.json[:-suffix_len] + "-2.mov"
        self.right_capture = cv2.VideoCapture(video_path)

        # read locations list
        self.locations = self.data['locations']
        self.easting = self.locations[0]['easting']
        self.northing = self.locations[0]['northing']
        self.orientation = self.locations[1]['course']

    @staticmethod
    def get_position(course, speed, dt, eps=1e-12):
        sgn = 1 if course >= 0 else -1
        dist = sgn * speed * dt
        rad_course = np.deg2rad(course)
        R = dist / (abs(rad_course) + eps)
        position = np.array([R * (1 - np.cos(rad_course)), R * np.sin(rad_course), 1])
        return position

    @staticmethod
    def get_relative_course(prev_course, crt_course):
        a = crt_course - prev_course
        a = (a + 180) % 360 - 180
        return a

    def get_closest_location(self, tp):
        return min(self.locations, key=lambda x:abs(x['timestamp']-tp))

    @staticmethod
    def get_rotation_matrix(course):
        rad_course = -np.deg2rad(course)
        R = np.array([
            [np.cos(rad_course), -np.sin(rad_course), 0], 
            [np.sin(rad_course), np.cos(rad_course), 0],
            [0, 0, 1]
        ])
        return R

    @staticmethod
    def get_translation_matrix(position):
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
        left_ret, left_frame = self.left_capture.read()
        right_ret, right_frame = self.right_capture.read()

        dt = 1. / self.center_capture.get(cv2.CAP_PROP_FPS)

        if not ret or not left_ret or not right_ret:
            return None, np.array([]), np.array([])

        # for the first frame return
        if self.frame_index == 0:
            self.prev_course = self.locations[0]['course']
            self.frame_index += 1
            return (frame, left_frame, right_frame), np.array([0, 0]), np.array([0, 0])

        # read course and speed for previous frame
        location = self.get_closest_location(1000 * dt * (self.frame_index - 1) + self.locations[0]['timestamp'])
        course = location['course']
        speed = location['speed']

        # compute relative course and save current course
        rel_course = self.get_relative_course(self.prev_course, course)
        self.prev_course = course

        # compute position from course, speed, dt
        position = SimpleMap.get_position(rel_course, speed, dt)
        real_position = np.array([
            location['easting'] - self.locations[0]['easting'], 
            location['northing'] - self.locations[0]['northing']
        ])
        ret = ((frame, left_frame, right_frame), np.dot(self.T, position)[:-1], real_position)

        # increase the frame index
        self.frame_index += 1

        # update T matrix
        R = SimpleMap.get_rotation_matrix(rel_course)
        T = SimpleMap.get_translation_matrix(position)
        self.T = np.matmul(self.T, np.matmul(T, R))

        return ret 

    def get_route(self, verbose=False):
        frame, position, real_position = self._next_image_position()
        x = []; y = []
        x_real = []; y_real = []
        frames = []

        while position.size > 0:
            # visual
            if verbose:
                import matplotlib.pyplot as plt
                plt.clf()
                plt.scatter(x_real, y_real)
                plt.axis('equal')
                plt.draw()
                plt.pause(0.001)

                import cv2
                cv2.imshow("FRAME", frame[2])
                cv2.waitKey(1)

            x.append(position[0] + self.deasting)
            y.append(position[1] + self.dnorthing)
            x_real.append(real_position[0] + self.deasting)
            y_real.append(real_position[1] + self.dnorthing)
            frames.append(frame)

            # get next data
            frame, position, real_position = self._next_image_position()
    
        return (x, y), (x_real, y_real), frames

if __name__ == "__main__":
    smap = SimpleMap("./raw_data/1c820d64b4af4c85.json")
    (x, y), _, _= smap.get_route()

    # abs_x = [abs(elem) for elem in x]
    # abs_y = [abs(elem) for elem in y]
    # print(max(abs_x), max(abs_y))

    import matplotlib.pyplot as plt
    plt.scatter(x, y)
    plt.axis('equal')
    plt.show()
