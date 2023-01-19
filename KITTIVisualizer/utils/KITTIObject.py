import os
import numpy as np


def roty(t):
    """ Rotation about the y-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def read_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [KITTIObject(line) for line in lines]
    return objects

def get_label_objects(label_dir, idx):
		label_filename = os.path.join(label_dir, "%06d.txt" % (idx))
		return read_label(label_filename)

class KITTIObject(object):
    """ 3d object label """

    def __init__(self, label_file_line):
        data = label_file_line.split(" ")
        data[1:] = [float(x) for x in data[1:]]

        self.label_str = data[0] # 차, 보행자 등의 라벨 정보 문자열 'Car', 'Pedestrian', ...
        self.truncation = data[1] # 이미지상에서 물체가 잘려있는 정도 truncated pixel ratio [0..1]
        self.occlusion = int(data[2]) # 폐섹 수준 0=전혀 가려지지 않음, 1=부분적으로 가려짐, 2=완전히 가려짐, 3=알 수 없음
        self.alpha = data[3]  # 물체의 관측각(카메라 입장에서 물체가 위치한 각도) [-pi..pi]

        self.xmin = data[4]  # left of 2d bbox
        self.ymin = data[5]  # top of 2d bbox
        self.xmax = data[6]  # right of 2d bbox
        self.ymax = data[7]  # bottom of 2d bbox
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        self.h = data[8]  # height of 3d bbox
        self.w = data[9]  # width of 3d bbox
        self.l = data[10]  # length of 3d bbox (in meters)
        self.t = (data[11], data[12], data[13])  # location (x,y,z) of 3d bbox in camera coord.
        self.ry = data[14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

    def estimate_diffculty(self):
        """ Function that estimate difficulty to detect the object as defined in kitti website"""
        # height of the bounding box
        bb_height = np.abs(self.xmax - self.xmin)

        if bb_height >= 40 and self.occlusion == 0 and self.truncation <= 0.15:
            return "Easy" # -> 폐색이 없고, 가려짐이 15% 이하며 크기가 40 pixel 이상
        elif bb_height >= 25 and self.occlusion in [0, 1] and self.truncation <= 0.30:
            return "Moderate" # -> 폐색이 없거나 부분적인 폐색, 가려짐이 30% 이하며 크기가 25 pixel 이상
        elif bb_height >= 25 and self.occlusion in [0, 1, 2] and self.truncation <= 0.50:
            return "Hard" # -> 폐색이 없거나 부분적인 폐색 혹은 완전한 폐색, 가려짐이 50% 이하며 크기가 25 pixel 이상
        else:
            return "Unknown" # 그 외의 경우

    def print_object(self):
        print(f"{self.__class__.__name__}")
        print(f" - Difficulty:{self.estimate_diffculty()}")
        print(f" - Label:{self.label_str} | Truncation:{self.truncation} | Occlusion:{self.occlusion} | Alpha:{self.alpha}")
        print(f" - 2DBbox @ Pos:({self.xmin},{self.ymin}),({self.xmax},{self.ymax})")
        print(f" - 3DBbox @ Pos:({self.t[0]},{self.t[1]},{self.t[2]}) | Yaw:{self.ry} | HWL:({self.h},{self.w},{self.l})")