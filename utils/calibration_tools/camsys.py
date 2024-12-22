import numpy as np
from pprint import pprint as print

class Camera:
    def __init__(self, data):
        self.intrinsics = np.eye(3)
        self.intrinsics[[0, 1, 0, 1], [0, 1, 2, 2]] = data["intrinsics"]

        # distortion
        self.distortion_coeffs = np.array(data["distortion_coeffs"])
        self.distortion_model = data["distortion_model"]
        self.resolution = data["resolution"]

        if "T_cn_cnm1" not in data:
            self.R = np.eye(3)
        else:
            self.R = np.array(data['T_cn_cnm1'])[:3,:3]

        self.K = self.intrinsics

    @property
    def num_pixels(self):
        return np.prod(self.resolution)

class CameraSystem:
    def __init__(self, data, fix_rotation=False):
        # load calibration
        T = np.array(data['cam1']['T_cn_cnm1'])
        T = np.linalg.inv(T)

        cam0 = Camera(data['cam0'])
        cam1 = Camera(data['cam1'])
        self.cam, self.event_cam = (cam0, cam1)  # if cam0.num_pixels > cam1.num_pixels else (cam1, cam0)
        # self.cam, self.event_cam = (cam0, cam1) if cam0.num_pixels > cam1.num_pixels else (cam1, cam0)

        if not fix_rotation:
            # camera chain parameters
            self.newK = self.event_cam.K

            # find new extrinsics
            self.t = T[:3,3]
            r3_cam0 = self.cam.R[:,2]

            r1 = self.t / np.linalg.norm(self.t)
            r2 = np.cross(r3_cam0, r1)
            r3 = np.cross(r1, r2)
            self.newR = np.stack([r1,r2,r3],-1)
            print("distance: %s" % (np.linalg.norm(self.t) * self.newK[0,0]))
        else:
            self.newR = self.cam.R
            self.newK = self.cam.K

        self.newres = tuple(self.event_cam.resolution)
        