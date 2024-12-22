import os
import cv2
import yaml
import argparse
import numpy as np

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
            self.R = np.array(data['T_cn_cnm1'])[:3, :3]

        self.K = self.intrinsics

    @property
    def num_pixels(self):
        return np.prod(self.resolution)

class CameraSystem:
    def __init__(self, data, fix_rotation=False):
        # load calibration
        T = np.array(data['cam1']['T_cn_cnm1'])
        # T = np.linalg.inv(T)

        cam0 = Camera(data['cam0'])
        cam1 = Camera(data['cam1'])
        self.cam, self.event_cam = (cam0, cam1) if cam0.num_pixels > cam1.num_pixels else (cam1, cam0)

        if not fix_rotation:
            # camera chain parameters
            self.newK = self.event_cam.K

            # find new extrinsics
            self.t = T[:3, 3]
            r3_cam0 = self.cam.R[:, 2]

            r1 = self.t / np.linalg.norm(self.t)
            r2 = np.cross(r3_cam0, r1)
            r3 = np.cross(r1, r2)
            self.newR = np.stack([r1, r2, r3], -1)
            print("distance: %s" % (np.linalg.norm(self.t) * self.newK[0, 0]))
        else:
            self.newR = self.cam.R
            self.newK = self.event_cam.K

        self.newres = tuple(self.event_cam.resolution)

    def getRemapping(self):
        # undistort image
        img_mapx, img_mapy = cv2.initUndistortRectifyMap(self.cam.K,
                                                         self.cam.distortion_coeffs,
                                                         None,
                                                         # evK @ T_depth_ev @ T_depth_ev^-1
                                                         self.newK  @ self.newR @ self.cam.R.T,
                                                         # evRes
                                                         self.newres,
                                                         cv2.CV_32FC1)

        ev_mapx, ev_mapy = cv2.initUndistortRectifyMap(self.event_cam.K,
                                                       self.event_cam.distortion_coeffs,
                                                       None,
                                                       # evK @ T_depth_ev @ I
                                                       self.newK  @ self.newR @ self.event_cam.R.T,
                                                       # evRes
                                                       self.newres,
                                                       cv2.CV_32FC1)

        W, H = self.event_cam.resolution
        coords = (np.stack(np.meshgrid(np.arange(W), np.arange(H)))
                  .reshape((2, -1)).T.reshape((-1, 1, 2)).astype("float32"))
        points = cv2.undistortPoints(coords, self.event_cam.K, self.event_cam.distortion_coeffs,
                                     None, self.newR @ self.event_cam.R.T, self.newK)
        inv_maps = points.reshape((H, W, 2))

        return {"img_mapx": img_mapx,
                "img_mapy": img_mapy,
                "ev_mapx": ev_mapx,
                "ev_mapy": ev_mapy,
                "inv_mapx": inv_maps[..., 0],
                "inv_mapy": inv_maps[..., 1]}

def remap_img(img, map, flip, rotate):
    if flip:
        img = img[:,::-1]
    mx, my = map
    img_remapped = cv2.remap(img, mx, my, cv2.INTER_CUBIC)
    if rotate:
        img_remapped = cv2.rotate(img_remapped, cv2.ROTATE_180)
    return img_remapped


def remap_events(events, map, rotate, shape):
    mx, my = map
    x, y = mx[events['y'], events['x']], my[events['y'], events['x']]
    p = events['p']
    t = events['t']

    target_width, target_height = shape

    if rotate:
        x = target_width - 1 - x
        y = target_height - 1 - y

    mask = (x >= 0) & (x <= target_width-1) & (y >= 0) & (y <= target_height-1)

    return {"x": x[mask], "y": y[mask], "t": t[mask], "p": p[mask]}

class Aligner():
    def __init__(self, calib_file):

        with open(calib_file, "r") as fh:
            cam_data = yaml.load(fh, Loader=yaml.SafeLoader)
        camsys = CameraSystem(cam_data, fix_rotation=True)
        distortion_maps = camsys.getRemapping()

        self.depth_map = (distortion_maps["img_mapx"], distortion_maps["img_mapy"])
        self.davis_map = (distortion_maps["ev_mapx"], distortion_maps["ev_mapy"])

    def align(self, depth=None, davis=None):

        out = {'depth': None, 'davis': None}
        if depth is not None:
            depth_remapped = remap_img(depth, self.depth_map, flip=False, rotate=False)
            out['depth'] = depth_remapped
        if davis is not None:
            davis_remapped = remap_img(davis, self.davis_map, flip=False, rotate=False)
            out['davis'] = davis_remapped

        return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib", default=None, type=str,
                        help="Path to the yaml file containing the cameras calibration")
    parser.add_argument("--davis_im", type=str, default="/capture_node/camera/image",
                        help="The ROS topic containing DAVIS event frames")
    parser.add_argument("--depth_im", type=str, default="/camera/depth/image_rect_raw",
                        help="The ROS topic containing depth images")
    parser.add_argument("--output_path", type=str, default=".")
    args = parser.parse_args()

    with open(args.calib, "r") as fh:
        cam_data = yaml.load(fh, Loader=yaml.SafeLoader)
    camsys = CameraSystem(cam_data, fix_rotation=True)
    distortion_maps = camsys.getRemapping()

    depth_map = (distortion_maps["img_mapx"], distortion_maps["img_mapy"])
    davis_map = (distortion_maps["ev_mapx"], distortion_maps["ev_mapy"])

    out_depth_path = os.path.join(args.output_path, "depth_remapped")
    out_davis_path = os.path.join(args.output_path, "davis_remapped")

    os.makedirs(out_depth_path, exist_ok=True)
    os.makedirs(out_davis_path, exist_ok=True)

    depth = cv2.imread(args.depth_im)
    depth_remapped = remap_img(depth, depth_map, flip=False, rotate=False)
    cv2.imwrite(os.path.join(out_depth_path, "depth.png"), depth_remapped)

    davis = cv2.imread(args.davis_im)
    davis_remapped = remap_img(davis, davis_map, flip=False, rotate=False)
    cv2.imwrite(os.path.join(out_davis_path, "davis.png"), davis_remapped)
