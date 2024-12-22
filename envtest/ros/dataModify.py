import numpy as np
import cv2
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt


class Gimbal():
    def __init__(self, fov, width, height):
        # f = ( image.height / 2.0 ) / tan( (M_PI * FOV/180.0 )/2.0 )
        # [ [fx, 0, image.width/2], [0, fy, image.height/2], [0, 0, 1] ]
        self.fc = (height / 2.0) / np.arctan(np.pi*(fov/360))
        self.K = np.array(
            [[self.fc, 0, width/2], [0, self.fc, height/2], [0, 0, 1]])

        self.maxRow = height
        self.maxCol = width

    def undistort(self, tmp):
        """
        Under construction
        @TODO(Dhruv) Parameterize
        f = ( image.height / 2.0 ) / tan( (pi* FOV/180.0 )/2.0 )
            [ [fx, 0, image.width/2], [0, fy, image.height/2], [0, 0, 1] ].
        """

        fc = (240 / 2.0) / np.arctan(np.pi*(180 * 120/180.0)/(2.0*180))
        K = np.array([[fc, 0, 320/2], [0, fc, 240/2], [0, 0, 1]])
        Knew = K.copy()
        Knew[0, 0] *= 2.2
        Knew[1, 1] *= 2.2
        return cv2.fisheye.undistortImage(tmp, K, D=np.array([4.5, 0., 0., 0.]), Knew=Knew), Knew

    def rotatePt(self, Pt, K, R):
        Pt = Pt.reshape(-1, 1)
        Rdc = np.array([[0, 0, 1],
                        [1, 0, 0],
                        [0, 1, 0]])
        newCenter = K@np.linalg.inv(Rdc)@np.linalg.inv(
            R)@Rdc@np.linalg.inv(K)@Pt
        newCenter = newCenter/newCenter[-1]
        return newCenter

    def constrainCorner(self, corner):
        corner[0] = self.constrain(corner[0], self.maxRow, 0)
        corner[1] = self.constrain(corner[1], self.maxCol, 0)
        return corner

    def constrain(self, val, high, low):
        if val > high:
            return high
        if val < low:
            return low
        return val

    def rotz(self, theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))

    def do_gimbal(self, img, quat, cropWidth, cropHeight, do_clip=True):

        # saveCropHeight =
        q = quat.flatten()

        rot = Rotation.from_quat([q[1], q[2], q[3], q[0]])
        rotations = rot.as_rotvec()

        rotations[1] = self.constrain(rotations[1], np.pi/12, -np.pi/15)
        rotations[0] = self.constrain(rotations[0], np.pi/6, -np.pi/6)
        rotations[0] *= -1
        rotations[2] = self.constrain(rotations[2], np.pi/6, -np.pi/6)
        R = Rotation.from_rotvec(rotations).as_matrix()

        # Image size is height,width -> 120,160
        newCenter = self.rotatePt(
            np.array([self.maxRow//2, self.maxCol//2, 1]), self.K, R)
        zeroCenter = self.rotatePt(
            np.array([self.maxRow//2, self.maxCol//2, 1]), self.K, np.eye(3))

        pts = np.float32([[-(cropHeight-self.maxRow)//2, -(cropWidth-self.maxCol)//2],
                          [(cropHeight+self.maxRow)//2, -
                           (cropWidth-self.maxCol)//2],
                          [-(cropHeight-self.maxRow)//2,
                           (cropWidth+self.maxCol)//2],
                          [(cropHeight+self.maxRow)//2, (cropWidth+self.maxCol)//2]])
        homoPts = np.column_stack((np.array(pts), np.ones((4, 1))))
        corner1 = self.constrainCorner(self.rotatePt(homoPts[0], self.K, R))
        corner2 = self.constrainCorner(self.rotatePt(homoPts[1], self.K, R))
        corner3 = self.constrainCorner(self.rotatePt(homoPts[2], self.K, R))
        corner4 = self.constrainCorner(self.rotatePt(homoPts[3], self.K, R))

        pts2 = np.float32([[corner1[0], corner1[1]],
                           [corner2[0], corner2[1]],
                           [corner3[0], corner3[1]],
                           [corner4[0], corner4[1]]])

        M = cv2.getPerspectiveTransform(pts, pts2)
        if np.linalg.det(M) == 0:
            print(rotations)

        dst = cv2.warpPerspective(img, M, (int(self.maxCol), int(self.maxRow)), flags=cv2.INTER_LINEAR)
        newCenter = (np.array([self.maxRow//2, self.maxCol//2, 1])).reshape(-1, 1)
        # zeroCenter = np.linalg.inv(M)@zeroCenter

        img3 = dst[int(newCenter[0])-cropHeight//2:int(newCenter[0])+cropHeight //
                   2, int(newCenter[1])-cropWidth//2:int(newCenter[1])+cropWidth//2]
        if do_clip:
            img3 = np.clip(np.array(img3),0,1)
        # , newCenter, zeroCenter, corner1,corner2,corner3,corner4
        return np.array(img3), pts2.squeeze()
