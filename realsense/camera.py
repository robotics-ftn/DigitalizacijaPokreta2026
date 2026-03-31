import numpy as np
import cv2


class Camera:

    def __init__(self, id: str, K: np.ndarray, D: np.ndarray, rot: np.ndarray, tvec: np.ndarray):
        """Camera object class.

        Args:
            id (str): name of camera
            K (np.ndarray): intrinsic camera params
            D (np.ndarray): distortion coefs
            rot (np.ndarray): rotation from world to camera coordinates
            tvec (np.ndarray): translation from world to camera coordinates (object translation in camera frame)
        """
        self.id = id
        self.K = K
        self.D = D
        self.tvec = np.array(tvec)           # object translation in camera frame
        self.R = rot                         # Rotation from world to camera coordinates
        self.T = self.get_camera_position()  # in world coordinates

    def get_camera_position(self):
        """ Camera position in world coordinates. """
        camera_position = -self.R.T @ self.tvec.ravel()  # ravel() forces (3,)
        return camera_position

    def get_ray(self, pixel_uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u_h = np.array([pixel_uv[0], pixel_uv[1], 1.0])
        u_cam = np.linalg.inv(self.K) @ u_h
        u_world = self.R.T @ u_cam
        ray_world = u_world / np.linalg.norm(u_world)
        return self.T, ray_world  # self.T already fixed above

    def reproject_point3d_to_pixel(self, point3d: np.ndarray):
        # q = K * R * (Q - T)
        # where T is camera position in world coords
        # and R is rotation from world to camera coords
        pixel_point = self.K @ self.R @ (point3d - self.T)
        uv_pixels = pixel_point[:2]  # [u, v, s] = s * [u', v', 1]
        uv_pixels = uv_pixels / pixel_point[2]  # Normalize by depth
        return uv_pixels
