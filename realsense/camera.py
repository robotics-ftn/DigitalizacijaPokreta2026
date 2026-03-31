import numpy as np
import cv2


class Camera:

    def __init__(self, id: str, K: np.ndarray, D: np.ndarray, rot: np.ndarray, tvec: np.ndarray):
        """Camera object class.

        Args:
            K (np.ndarray): _description_
            D (np.ndarray): _description_
            rvec (np.ndarray): _description_
            tvec (np.ndarray): _description_
        """
        self.id = id
        self.K = K
        self.D = D
        self.tvec = np.array(tvec)  # object translation in camera frame
        self.R = rot
        self.T = self.get_camera_position()

    def get_camera_position(self):
        camera_position = -self.R.T @ self.tvec.ravel()  # ravel() forces (3,)
        return camera_position

    def get_ray(self, pixel_uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u_h = np.array([pixel_uv[0], pixel_uv[1], 1.0])
        u_cam = np.linalg.inv(self.K) @ u_h
        u_world = self.R.T @ u_cam
        ray_world = u_world / np.linalg.norm(u_world)
        return self.T, ray_world  # self.T already fixed above
