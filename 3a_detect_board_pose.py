"""Detekcija poze table i iscrtavanje na nju
    """
from pathlib import Path
import cv2
import numpy as np


def get_board_pose(img, cam_mtx, dist_coefs, cols, rows, cell_size, visualize=True):

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]   # (w, h)

    found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

    if found:
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        preview = img.copy()
        if visualize:
            cv2.drawChessboardCorners(preview, (cols, rows), corners_refined, found)

        return preview
    return img


if __name__ == "__main__":

    camera_param_path = "output/calib/camera.npz"

    camera_data = np.load(camera_param_path)
    camera_matrix = camera_data["camera_matrix"]
    dist_coeffs = camera_data["dist_coeffs"]

    camera = cv2.VideoCapture(0)
    rows = 5
    cols = 8
    cell_size = 30.0  # mm
    while True:

        ret, frame = camera.read()

        undistorted_image = cv2.undistort(frame, camera_matrix, dist_coeffs)
        img = get_board_pose(undistorted_image,
                             camera_matrix,
                             dist_coeffs,
                             rows, cols, cell_size,
                             True  # Visualize
                             )
        cv2.imshow("Camera", img)

        if cv2.waitKey(1) == 27:  # ESC
            break

    camera.release()
