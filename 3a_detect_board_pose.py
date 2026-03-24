"""Detekcija poze table i iscrtavanje na nju
    """
from pathlib import Path
import cv2
import numpy as np


def get_board_pose(img, cam_mtx, dist_coefs, cols, rows, cell_size, visualize=True):

    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * cell_size

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

    if found:
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        ret, rvecs, tvecs = cv2.solvePnP(objp, corners_refined, camera_matrix, dist_coefs)

        preview = img.copy()
        if visualize and ret:
            cv2.drawFrameAxes(preview, camera_matrix, dist_coefs, rvecs, tvecs, 3 * cell_size)
            cv2.drawChessboardCorners(preview, (cols, rows), corners_refined, found)

        return preview, rvecs, tvecs
    return img, None, None


if __name__ == "__main__":

    camera_param_path = "output/calib/camera.npz"
    pose_save_path = "output/pose/camera.npz"

    camera_data = np.load(camera_param_path)
    camera_matrix = camera_data["camera_matrix"]
    dist_coeffs = camera_data["dist_coeffs"]

    camera = cv2.VideoCapture(0)
    rows = 5
    cols = 8
    cell_size = 30.0  # mm

    all_tvecs = []
    all_rvecs = []
    flg_store = False
    while True:

        ret, frame = camera.read()

        undistorted_image = cv2.undistort(frame, camera_matrix, dist_coeffs)
        img, rvec, tvec = get_board_pose(undistorted_image,
                                         camera_matrix,
                                         dist_coeffs,
                                         rows, cols, cell_size,
                                         True  # Visualize
                                         )
        cv2.imshow("Camera", img)

        if tvec is not None and rvec is not None:
            if flg_store:
                all_tvecs.append(tvec)
                all_rvecs.append(rvec)
            # print(f"Translation: {tvec}")

            # rot, _ = cv2.Rodrigues(rvec)
            # print(f"Rotation vector: \n{rvec}")
            # print(f"Rotation: \n{rot}")

        if cv2.waitKey(1) == 27:  # ESC
            if rvec is not None and tvec is not None:
                print(f"Translation: {tvec}")
                rot, _ = cv2.Rodrigues(rvec)
                print(f"Rotation vector: \n{rvec}")
                print(f"Rotation: \n{rot}")
            break

        if cv2.waitKey(1) == 32:  # SPACE
            print("STORING DATA")
            flg_store = True

    if len(all_tvecs) > 3:
        avg_tvec = np.mean(all_rvecs, axis=0)
        avg_rvec = np.mean(all_tvecs, axis=0)
        rot, _ = cv2.Rodrigues(avg_rvec)
        print(f"Saving to path: {pose_save_path}")
        np.savez(
            pose_save_path,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            tvec=avg_tvec,
            rvec=avg_rvec
        )

        print(f"Translation: {tvec}")
        print(f"Rotation vector: \n{rvec}")
        print(f"Rotation: \n{rot}")

    camera.release()
