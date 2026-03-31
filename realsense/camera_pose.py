"""Load all images in given folder, detect chessboard and use PnPRansac to calculate camera pose 
    relative to the chessboard. Draw axes on the board. Save the camera pose in a yaml file.
    """
import cv2
import numpy as np
from pathlib import Path
import os


def camera_pose(images_path, calib_file,  file_name, widht, height, cell_size):
    # Load cam mat and dist_coeffs from calib_file
    cam_mat, dist_coeffs = None, None
    file = cv2.FileStorage(calib_file, cv2.FILE_STORAGE_READ)
    if file.isOpened():
        cam_mat = file.getNode("intrinsic").mat()
        dist_coeffs = file.getNode("distortion").mat()
    else:
        print("Failed to open calibration file.")
        return

    # Load all images in images_path
    p = Path(images_path)
    all_files = list(p.glob("*.jpg"))
    all_files += list(p.glob("*.png"))

    object_points = np.zeros((widht*height, 3), np.float32)
    object_points[:, :2] = np.mgrid[0:widht, 0:height].T.reshape(-1, 2) * cell_size

    all_rvec = []
    all_tvec = []
    valid_images = []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    cv2.namedWindow('RealSense', cv2.WINDOW_KEEPRATIO)
    for image in all_files:
        img = cv2.imread(image)
        undistorted = cv2.undistort(img, cam_mat, dist_coeffs, None, cam_mat)
        gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (widht, height), None)
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            ret, rvec, tvec, _ = cv2.solvePnPRansac(object_points, corners2, cam_mat, None)
            if ret:
                all_rvec.append(rvec)
                all_tvec.append(tvec)
                valid_images.append(image)

            cv2.drawChessboardCorners(undistorted, (widht, height), corners2, ret)
            txt = str(image).split("/")[-1]
            cv2.putText(undistorted, txt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.imshow('RealSense', undistorted)
            cv2.waitKey(1)

    cv2.destroyAllWindows()

    all_rvec = np.hstack(all_rvec)
    all_tvec = np.hstack(all_tvec)

    rvec = np.mean(all_rvec, axis=1).reshape(3, 1)
    tvec = np.mean(all_tvec, axis=1).reshape(3, 1)
    cv2.namedWindow('Undistorted', cv2.WINDOW_KEEPRATIO)
    rot_mat, _ = cv2.Rodrigues(rvec)

    for i, image in enumerate(valid_images):
        img = cv2.imread(image)
        undistorted = cv2.undistort(img, cam_mat, dist_coeffs, None, cam_mat)
        rot_mat, _ = cv2.Rodrigues(all_rvec[:, i])
        undistorted = cv2.drawFrameAxes(undistorted, cam_mat, None, all_rvec[:, i], all_tvec[:, i], cell_size * 3)
        undistorted = cv2.drawChessboardCorners(undistorted, (widht, height), corners2, ret)
        cv2.imshow('Undistorted', undistorted)
        cv2.waitKey(250)

    cv2.destroyAllWindows()

    file = cv2.FileStorage(file_name, cv2.FILE_STORAGE_WRITE)
    file.write("distortion", dist_coeffs)
    file.write("intrinsic", cam_mat)
    file.write("Rot", rot_mat)
    file.write("Trans", tvec)
    file.release()


if __name__ == "__main__":
    ids = ['104122061649', '950122060411']
    # ids = ['950122061707']
    dir_path = os.path.dirname(os.path.realpath(__file__))

    for id in ids:
        images_path = dir_path + f"/data/pose/images/{id}"
        w, h = 8, 5
        cell_size = 30

        output_calib_file = dir_path + f"/output/calib/{id}/calib.yaml"
        camera_pose(images_path, output_calib_file, output_calib_file, int(w), int(h), cell_size)
