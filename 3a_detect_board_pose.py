"""Detekcija poze table i iscrtavanje na nju
    """
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
    image_save_path_root = "output/pose/images"

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
    counter = 0
    while True:

        ret, frame = camera.read()

        undistorted_image = cv2.undistort(frame, camera_matrix, dist_coeffs)
        img, rvec, tvec = get_board_pose(undistorted_image,
                                         camera_matrix,
                                         dist_coeffs,
                                         rows, cols, cell_size,
                                         True  # Visualize
                                         )

        key = cv2.waitKey(1)
        display = img.copy()
        cv2.putText(display, f"Captured: {counter} Store: {flg_store}| SPACE=save  ESC=quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if tvec is not None and rvec is not None:
            if flg_store:
                all_tvecs.append(tvec)
                all_rvecs.append(rvec)
                counter += 1
                flg_store = False
                save_path = f"{image_save_path_root}/pose_{counter:04d}.png"
                print(f"Saving image to path: {save_path}")
                cv2.imwrite(save_path, img)

        if key == 27:  # ESC
            if all_tvecs and all_rvecs:
                print(f"Translation: {all_tvecs[-1]}")
                rot, _ = cv2.Rodrigues(all_rvecs[-1])

                print(f"Rotation vector: \n{all_rvecs[-1]}")
                print(f"Rotation: \n{rot}")
            break

        cv2.imshow("Camera", img)

    if len(all_tvecs) > 5 and len(all_rvecs) > 5:
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

        print(f"Average translation: \n{tvec}")
        print(f"Average rotation vector: \n{rvec}")
        print(f"Rotation matrix: \n{rot}")

    camera.release()
