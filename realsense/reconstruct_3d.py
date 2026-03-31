import numpy as np
import cv2
from camera import Camera


def reconstruct_points(cameras: list[Camera], pixel_coords: list[np.ndarray]) -> np.ndarray:
    """
    Reconstruct 3D point from multiple camera views.

    Args:
        cameras (list[Camera]): List of Camera objects.
        pixel_coords (list[np.ndarray]): List of pixel coordinates corresponding to each camera.

    Returns:
        np.ndarray: Reconstructed 3D point in world coordinates.
    """
    n = len(cameras)
    assert n >= 2, 'Need at least 2 cameras for triangulation.'
    assert len(pixel_coords) == n, 'Number of pixel coordinates must match number of cameras.'

    A = np.zeros((3*n, 3 + n))
    b = np.zeros(3*n)

    for i, (cam, pixel) in enumerate(zip(cameras, pixel_coords)):
        T_i, u_i = cam.get_ray(pixel)
        row = i * 3
        col_w = 3 + i

        A[row:row+3, :3] = np.eye(3)  # Identity for world point
        A[row:row+3, col_w] = -u_i      # Ray direction
        b[row:row+3] = T_i              # Camera position

    # Solve Ax = b
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return x[:3]  # Return the world point


def reconstruct_all_points(cameras: list[Camera], pixel_coords_list: list[list[np.ndarray]]) -> np.ndarray:
    """Reconstruct multiple 3D points from multiple camera views.

    Args:
        cameras (list[Camera]): List of Camera objects.
        pixel_coords_list (list[list[np.ndarray]]): List of lists of pixel coordinates for each point.

    Returns:
        np.ndarray: Reconstructed 3D points in world coordinates.
    """
    points_3d = []
    for pixel_coords in pixel_coords_list:
        point_3d = reconstruct_points(cameras, pixel_coords)
        points_3d.append(point_3d)
    return np.array(points_3d)


if __name__ == "__main__":
    import os
    from pathlib import Path

    ids = ['104122061649', '950122060411']
    # ids = ['950122061707']
    dir_path = os.path.dirname(os.path.realpath(__file__))

    cameras = []
    for cam_id in ids:
        calib_file = dir_path + f"/output/calib/{cam_id}/calib.yaml"

        file = cv2.FileStorage(calib_file, cv2.FILE_STORAGE_READ)
        intrinsics = file.getNode("intrinsic").mat()
        distortion = file.getNode("distortion").mat()
        rot = file.getNode("Rot").mat()
        trans = file.getNode("Trans").mat()
        file.release()
        cam = Camera(cam_id, intrinsics, distortion, rot, trans)
        cameras.append(cam)

    detected_points = []
    for cam in cameras:

        # Load random image, extract chessboard corners, and reconstruct 3D points
        images_path = dir_path + f"/data/pose/images/{cam.id}"
        img_files = list(Path(images_path).glob("*.png"))
        img = cv2.imread(str(img_files[1]))

        img_undistort = cv2.undistort(img, cam.K, cam.D)

        gray = cv2.cvtColor(img_undistort, cv2.COLOR_BGR2GRAY)
        pattern_size = (8, 5)  # Adjust based on your chessboard pattern
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            corners2 = corners2.reshape(-1, 2)  # (N, 2)
            zero_corner = corners2[0]  # (2,)
            cv2.namedWindow(f"Camera {cam.id} Corners", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.circle(img_undistort, (int(corners2[0, 0]), int(corners2[0, 1])), 5, (0, 255, 0), -1)
            cv2.imshow(f"Camera {cam.id} Corners", img_undistort)
            cv2.waitKey(0)
            detected_points.append([corners2[1]])
        else:
            print(f"Chessboard corners not found in image for camera {cam_id}.")

    detected_points = np.array(detected_points)  # (num_cameras, num_points, 2)

    # repack in pairs [cam_1_uv, cam_2_u2] for each point
    num_points = detected_points.shape[1]
    pixel_coords_list = []
    for i in range(num_points):
        pixel_coords = [detected_points[j, i] for j in range(len(cameras))]
        pixel_coords_list.append(pixel_coords)

    # Reconstruct 3D points from multiple camera views
    points_3d = reconstruct_all_points(cameras, pixel_coords_list)
    # Print pretty numpy matrix with 2 decimal places
    np.set_printoptions(precision=2, suppress=True)
    print(f"Reconstructed 3D points, in [mm]:\n{points_3d}")
