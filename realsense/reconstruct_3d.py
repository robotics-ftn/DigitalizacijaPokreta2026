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


def reconstruct_points_analytical(cameras: list, pixel_coords_list: np.ndarray) -> np.ndarray:
    """
    Fully vectorized 3D reconstruction via closed-form normal equations.

    Args:
        cameras: list of Camera objects (N cameras)
        pixel_coords_list: (P, N, 2) — P points, N cameras, 2 pixel coords

    Returns:
        points_3d: (P, 3)
    """
    N = len(cameras)
    P = pixel_coords_list.shape[0]

    T_all = np.array([cam.T for cam in cameras])       # (N, 3)

    # Precompute ray directions: (N, P, 3)
    u_all = np.zeros((N, P, 3))
    for i, cam in enumerate(cameras):
        pixels = pixel_coords_list[:, i, :]            # (P, 2)
        q_h = np.hstack([pixels, np.ones((P, 1))])  # (P, 3)
        u_cam = (np.linalg.inv(cam.K) @ q_h.T).T     # (P, 3)
        u_world = (cam.R.T @ u_cam.T).T                # (P, 3)
        norms = np.linalg.norm(u_world, axis=1, keepdims=True)
        u_all[i] = u_world / norms                     # (P, 3)

    # Build LHS (P, 3, 3) and RHS (P, 3)
    LHS = np.zeros((P, 3, 3))
    RHS = np.zeros((P, 3))

    for i in range(N):
        u = u_all[i]                                 # (P, 3)
        uuT = u[:, :, None] * u[:, None, :]            # (P, 3, 3)
        proj = np.eye(3)[None] - uuT                   # (P, 3, 3)

        LHS += proj                                                  # (P, 3, 3)
        RHS += (proj @ T_all[i]).reshape(P, 3)              # (P, 3)
    print(f"LHS shape: {LHS.shape}")   # (40, 3, 3)
    print(f"RHS shape: {RHS.shape}")   # (40, 3)
    # Batched solve: LHS[p] @ Q[p] = RHS[p] for all p
    points_3d = np.linalg.solve(LHS, RHS[:, :, None]).squeeze(-1)  # (P, 3)
    return points_3d


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
        img_files.sort()
        img = cv2.imread(str(img_files[11]))

        img_undistort = cv2.undistort(img, cam.K, cam.D)

        gray = cv2.cvtColor(img_undistort, cv2.COLOR_BGR2GRAY)
        pattern_size = (8, 5)  # Adjust based on your chessboard pattern
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), criteria)
            corners2 = corners2.reshape(-1, 2)  # (N, 2)
            # cv2.namedWindow(f"Camera Corners", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            # cv2.drawChessboardCorners(img_undistort, pattern_size, corners2, ret)
            # cv2.imshow(f"Camera Corners", img_undistort)
            # cv2.waitKey(1)
            detected_points.append(corners2)
        else:
            print(f"Chessboard corners not found in image for camera {cam_id}.")

    detected_points = np.array(detected_points)  # (num_cameras, num_points, 2)

    # repack in pairs [Points, Cameras, 2]
    pixel_coords_list = detected_points.transpose(1, 0, 2)  # (num_points, num_cameras, 2)

    # Reconstruct 3D points from multiple camera views
    points_3d = reconstruct_all_points(cameras, pixel_coords_list)
    # Print numpy matrix with 2 decimal places
    np.set_printoptions(precision=2, suppress=True)
    # print(f"Reconstructed 3D points, in [mm]:\n{points_3d}")

    # Reproject 3D points in camera frame, draw circles
    for cam in cameras:
        images_path = dir_path + f"/data/pose/images/{cam.id}"
        img_files = list(Path(images_path).glob("*.png"))
        img_files.sort()
        img = cv2.imread(str(img_files[11]))

        img_undistort = cv2.undistort(img, cam.K, cam.D)

        # q = M * R * (Q - T)
        cv2.namedWindow("Reproject points", cv2.WINDOW_KEEPRATIO)
        for point in points_3d:
            uv_pixels = cam.reproject_point3d_to_pixel(point)
            cv2.circle(img_undistort, (int(uv_pixels[0]), int(uv_pixels[1])), 5, (0, 0, 255), 2)

        cv2.imshow("Reproject points", img_undistort)
        cv2.waitKey(0)
