import cv2
import os
import glob
import numpy as np


def intrinsic_calibrate(images_dir, rows, cols, cell_size, output_file):

    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")) +
                         glob.glob(os.path.join(images_dir, "*.jpg")))

    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * cell_size

    obj_points = []   # 3-D points in world space
    img_points = []   # 2-D points in image space
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for image in image_paths:
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]   # (w, h)

        found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

        if found:
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners_refined)

            preview = img.copy()
            cv2.drawChessboardCorners(preview, (cols, rows), corners_refined, found)
            cv2.imshow("Detected corners", preview)
            cv2.waitKey(100)
            print(f"  OK  {os.path.basename(image)}")

    cv2.destroyAllWindows()

    print(f"Valid images: {len(obj_points)}")
    print(f"image_size (W, H): {image_size}")
    print(f"objp shape: {objp.shape}")
    print(f"img_points[0] shape: {img_points[0].shape}")

    print(f"\nCalibrating with {len(obj_points)} images ...")
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )

    print(f"\nRMS reprojection error : {rms:.4f} px")
    print("\nCamera matrix:\n", camera_matrix)
    print("\nDistortion coefficients:\n", dist_coeffs.ravel())

    np.savez(output_file, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, rms=rms)
    print(f"\nSaved to '{output_file}'")

    return camera_matrix, dist_coeffs, rms, image_size


if __name__ == "__main__":

    images_root_dir = "output/calib/images"
    output_file = "output/calib/camera.npz"
    intrinsic_calibrate(images_root_dir, 5, 8, 30.0, output_file)
