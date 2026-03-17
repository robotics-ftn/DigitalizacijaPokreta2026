"""
Loads intrinsic parameters from calibration and displays real-time camera feed
with undistortion applied. Press ESC to quit.
"""

import cv2
import numpy as np
import sys


def undistort_camera_feed(intrinsics_npz="camera_intrinsics.npz", camera_id=0):
    """
    Display real-time camera feed with undistortion applied.

    Args:
        intrinsics_npz (str): File containing intrinsic parameters.
        camera_id (int): Camera device ID (default 0 for main camera).
    """
    # Load intrinsics
    try:
        data = np.load(intrinsics_npz)
        camera_matrix = data["camera_matrix"]
        dist_coeffs = data["dist_coeffs"]
    except FileNotFoundError:
        print(f"Error: Intrinsics file '{intrinsics_npz}' not found.")
        return

    print(f"Loaded intrinsics from '{intrinsics_npz}'")
    print("Camera matrix:\n", camera_matrix)
    print("\nDistortion coefficients:\n", dist_coeffs.ravel())

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {camera_id}")
        return

    # Get frame dimensions to create undistortion map
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Pre-compute undistortion map for efficiency
    map_x, map_y = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, camera_matrix,
        (frame_width, frame_height), cv2.CV_32F
    )

    print(f"\nCamera resolution: {frame_width}x{frame_height}")
    print("Press ESC to quit, SPACE to save frame, 'U' to toggle undistortion.\n")

    frame_count = 0
    undistort_enabled = True

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Apply undistortion
        if undistort_enabled:
            undistorted = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
            display = undistorted
            status_text = "UNDISTORT: ON"
        else:
            display = frame
            status_text = "UNDISTORT: OFF"

        # Display info
        cv2.putText(display, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, "ESC=quit  SPACE=save  U=toggle", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(display, f"Frame: {frame_count}", (10, frame_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Undistort Camera Feed", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE - save frame
            filename = f"undistorted_frame_{frame_count:04d}.png"
            cv2.imwrite(filename, display)
            print(f"  Saved {filename}")
        elif key == ord('u') or key == ord('U'):  # Toggle undistortion
            undistort_enabled = not undistort_enabled
            print(f"Undistortion {'enabled' if undistort_enabled else 'disabled'}")

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Processed {frame_count} frames.")


def undistort_image(image_path, intrinsics_npz="camera_intrinsics.npz",
                    output_path=None):
    """
    Undistort a single image.

    Args:
        image_path (str): Path to the image to undistort.
        intrinsics_npz (str): File containing intrinsic parameters.
        output_path (str): Output path for undistorted image (optional).

    Returns:
        np.ndarray: Undistorted image.
    """
    # Load intrinsics
    try:
        data = np.load(intrinsics_npz)
        camera_matrix = data["camera_matrix"]
        dist_coeffs = data["dist_coeffs"]
    except FileNotFoundError:
        print(f"Error: Intrinsics file '{intrinsics_npz}' not found.")
        return None

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot load image '{image_path}'")
        return None

    # Undistort
    undistorted = cv2.undistort(img, camera_matrix, dist_coeffs)

    # Save if output path provided
    if output_path:
        cv2.imwrite(output_path, undistorted)
        print(f"Saved undistorted image to '{output_path}'")

    return undistorted


if __name__ == "__main__":
    camera_params = "output/calib/camera.npz"
    camera_id = 0
    undistort_camera_feed(camera_params, camera_id)
