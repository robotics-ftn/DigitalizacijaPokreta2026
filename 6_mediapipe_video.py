from pose_estimator.pose_estimator import PoseEstimator
import os
import cv2
import json
import numpy as np
from mediapipe.tasks.python.vision import RunningMode


def rotate_keypoints_back_ccw(keypoints: np.ndarray) -> np.ndarray:
    """
    Invert the effect of rotating a frame 90° CCW before MediaPipe.

    MediaPipe returns normalised (x', y') on the rotated frame.
    This maps them back to the original (un-rotated) frame coords:
        x_orig = 1 - y'
        y_orig = x'
    z and visibility are left unchanged.

    Use this when the physical camera was rotated 90° CW (tilted right) and
    you corrected it with cv2.ROTATE_90_COUNTERCLOCKWISE.

    If the camera was tilted the other way, swap to:
        x_orig = y'
        y_orig = 1 - x'
    (that corresponds to cv2.ROTATE_90_CLOCKWISE correction).
    """
    kp = keypoints.copy()
    x_prime = keypoints[:, 0].copy()
    y_prime = keypoints[:, 1].copy()
    kp[:, 0] = 1.0 - y_prime   # x_orig
    kp[:, 1] = x_prime          # y_orig
    return kp


if __name__ == "__main__":

    ids = ['104122061649', '950122061749']
    dir_path = os.path.dirname(os.path.realpath(__file__)) + "/realsense/"

    video_processed_data = {}

    for cam_id in ids:
        video_processed_data[f"{cam_id}"] = {}

        video_path = dir_path + f"/data/Lazar/Squat/video/{cam_id}/vid_undistorted.mp4"

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_duration_ms = 1000.0 / fps

        estimator = PoseEstimator(mode=RunningMode.VIDEO)

        idx = 0
        cv2.namedWindow(f'Pose Estimation {cam_id}', cv2.WINDOW_KEEPRATIO | cv2.WINDOW_NORMAL)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Correct 90° CW camera rotation by rotating the frame CCW
            frame_corrected = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            timestamp_ms = int(idx * frame_duration_ms)
            keypoints = estimator.process_video_frame(frame_corrected, timestamp_ms)

            if keypoints is not None:
                # Draw on the corrected frame for preview
                from pose_estimator.pose_estimator import _draw_pose
                _draw_pose(frame_corrected, keypoints)

                # Map normalised coords back to the original (un-rotated) frame
                keypoints = rotate_keypoints_back_ccw(keypoints)
                video_processed_data[f"{cam_id}"][f"frame_{idx:05d}"] = keypoints.tolist()
                # print(f"Cam {cam_id} | Frame {idx}: {keypoints[:, :2]}")

            cv2.imshow(f'Pose Estimation {cam_id}', frame_corrected)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                break
            idx += 1
            cv2.waitKey(int(1000/fps))

        cap.release()
        estimator.close()
        cv2.destroyAllWindows()

    out_path = dir_path + "/data/Lazar/Squat/video/processed_data.json"
    json.dump(video_processed_data, open(out_path, 'w'), indent=4)
