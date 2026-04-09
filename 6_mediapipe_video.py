from pose_estimator.pose_estimator import VideoPoseEstimator
import os
import cv2
import json

if __name__ == "__main__":

    ids = ['104122061649', '950122061749']
    dir_path = os.path.dirname(os.path.realpath(__file__)) + "/realsense/"

    video_processed_data = {}

    for cam_id in ids:
        video_processed_data[f"{cam_id}"] = {}

        video_path = dir_path + f"/data/Lazar/Squat/video/{cam_id}/vid_undistorted.mp4"

        pose_estimator = VideoPoseEstimator(video_path, draw_landmarks=True)

        for idx, keypoints, frame in pose_estimator.stream():
            video_processed_data[f"{cam_id}"][f"frame_{idx:05d}"] = keypoints.tolist()

            print(f"Frame {idx}: {keypoints}")
            cv2.imshow('Pose Estimation', frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
                break
        cv2.destroyAllWindows()

        pose_estimator.close()

    json.dump(video_processed_data, open(dir_path + f"/data/Lazar/Squat/video/processed_data.json", 'w'), indent=4)
