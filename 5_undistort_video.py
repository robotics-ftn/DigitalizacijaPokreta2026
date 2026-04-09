import numpy as np
import cv2
import json
import os


def load_calib(calib_path):
    with open(calib_path, 'r') as f:
        file = cv2.FileStorage(calib_path, cv2.FILE_STORAGE_READ)
        if file.isOpened():
            cam_mat = file.getNode("intrinsic").mat()
            dist_coeffs = file.getNode("distortion").mat()
        else:
            print("Failed to open calibration file.")
            return None, None
    return cam_mat, dist_coeffs


def load_video_and_undistort(video_path, cam_mat, dist_coeffs):
    cap = cv2.VideoCapture(video_path)
    undistorted_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        undistorted = cv2.undistort(frame, cam_mat, dist_coeffs, None, cam_mat)
        undistorted_frames.append(undistorted)
    cap.release()
    return undistorted_frames


if __name__ == "__main__":

    # Load all videos, undistort them and save them again as mp4
    ids = ['104122061649', '950122061749']
    dir_path = os.path.dirname(os.path.realpath(__file__)) + "/realsense/"

    vid_name = "{serial}/vid_{num:05d}.mp4"

    for cam_id in ids:

        video_path = dir_path + f"/data/Lazar/Squat/video/{cam_id}/vid_00000.mp4"
        calib_path = dir_path + f"/output/calib/{cam_id}/pose.yaml"

        undistorted_frames = load_video_and_undistort(video_path, *load_calib(calib_path))
        save_path = dir_path + f"/data/Lazar/Squat/video/{cam_id}/vid_undistorted.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width, _ = undistorted_frames[0].shape
        out = cv2.VideoWriter(save_path, fourcc, 15.0, (width, height))
        for frame in undistorted_frames:
            out.write(frame)
        out.release()
