import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import savgol_filter

from realsense.types import Camera
from realsense.reconstruct_3d import reconstruct_points

# MediaPipe Pose landmark connections (indices into 33 keypoints)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]

CONFIDENCE_THRESHOLD = 0.3
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # ---- Load keypoint data ----
    data_path = os.path.join(dir_path, "data", "Lazar", "Squat", "video", "processed_data.json")
    with open(data_path, 'r') as f:
        raw_data = json.load(f)

    cam_ids = list(raw_data.keys())
    print(f"Cameras: {cam_ids}")

    # ---- Load camera calibration ----
    # pose.yaml contains extrinsics (Rot, Trans) in a common world frame computed
    # via solvePnPRansac against a fixed chessboard — required for 3D reconstruction.
    cameras = []
    for cam_id in cam_ids:
        calib_file = os.path.join(dir_path, "output", "calib", cam_id, "pose.yaml")
        fs = cv2.FileStorage(calib_file, cv2.FILE_STORAGE_READ)
        intrinsics = fs.getNode("intrinsic").mat()
        distortion = fs.getNode("distortion").mat()
        rot = fs.getNode("Rot").mat()
        trans = fs.getNode("Trans").mat()
        fs.release()
        cameras.append(Camera(cam_id, intrinsics, distortion, rot, trans))

    # ---- Parse keypoints: (num_frames, num_keypoints, 4) per camera ----
    frame_keys = sorted(raw_data[cam_ids[0]].keys())
    num_frames = len(frame_keys)
    num_keypoints = 33

    keypoints = {}
    for cam_id in cam_ids:
        kp_frames = []
        for fk in frame_keys:
            kp_frames.append(np.array(raw_data[cam_id][fk]))  # (33, 4)
        keypoints[cam_id] = np.array(kp_frames)  # (num_frames, 33, 4)

    # ---- Reconstruct 3D points for every frame ----
    all_points_3d = []  # list of (33,) elements, each is np.ndarray(3,) or None

    for frame_idx in range(num_frames):
        frame_points = []
        for kp_idx in range(num_keypoints):
            valid_cameras = []
            valid_pixels = []

            for cam_id, cam in zip(cam_ids, cameras):
                pt = keypoints[cam_id][frame_idx, kp_idx]  # [x_norm, y_norm, z, conf]
                if pt[3] > CONFIDENCE_THRESHOLD:
                    x_px = pt[0] * IMAGE_WIDTH
                    y_px = pt[1] * IMAGE_HEIGHT
                    valid_cameras.append(cam)
                    valid_pixels.append(np.array([x_px, y_px]))

            if len(valid_cameras) >= 2:
                point_3d = reconstruct_points(valid_cameras, valid_pixels)
                frame_points.append(point_3d)
            else:
                frame_points.append(None)

        all_points_3d.append(frame_points)

    # ---- Keep raw data; build filtered copies ----
    def copy_frames(src):
        return [[p.copy() if p is not None else None for p in frame] for frame in src]

    raw_points_3d = copy_frames(all_points_3d)

    # EMA filter (applied to raw)
    EMA_ALPHA = 0.4
    ema_points_3d = copy_frames(all_points_3d)
    for kp_idx in range(num_keypoints):
        ema_val = None
        for frame_idx in range(num_frames):
            pt = ema_points_3d[frame_idx][kp_idx]
            if pt is not None:
                ema_val = pt.copy() if ema_val is None else EMA_ALPHA * pt + (1.0 - EMA_ALPHA) * ema_val
                ema_points_3d[frame_idx][kp_idx] = ema_val.copy()

    # Savitzky-Golay filter (applied to raw)
    SG_WINDOW = 21
    SG_POLY = 5
    sg_points_3d = copy_frames(all_points_3d)
    for kp_idx in range(num_keypoints):
        valid_idx = [fi for fi in range(num_frames) if sg_points_3d[fi][kp_idx] is not None]
        if len(valid_idx) >= SG_WINDOW:
            pts_arr = np.array([sg_points_3d[fi][kp_idx] for fi in valid_idx])
            smoothed = savgol_filter(pts_arr, window_length=SG_WINDOW, polyorder=SG_POLY, axis=0)
            for arr_i, fi in enumerate(valid_idx):
                sg_points_3d[fi][kp_idx] = smoothed[arr_i]

    # ---- Compute shared axis limits from raw data ----
    all_valid = np.array([p for frame in raw_points_3d for p in frame if p is not None])
    margin = 200
    xlim = (all_valid[:, 0].min() - margin, all_valid[:, 0].max() + margin)
    ylim = (all_valid[:, 1].min() - margin, all_valid[:, 1].max() + margin)
    zlim = (all_valid[:, 2].min() - margin, all_valid[:, 2].max() + margin)

    # ---- Set up 3-panel animation ----
    datasets = [
        (raw_points_3d, 'Original',                              'steelblue',     'dimgray'),
        (ema_points_3d, f'EMA (\u03b1={EMA_ALPHA})',             'tomato',        'firebrick'),
        (sg_points_3d,  f'Savitzky-Golay (w={SG_WINDOW}, p={SG_POLY})', 'mediumseagreen', 'darkgreen'),
    ]

    fig = plt.figure(figsize=(21, 8))
    axes = [fig.add_subplot(1, 3, i + 1, projection='3d') for i in range(3)]

    scatters = []
    all_bone_lines = []
    for ax, (_, title, pt_color, ln_color) in zip(axes, datasets):
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
        ax.set_title(title)
        scatters.append(ax.scatter([], [], [], c=pt_color, s=15, depthshade=True))
        all_bone_lines.append(
            [ax.plot([], [], [], color=ln_color, lw=1.2)[0] for _ in POSE_CONNECTIONS]
        )

    def _draw_skeleton(sc, bone_lines, pts):
        coords = np.array([p if p is not None else [np.nan, np.nan, np.nan] for p in pts])
        sc._offsets3d = (coords[:, 0], coords[:, 1], coords[:, 2])
        for line, (i, j) in zip(bone_lines, POSE_CONNECTIONS):
            pi, pj = pts[i], pts[j]
            if pi is not None and pj is not None:
                line.set_data([pi[0], pj[0]], [pi[1], pj[1]])
                line.set_3d_properties([pi[2], pj[2]])
            else:
                line.set_data([], [])
                line.set_3d_properties([])

    def update(frame_idx):
        fig.suptitle(f'Squat \u2014 Frame {frame_idx + 1}/{num_frames}', fontsize=13)
        artists = []
        for sc, bone_lines, (pts_list, _, _, _) in zip(scatters, all_bone_lines, datasets):
            _draw_skeleton(sc, bone_lines, pts_list[frame_idx])
            artists.append(sc)
            artists.extend(bone_lines)
        return artists

    ani = animation.FuncAnimation(
        fig, update, frames=num_frames, interval=50, blit=False
    )

    plt.tight_layout()
    plt.show()
