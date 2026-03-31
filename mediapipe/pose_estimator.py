"""
MediaPipe Pose Estimator  (MediaPipe Tasks API, v0.10+)
=======================================================
Provides keypoints (np.ndarray) and skeleton connections from:
  - live camera feed      → LivePoseEstimator
  - offline video file    → VideoPoseEstimator
  - single image / array  → ImagePoseEstimator

The default model (pose_landmarker_full.task) is downloaded automatically on
first use to the same directory as this script.

Keypoints are returned as np.ndarray of shape (33, 4):
    columns: [x, y, z, visibility]   (x, y in normalised [0,1] coords)

Connections are a list of (id_start, id_end) tuples.
"""

from __future__ import annotations

import os
import threading
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe import Image as MpImage, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarkerResult,
    PoseLandmarksConnections,
    RunningMode,
)

# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

_MODEL_URLS = {
    "lite":  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "full":  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
}


def _ensure_model(complexity: str, model_path: str | None) -> str:
    if model_path is None:
        model_path = os.path.join(
            os.path.dirname(__file__), f"pose_landmarker_{complexity}.task"
        )
    if not os.path.isfile(model_path):
        url = _MODEL_URLS[complexity]
        print(f"[PoseEstimator] Downloading {complexity} model → {model_path}")
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        urllib.request.urlretrieve(url, model_path)
        print("[PoseEstimator] Download complete.")
    return model_path


# ---------------------------------------------------------------------------
# Skeleton connections (static, same for all instances)
# ---------------------------------------------------------------------------

CONNECTIONS: list[tuple[int, int]] = [
    (c.start, c.end) for c in PoseLandmarksConnections.POSE_LANDMARKS
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _result_to_keypoints(result: PoseLandmarkerResult) -> np.ndarray | None:
    """Convert a PoseLandmarkerResult to a (33, 4) float32 array, or None."""
    if not result.pose_landmarks:
        return None
    lms = result.pose_landmarks[0]  # first detected person
    return np.array(
        [[lm.x, lm.y, lm.z, lm.visibility] for lm in lms],
        dtype=np.float32,
    )


def _draw_pose(frame_bgr: np.ndarray, keypoints: np.ndarray) -> None:
    """Draw landmarks and skeleton connections on frame_bgr in-place."""
    h, w = frame_bgr.shape[:2]
    for start, end in CONNECTIONS:
        if keypoints[start, 3] > 0.5 and keypoints[end, 3] > 0.5:
            pt1 = (int(keypoints[start, 0] * w), int(keypoints[start, 1] * h))
            pt2 = (int(keypoints[end, 0] * w), int(keypoints[end, 1] * h))
            cv2.line(frame_bgr, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)
    for x, y, _, vis in keypoints:
        if vis > 0.5:
            cv2.circle(frame_bgr, (int(x * w), int(y * h)), 4, (0, 0, 255), -1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Core class (IMAGE / VIDEO modes — synchronous)
# ---------------------------------------------------------------------------

class PoseEstimator:
    """
    Low-level wrapper around PoseLandmarker for IMAGE and VIDEO running modes.

    Parameters
    ----------
    model_path : str | None
        Path to the .task model file.  Downloaded automatically if omitted.
    model_complexity : str
        'lite', 'full' (default), or 'heavy'.
    num_poses : int
        Maximum number of poses to detect (default 1).
    mode : RunningMode
        RunningMode.IMAGE (default) or RunningMode.VIDEO.
    min_pose_detection_confidence : float
    min_pose_presence_confidence : float
    min_tracking_confidence : float
    """

    def __init__(
        self,
        model_path: str | None = None,
        model_complexity: str = "full",
        num_poses: int = 1,
        mode: RunningMode = RunningMode.IMAGE,
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        model_path = _ensure_model(model_complexity, model_path)
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=mode,
            num_poses=num_poses,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        self._mode = mode

    @property
    def connections(self) -> list[tuple[int, int]]:
        """Skeleton connections as a list of (id_start, id_end) tuples."""
        return CONNECTIONS

    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """
        Run pose estimation on a single BGR frame (IMAGE mode).

        Returns np.ndarray (33, 4) with [x, y, z, visibility], or None.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = MpImage(image_format=ImageFormat.SRGB, data=rgb)
        return _result_to_keypoints(self._landmarker.detect(mp_image))

    def process_video_frame(
        self, frame_bgr: np.ndarray, timestamp_ms: int
    ) -> np.ndarray | None:
        """
        Run pose estimation on a video frame (VIDEO mode).

        timestamp_ms must be monotonically increasing.
        Returns np.ndarray (33, 4) with [x, y, z, visibility], or None.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = MpImage(image_format=ImageFormat.SRGB, data=rgb)
        return _result_to_keypoints(
            self._landmarker.detect_for_video(mp_image, timestamp_ms)
        )

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# Live camera estimator  (LIVE_STREAM mode — async callback)
# ---------------------------------------------------------------------------

class LivePoseEstimator:
    """
    Streams pose estimation from a live camera using LIVE_STREAM mode.

    Usage
    -----
    with LivePoseEstimator(camera_index=0) as cam:
        print("Connections:", cam.connections[:5])
        for keypoints, frame in cam.stream():
            # keypoints : np.ndarray (33, 4) | None
            # frame     : annotated BGR frame
            cv2.imshow("Pose", frame)
        # Press 'q' to stop the generator
    """

    def __init__(
        self,
        camera_index: int = 0,
        draw_landmarks: bool = True,
        model_path: str | None = None,
        model_complexity: str = "full",
        num_poses: int = 1,
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self._cap = cv2.VideoCapture(camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {camera_index}")

        self._draw = draw_landmarks
        self._lock = threading.Lock()
        self._latest_keypoints: np.ndarray | None = None

        model_path = _ensure_model(model_complexity, model_path)
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.LIVE_STREAM,
            num_poses=num_poses,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            result_callback=self._on_result,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)

    def _on_result(
        self,
        result: PoseLandmarkerResult,
        output_image: MpImage,
        timestamp_ms: int,
    ) -> None:
        with self._lock:
            self._latest_keypoints = _result_to_keypoints(result)

    @property
    def connections(self) -> list[tuple[int, int]]:
        return CONNECTIONS

    def stream(self):
        """
        Generator yielding (keypoints, annotated_frame) for each camera frame.
        Stops when the feed ends or the user presses 'q'.
        """
        start_time = time.monotonic()
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break

            timestamp_ms = int((time.monotonic() - start_time) * 1000)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = MpImage(image_format=ImageFormat.SRGB, data=rgb)
            self._landmarker.detect_async(mp_image, timestamp_ms)

            with self._lock:
                keypoints = self._latest_keypoints

            if keypoints is not None and self._draw:
                _draw_pose(frame, keypoints)

            yield keypoints, frame

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    def close(self):
        self._landmarker.close()
        self._cap.release()
        cv2.destroyAllWindows()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# Offline video estimator  (VIDEO mode — synchronous per frame)
# ---------------------------------------------------------------------------

class VideoPoseEstimator:
    """
    Runs pose estimation on an offline video file.

    Usage
    -----
    with VideoPoseEstimator("clip.mp4") as vid:
        for frame_idx, keypoints, frame in vid.stream():
            ...                           # keypoints: (33,4) or None

        # Or process everything at once:
        all_kp = vid.process_all()        # list[(33,4)|None], one per frame
    """

    def __init__(
        self,
        video_path: str,
        draw_landmarks: bool = False,
        model_path: str | None = None,
        model_complexity: str = "full",
        num_poses: int = 1,
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        self._draw = draw_landmarks
        self._estimator = PoseEstimator(
            model_path=model_path,
            model_complexity=model_complexity,
            num_poses=num_poses,
            mode=RunningMode.VIDEO,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    @property
    def connections(self) -> list[tuple[int, int]]:
        return CONNECTIONS

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    @property
    def frame_count(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def stream(self):
        """Generator yielding (frame_index, keypoints, annotated_frame)."""
        idx = 0
        frame_duration_ms = 1000.0 / self.fps
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            timestamp_ms = int(idx * frame_duration_ms)
            keypoints = self._estimator.process_video_frame(frame, timestamp_ms)
            if keypoints is not None and self._draw:
                _draw_pose(frame, keypoints)
            yield idx, keypoints, frame
            idx += 1
        # Rewind so stream() can be called again
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def process_all(self) -> list[np.ndarray | None]:
        """
        Process every frame and return a list of keypoint arrays (or None per frame).
        """
        return [kp for _, kp, _ in self.stream()]

    def close(self):
        self._cap.release()
        self._estimator.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# Single image estimator  (IMAGE mode — static)
# ---------------------------------------------------------------------------

class ImagePoseEstimator:
    """
    Runs pose estimation on a single image.

    Usage
    -----
    with ImagePoseEstimator() as est:
        kp = est.process("photo.jpg")               # (33, 4) or None
        kp, annotated = est.process_and_draw("photo.jpg")
        connections = est.connections               # list[(int,int)]
    """

    def __init__(
        self,
        model_path: str | None = None,
        model_complexity: str = "full",
        num_poses: int = 1,
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
    ):
        self._estimator = PoseEstimator(
            model_path=model_path,
            model_complexity=model_complexity,
            num_poses=num_poses,
            mode=RunningMode.IMAGE,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
        )

    @property
    def connections(self) -> list[tuple[int, int]]:
        return CONNECTIONS

    def process(self, image: str | np.ndarray) -> np.ndarray | None:
        """
        Parameters
        ----------
        image : str | np.ndarray
            File path or BGR np.ndarray (e.g. from cv2.imread).

        Returns
        -------
        np.ndarray (33, 4) with [x, y, z, visibility], or None.
        """
        return self._estimator.process_frame(self._load(image))

    def process_and_draw(
        self, image: str | np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray]:
        """
        Same as process() but also returns an annotated BGR image.
        """
        frame = self._load(image).copy()
        keypoints = self._estimator.process_frame(frame)
        if keypoints is not None:
            _draw_pose(frame, keypoints)
        return keypoints, frame

    @staticmethod
    def _load(image: str | np.ndarray) -> np.ndarray:
        if isinstance(image, str):
            frame = cv2.imread(image)
            if frame is None:
                raise FileNotFoundError(f"Cannot load image: {image}")
            return frame
        return image

    def close(self):
        self._estimator.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# Quick demo — run this file directly to test with the default webcam
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Skeleton connections ({len(CONNECTIONS)}):", CONNECTIONS[:5], "...")
    print("Press 'q' to quit.")

    with LivePoseEstimator(camera_index=0, draw_landmarks=True, model_complexity="lite") as cam:
        for keypoints, frame in cam.stream():
            if keypoints is not None:
                nose = keypoints[0]
                print(
                    f"\rNose → x={nose[0]:.3f}  y={nose[1]:.3f}  vis={nose[3]:.2f}",
                    end="",
                    flush=True,
                )
            cv2.imshow("Live Pose", frame)
