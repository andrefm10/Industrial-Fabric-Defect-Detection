import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Optional, Tuple


class VideoHandler:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.path}")

    @property
    def fps(self) -> float:
        return self.cap.get(cv2.CAP_PROP_FPS) or 30.0

    @property
    def frame_count(self) -> int:
        count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count <= 0:
            return 0
        return count

    @property
    def resolution(self) -> Tuple[int, int]:
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    def seek_to_time(self, seconds: float) -> Optional[np.ndarray]:
        target_frame = int(seconds * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = self.cap.read()
        return frame if ret else None

    def get_frame(self, index: int) -> Optional[np.ndarray]:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self.cap.read()
        return frame if ret else None

    def get_frame_window(self, center_index: int, window_size: int) -> list[np.ndarray]:
        half = window_size // 2
        start = max(0, center_index - half)
        end = min(self.frame_count - 1, center_index + half)
        frames = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for _ in range(start, end + 1):
            ret, frame = self.cap.read()
            if ret:
                frames.append(frame)
        return frames

    def iterate_frames(
        self, start: int = 0, end: Optional[int] = None, step: int = 1
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        end = end or self.frame_count
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        idx = start
        while idx < end:
            ret, frame = self.cap.read()
            if not ret:
                break
            if (idx - start) % step == 0:
                yield idx, frame
            idx += 1

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()
