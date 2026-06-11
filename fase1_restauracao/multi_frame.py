import cv2
import numpy as np
from typing import List


class MultiFrameRestorer:
    def __init__(self, window_size: int = 5, use_optical_flow: bool = True):
        self.window_size = window_size
        self.use_optical_flow = use_optical_flow

    def _align_frame(
        self, reference: np.ndarray, target: np.ndarray
    ) -> np.ndarray:
        ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY) if len(reference.shape) == 3 else reference
        tgt_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY) if len(target.shape) == 3 else target

        if self.use_optical_flow:
            flow = cv2.calcOpticalFlowFarneback(
                tgt_gray, ref_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            h, w = ref_gray.shape
            map_x = np.float32(np.tile(np.arange(w), (h, 1))) + flow[:, :, 0]
            map_y = np.float32(np.tile(np.arange(h).reshape(-1, 1), (1, w))) + flow[:, :, 1]
            return cv2.remap(target, map_x, map_y, cv2.INTER_LINEAR)

        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
        try:
            _, warp_matrix = cv2.findTransformECC(
                ref_gray, tgt_gray, warp_matrix, cv2.MOTION_TRANSLATION, criteria
            )
        except cv2.error:
            pass
        return cv2.warpAffine(target, warp_matrix, (target.shape[1], target.shape[0]))

    def restore(self, frames: List[np.ndarray]) -> np.ndarray:
        if not frames:
            raise ValueError("Empty frame list")
        if len(frames) == 1:
            return frames[0]

        center_idx = len(frames) // 2
        reference = frames[center_idx]
        h, w = reference.shape[:2]

        accumulator = np.zeros_like(reference, dtype=np.float64)
        weight_sum = 0.0

        for i, frame in enumerate(frames):
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))

            aligned = self._align_frame(reference, frame) if i != center_idx else frame
            distance = abs(i - center_idx)
            weight = 1.0 / (1.0 + distance)

            accumulator += aligned.astype(np.float64) * weight
            weight_sum += weight

        result = accumulator / weight_sum
        return np.clip(result, 0, 255).astype(np.uint8)

    def restore_from_video(self, video_handler, frame_index: int) -> np.ndarray:
        frames = video_handler.get_frame_window(frame_index, self.window_size)
        if not frames:
            raise ValueError(f"No frames around index {frame_index}")
        return self.restore(frames)
