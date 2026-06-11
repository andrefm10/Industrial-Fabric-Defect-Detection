import cv2
import numpy as np
from scipy.signal import wiener as scipy_wiener


def create_motion_psf(length: int, angle: float = 0.0) -> np.ndarray:
    size = max(length, 3)
    if size % 2 == 0:
        size += 1

    psf = np.zeros((size, size), dtype=np.float64)
    center = size // 2
    rad = np.radians(angle)

    for i in range(length):
        offset = i - length // 2
        x = int(center + offset * np.cos(rad))
        y = int(center + offset * np.sin(rad))
        if 0 <= x < size and 0 <= y < size:
            psf[y, x] = 1.0

    psf /= psf.sum()
    return psf


def wiener_deconvolution(
    image: np.ndarray, psf: np.ndarray, snr: float = 0.01
) -> np.ndarray:
    if len(image.shape) == 3:
        channels = cv2.split(image)
        restored = [_wiener_channel(ch, psf, snr) for ch in channels]
        return cv2.merge(restored)
    return _wiener_channel(image, psf, snr)


def _wiener_channel(
    channel: np.ndarray, psf: np.ndarray, snr: float
) -> np.ndarray:
    channel_f = np.float64(channel)
    h, w = channel_f.shape

    psf_padded = np.zeros((h, w), dtype=np.float64)
    kh, kw = psf.shape
    ph, pw = kh // 2, kw // 2
    psf_padded[:kh, :kw] = psf
    psf_padded = np.roll(psf_padded, -ph, axis=0)
    psf_padded = np.roll(psf_padded, -pw, axis=1)

    H = np.fft.fft2(psf_padded)
    G = np.fft.fft2(channel_f)

    H_conj = np.conj(H)
    H_sq = np.abs(H) ** 2
    W = H_conj / (H_sq + snr)

    restored = np.fft.ifft2(G * W)
    restored = np.abs(restored)
    return np.clip(restored, 0, 255).astype(np.uint8)


def wiener_scipy(image: np.ndarray, noise_power: float = None) -> np.ndarray:
    if len(image.shape) == 3:
        channels = cv2.split(image)
        restored = [
            np.clip(scipy_wiener(ch.astype(np.float64), noise=noise_power), 0, 255).astype(np.uint8)
            for ch in channels
        ]
        return cv2.merge(restored)
    result = scipy_wiener(image.astype(np.float64), noise=noise_power)
    return np.clip(result, 0, 255).astype(np.uint8)


class WienerRestorer:
    def __init__(
        self,
        blur_length: int = 15,
        blur_angle: float = 0.0,
        snr: float = 0.01,
    ):
        self.blur_length = blur_length
        self.blur_angle = blur_angle
        self.snr = snr
        self.psf = create_motion_psf(blur_length, blur_angle)

    def restore(self, image: np.ndarray) -> np.ndarray:
        return wiener_deconvolution(image, self.psf, self.snr)

    def restore_adaptive(self, image: np.ndarray) -> np.ndarray:
        best_result = None
        best_sharpness = -1

        for snr_candidate in [0.001, 0.005, 0.01, 0.02, 0.05]:
            result = wiener_deconvolution(image, self.psf, snr_candidate)
            sharpness = cv2.Laplacian(
                cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result,
                cv2.CV_64F,
            ).var()
            if sharpness > best_sharpness:
                best_sharpness = sharpness
                best_result = result

        return best_result
