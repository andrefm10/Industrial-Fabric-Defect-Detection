import cv2
import numpy as np
from typing import List, Tuple


class GaborAnalyzer:
    def __init__(
        self,
        orientations: List[float] = None,
        frequencies: List[float] = None,
        sigma: float = 3.0,
        kernel_size: int = 31,
    ):
        self.orientations = orientations or [0, 45, 90, 135]
        self.frequencies = frequencies or [0.05, 0.1, 0.15, 0.25]
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.kernels = self._build_filter_bank()

    def _build_filter_bank(self) -> List[Tuple[float, float, np.ndarray]]:
        kernels = []
        for theta_deg in self.orientations:
            theta = np.radians(theta_deg)
            for freq in self.frequencies:
                lambd = 1.0 / freq
                kernel = cv2.getGaborKernel(
                    (self.kernel_size, self.kernel_size),
                    self.sigma, theta, lambd, gamma=0.5, psi=0,
                )
                kernel /= kernel.sum() + 1e-8
                kernels.append((theta_deg, freq, kernel))
        return kernels

    def compute_responses(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gray = gray.astype(np.float64)

        responses = np.zeros(
            (len(self.kernels), gray.shape[0], gray.shape[1]), dtype=np.float64
        )
        for i, (_, _, kernel) in enumerate(self.kernels):
            responses[i] = np.abs(cv2.filter2D(gray, cv2.CV_64F, kernel))

        return responses

    def compute_energy_map(self, image: np.ndarray) -> np.ndarray:
        responses = self.compute_responses(image)
        return responses.mean(axis=0)

    def compute_orientation_map(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gray = gray.astype(np.float64)

        orientation_energies = {}
        for theta_deg, _, kernel in self.kernels:
            response = np.abs(cv2.filter2D(gray, cv2.CV_64F, kernel))
            if theta_deg not in orientation_energies:
                orientation_energies[theta_deg] = []
            orientation_energies[theta_deg].append(response)

        orientation_map = np.zeros_like(gray)
        max_energy = np.zeros_like(gray)

        for theta_deg, responses in orientation_energies.items():
            avg_energy = np.mean(responses, axis=0)
            mask = avg_energy > max_energy
            orientation_map[mask] = theta_deg
            max_energy[mask] = avg_energy[mask]

        return orientation_map

    def detect_anomalies(
        self, image: np.ndarray, sensitivity: float = 2.0
    ) -> np.ndarray:
        energy = self.compute_energy_map(image)
        mean_energy = energy.mean()
        std_energy = energy.std()
        threshold = mean_energy + sensitivity * std_energy
        anomaly_mask = (energy > threshold).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_CLOSE, kernel)
        anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_OPEN, kernel)
        return anomaly_mask
