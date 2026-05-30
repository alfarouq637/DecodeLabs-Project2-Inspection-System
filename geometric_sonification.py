from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class FFTSignalModel:
    """Deterministic FFT-spectrum classifier for gear contour integrity.

    The model expects spectra from image_to_frequency_spectrum and returns
    probabilities shaped (batch, 2): [intact_probability, defective_probability].
    It uses peak-to-noise, sideband energy, broadband energy, and harmonic
    support instead of learned weights.
    """

    def __init__(
        self,
        *,
        min_tooth_bin: int = 4,
        max_tooth_bin: int = 96,
        harmonic_count: int = 3,
        decision_bias: float = -0.15,
        score_scale: float = 3.5,
    ) -> None:
        self.min_tooth_bin = min_tooth_bin
        self.max_tooth_bin = max_tooth_bin
        self.harmonic_count = harmonic_count
        self.decision_bias = decision_bias
        self.score_scale = score_scale

    def predict_proba(self, spectrum: np.ndarray) -> np.ndarray:
        spectra = np.asarray(spectrum, dtype=np.float32)
        if spectra.ndim == 1:
            spectra = spectra[None, :]
        if spectra.ndim != 2:
            raise ValueError("FFTSignalModel expects spectrum shaped (bins,) or (batch, bins).")

        probabilities = np.empty((spectra.shape[0], 2), dtype=np.float32)
        for i, row in enumerate(spectra):
            probabilities[i] = self._predict_one(row)
        return probabilities

    def _predict_one(self, spectrum: np.ndarray) -> np.ndarray:
        spectrum = np.nan_to_num(np.abs(spectrum).astype(np.float32), copy=False)
        if spectrum.size < self.min_tooth_bin + 4 or float(spectrum.sum()) <= 1e-9:
            return np.array([0.5, 0.5], dtype=np.float32)

        spectrum = spectrum.copy()
        spectrum[0] = 0.0
        norm = np.linalg.norm(spectrum)
        if norm > 1e-9:
            spectrum /= norm

        search_start = min(self.min_tooth_bin, spectrum.size - 2)
        search_stop = min(max(self.max_tooth_bin, search_start + 1), spectrum.size)
        search_band = spectrum[search_start:search_stop]
        if search_band.size == 0:
            return np.array([0.5, 0.5], dtype=np.float32)

        peak_offset = int(np.argmax(search_band))
        peak_bin = search_start + peak_offset
        peak_amp = float(spectrum[peak_bin])
        noise_floor = float(np.median(search_band)) + 1e-6
        peak_to_noise = peak_amp / noise_floor

        local_start = max(search_start, peak_bin - 3)
        local_stop = min(search_stop, peak_bin + 4)
        sideband = spectrum[local_start:local_stop].copy()
        sideband[peak_bin - local_start] = 0.0
        sideband_ratio = float(sideband.sum() / (peak_amp + 1e-6))

        broadband = spectrum[search_start:search_stop].copy()
        broadband[max(0, peak_bin - 2 - search_start) : min(search_band.size, peak_bin + 3 - search_start)] = 0.0
        broadband_ratio = float(broadband.sum() / (search_band.sum() + 1e-6))

        harmonic_support = self._harmonic_support(spectrum, peak_bin, peak_amp)
        sharpness = peak_amp / (float(search_band.mean()) + 1e-6)

        defect_evidence = (
            0.36 * _sigmoid((2.2 - peak_to_noise) / 0.65)
            + 0.26 * _sigmoid((sideband_ratio - 0.55) / 0.20)
            + 0.24 * _sigmoid((broadband_ratio - 0.72) / 0.12)
            + 0.14 * _sigmoid((1.6 - sharpness) / 0.35)
        )
        intact_evidence = (
            0.42 * _sigmoid((peak_to_noise - 2.6) / 0.75)
            + 0.28 * _sigmoid((sharpness - 2.0) / 0.45)
            + 0.18 * harmonic_support
            + 0.12 * _sigmoid((0.65 - broadband_ratio) / 0.14)
        )

        defective_logit = self.score_scale * (defect_evidence - intact_evidence + self.decision_bias)
        defective_prob = float(_sigmoid(defective_logit))
        intact_prob = 1.0 - defective_prob
        return np.array([intact_prob, defective_prob], dtype=np.float32)

    def _harmonic_support(self, spectrum: np.ndarray, peak_bin: int, peak_amp: float) -> float:
        if peak_amp <= 1e-9:
            return 0.0

        support = []
        for harmonic in range(2, self.harmonic_count + 1):
            target = peak_bin * harmonic
            if target >= spectrum.size:
                break
            start = max(1, target - 1)
            stop = min(spectrum.size, target + 2)
            support.append(float(spectrum[start:stop].max() / (peak_amp + 1e-6)))

        if not support:
            return 0.0
        return float(np.clip(np.mean(support), 0.0, 1.0))


def image_to_frequency_spectrum(
    image_path: str | Path,
    *,
    num_samples: int = 2048,
    spectrum_bins: int | None = 512,
    apply_log: bool = True,
) -> np.ndarray:
    """Convert a gear image into a normalized 1D geometric frequency spectrum.

    The spatial contour is treated as a virtual vibration signal: radius from
    gear center as a function of angular position. Only NumPy and OpenCV are
    used so the method remains lightweight for edge deployment.
    """
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Could not open image: {image_path}")
    if num_samples < 32:
        raise ValueError("num_samples must be at least 32.")

    contour = _extract_outer_contour(gray)
    center = _contour_center(contour)
    radial_signal = _contour_to_radial_signal(contour, center, num_samples)

    radial_signal = radial_signal.astype(np.float32, copy=False)
    radial_signal -= radial_signal.mean()
    radial_signal /= radial_signal.std() + 1e-6

    spectrum = np.abs(np.fft.rfft(radial_signal)).astype(np.float32)
    spectrum[0] = 0.0

    if spectrum_bins is not None:
        if spectrum_bins <= 0:
            raise ValueError("spectrum_bins must be positive or None.")
        spectrum = spectrum[: min(spectrum_bins, spectrum.size)]

    if apply_log:
        spectrum = np.log1p(spectrum)

    norm = np.linalg.norm(spectrum)
    if norm > 1e-9:
        spectrum /= norm
    return spectrum.astype(np.float32, copy=False)


def image_to_radial_signal(
    image_path: str | Path,
    *,
    num_samples: int = 2048,
) -> np.ndarray:
    """Return the normalized radius-vs-angle signal before FFT."""
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Could not open image: {image_path}")
    contour = _extract_outer_contour(gray)
    signal = _contour_to_radial_signal(contour, _contour_center(contour), num_samples)
    signal = signal.astype(np.float32, copy=False)
    signal -= signal.mean()
    signal /= signal.std() + 1e-6
    return signal


def _extract_outer_contour(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, binary_inv = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    candidates = []
    for mask in (binary, binary_inv):
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        candidates.extend(contours)

    if not candidates:
        raise ValueError("No gear contour found.")

    image_height, image_width = gray.shape[:2]
    image_area = image_height * image_width
    best_contour = None
    best_score = -1.0

    for contour in candidates:
        area = cv2.contourArea(contour)
        if area < image_area * 0.01 or area > image_area * 0.95:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        touches_border = x <= 1 or y <= 1 or x + w >= image_width - 1 or y + h >= image_height - 1
        border_weight = 0.20 if touches_border else 1.0
        cx = x + w / 2.0
        cy = y + h / 2.0
        center_distance = np.hypot(cx - image_width / 2.0, cy - image_height / 2.0)
        center_weight = 1.0 - min(0.8, center_distance / max(image_width, image_height))
        score = area * border_weight * center_weight

        if score > best_score:
            best_score = score
            best_contour = contour

    if best_contour is None:
        best_contour = max(candidates, key=cv2.contourArea)
    return best_contour


def _contour_center(contour: np.ndarray) -> tuple[float, float]:
    moments = cv2.moments(contour)
    if abs(moments["m00"]) > 1e-9:
        return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]

    points = contour.reshape(-1, 2).astype(np.float32)
    return float(points[:, 0].mean()), float(points[:, 1].mean())


def _contour_to_radial_signal(
    contour: np.ndarray,
    center: tuple[float, float],
    num_samples: int,
) -> np.ndarray:
    points = contour.reshape(-1, 2).astype(np.float32)
    dx = points[:, 0] - center[0]
    dy = points[:, 1] - center[1]
    theta = (np.arctan2(dy, dx) + 2.0 * np.pi) % (2.0 * np.pi)
    radius = np.sqrt(dx * dx + dy * dy)

    bin_index = np.floor(theta * num_samples / (2.0 * np.pi)).astype(np.int32)
    bin_index = np.clip(bin_index, 0, num_samples - 1)

    radial = np.full(num_samples, -np.inf, dtype=np.float32)
    np.maximum.at(radial, bin_index, radius)

    valid = np.isfinite(radial)
    if valid.all():
        return radial
    if not valid.any():
        raise ValueError("Could not build radial signal from contour.")

    valid_index = np.flatnonzero(valid)
    valid_radius = radial[valid]
    circular_x = np.concatenate(
        [valid_index - num_samples, valid_index, valid_index + num_samples]
    )
    circular_y = np.concatenate([valid_radius, valid_radius, valid_radius])
    return np.interp(np.arange(num_samples), circular_x, circular_y).astype(np.float32)


def _sigmoid(value: float) -> float:
    value = float(np.clip(value, -40.0, 40.0))
    return 1.0 / (1.0 + np.exp(-value))
