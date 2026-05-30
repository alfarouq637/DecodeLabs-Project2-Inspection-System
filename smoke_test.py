from __future__ import annotations

from pathlib import Path

import numpy as np
import cv2

from fusion_coordinator import FusionCoordinator
from geometric_sonification import FFTSignalModel, image_to_frequency_spectrum
from scratch_cnn import build_gear_cnn, convolve2d, make_toy_gear_dataset, preprocess_bgr_for_cnn


class _MockModel:
    def __init__(self, probabilities: np.ndarray) -> None:
        self.probabilities = np.asarray(probabilities, dtype=np.float32)

    def predict_proba(self, model_input: np.ndarray) -> np.ndarray:
        return np.repeat(self.probabilities[None, :], model_input.shape[0], axis=0)


def main() -> None:
    kernel = np.array([[1, 0], [0, -1]], dtype=np.float32)
    image = np.arange(16, dtype=np.float32).reshape(4, 4)
    feature_map = convolve2d(image, kernel)
    assert feature_map.shape == (3, 3)

    bgr = np.zeros((128, 128, 3), dtype=np.uint8)
    cv2.circle(bgr, (64, 64), 46, (160, 160, 160), thickness=-1)
    cv2.circle(bgr, (64, 64), 18, (0, 0, 0), thickness=-1)
    cv2.putText(bgr, "STEEL 24T", (28, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (40, 40, 40), 1)
    prepared = preprocess_bgr_for_cnn(bgr, (32, 32))
    assert prepared.shape == (32, 32)
    assert np.isfinite(prepared).all()

    smoke_path = Path("_smoke_gear.png")
    cv2.imwrite(str(smoke_path), bgr)
    spectrum = image_to_frequency_spectrum(smoke_path, num_samples=256, spectrum_bins=64)
    smoke_path.unlink(missing_ok=True)
    assert spectrum.shape == (64,)
    assert np.isfinite(spectrum).all()
    signal_probabilities = FFTSignalModel().predict_proba(spectrum)
    assert signal_probabilities.shape == (1, 2)
    assert np.isclose(signal_probabilities.sum(), 1.0)

    coordinator = FusionCoordinator(
        _MockModel(np.array([0.55, 0.45])),
        _MockModel(np.array([0.97, 0.03])),
    )
    result = coordinator.predict_fusion(prepared, spectrum)
    assert result.label == "intact"
    assert result.fusion_weights[1] > result.fusion_weights[0]

    x, y, class_names = make_toy_gear_dataset(samples_per_class=4, image_size=32, seed=3)
    model = build_gear_cnn((1, 32, 32), class_names=class_names, seed=3)
    logits = model.forward(x[:2])
    assert logits.shape == (2, 2)

    history = model.fit(
        x,
        y,
        epochs=1,
        learning_rate=0.005,
        batch_size=4,
        rng=np.random.default_rng(3),
    )
    assert len(history.losses) == 1
    assert np.isfinite(history.losses[0])
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
