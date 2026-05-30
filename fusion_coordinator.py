from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


@dataclass
class FusionResult:
    label: str
    class_index: int
    fused_probabilities: np.ndarray
    visual_probabilities: np.ndarray
    signal_probabilities: np.ndarray
    fusion_weights: np.ndarray
    visual_confidence: float
    signal_confidence: float
    fused_confidence: float
    method: str


def setup_research_logger(
    log_path: str | Path = "research_log.txt",
    *,
    logger_name: str = "decodelabs_robotics_automation_internship_project_2.fusion",
) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    resolved_path = str(Path(log_path))
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == resolved_path:
            return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(resolved_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


class FusionCoordinator:
    """Late-fusion coordinator for visual and geometric-spectrum engines."""

    def __init__(
        self,
        visual_model: Any,
        signal_model: Any,
        *,
        class_names: Sequence[str] = ("intact", "defective"),
        visual_reliability: float = 1.0,
        signal_reliability: float = 1.0,
        high_confidence: float = 0.95,
        low_confidence: float = 0.65,
        meta_learner: Any | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.visual_model = visual_model
        self.signal_model = signal_model
        self.class_names = list(class_names)
        self.visual_reliability = float(visual_reliability)
        self.signal_reliability = float(signal_reliability)
        self.high_confidence = float(high_confidence)
        self.low_confidence = float(low_confidence)
        self.meta_learner = meta_learner
        self.logger = logger or setup_research_logger()

    def predict_fusion(self, image_2d: np.ndarray, signal_1d: np.ndarray) -> FusionResult:
        visual_input = _as_visual_batch(image_2d)
        signal_input = _as_signal_batch(signal_1d)

        visual_probs = _predict_proba(self.visual_model, visual_input)[0]
        signal_probs = _predict_proba(self.signal_model, signal_input)[0]
        _validate_probabilities(visual_probs, signal_probs)

        self.logger.info(
            "Fusion inputs | visual_shape=%s | signal_shape=%s | visual_probs=%s | signal_probs=%s",
            visual_input.shape,
            signal_input.shape,
            np.round(visual_probs, 6).tolist(),
            np.round(signal_probs, 6).tolist(),
        )

        if self.meta_learner is not None:
            fused_probs, weights = self._predict_meta_fusion(visual_probs, signal_probs)
            method = "sklearn_late_fusion_meta_learner"
        else:
            weights = self._dynamic_confidence_weights(visual_probs, signal_probs)
            fused_probs = weights[0] * visual_probs + weights[1] * signal_probs
            fused_probs = fused_probs / (fused_probs.sum() + 1e-9)
            method = "confidence_weighted_late_fusion"

        class_index = int(np.argmax(fused_probs))
        result = FusionResult(
            label=self.class_names[class_index],
            class_index=class_index,
            fused_probabilities=fused_probs.astype(np.float32),
            visual_probabilities=visual_probs.astype(np.float32),
            signal_probabilities=signal_probs.astype(np.float32),
            fusion_weights=weights.astype(np.float32),
            visual_confidence=float(np.max(visual_probs)),
            signal_confidence=float(np.max(signal_probs)),
            fused_confidence=float(np.max(fused_probs)),
            method=method,
        )

        self.logger.info(
            "Fusion result | method=%s | weights=%s | visual_conf=%.6f | signal_conf=%.6f | "
            "fused_conf=%.6f | label=%s | fused_probs=%s",
            result.method,
            np.round(result.fusion_weights, 6).tolist(),
            result.visual_confidence,
            result.signal_confidence,
            result.fused_confidence,
            result.label,
            np.round(result.fused_probabilities, 6).tolist(),
        )
        return result

    def _dynamic_confidence_weights(
        self,
        visual_probs: np.ndarray,
        signal_probs: np.ndarray,
    ) -> np.ndarray:
        visual_conf = float(np.max(visual_probs))
        signal_conf = float(np.max(signal_probs))

        visual_trust = self.visual_reliability * _certainty_score(visual_probs)
        signal_trust = self.signal_reliability * _certainty_score(signal_probs)

        if visual_conf >= self.high_confidence and signal_conf <= self.low_confidence:
            visual_trust *= 4.0
        if signal_conf >= self.high_confidence and visual_conf <= self.low_confidence:
            signal_trust *= 4.0

        trusts = np.array([visual_trust, signal_trust], dtype=np.float32)
        trusts = np.maximum(trusts, 1e-4)
        return trusts / trusts.sum()

    def _predict_meta_fusion(
        self,
        visual_probs: np.ndarray,
        signal_probs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        features = build_meta_features(visual_probs[None, :], signal_probs[None, :])
        fused_probs = np.asarray(self.meta_learner.predict_proba(features), dtype=np.float32)[0]
        fused_probs = fused_probs / (fused_probs.sum() + 1e-9)
        weights = modality_weights_from_meta_features(self.meta_learner, visual_probs.size)
        self.logger.info(
            "Meta fusion layer | feature_shape=%s | modality_importance=%s",
            features.shape,
            np.round(weights, 6).tolist(),
        )
        return fused_probs, weights


class SklearnLateFusionMetaLearner:
    """Stacking classifier over visual and signal probability vectors."""

    def __init__(
        self,
        *,
        class_names: Sequence[str] = ("intact", "defective"),
        random_state: int = 7,
        logger: logging.Logger | None = None,
    ) -> None:
        self.class_names = list(class_names)
        self.random_state = random_state
        self.logger = logger or setup_research_logger()
        self.estimator: Any | None = None

    def fit(
        self,
        visual_probabilities: np.ndarray,
        signal_probabilities: np.ndarray,
        labels: np.ndarray,
    ) -> "SklearnLateFusionMetaLearner":
        from sklearn.linear_model import LogisticRegression

        features = build_meta_features(visual_probabilities, signal_probabilities)
        labels = np.asarray(labels, dtype=np.int64)
        self.estimator = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=self.random_state,
        )
        self.estimator.fit(features, labels)

        weights = modality_weights_from_meta_features(self, visual_probabilities.shape[1])
        self.logger.info(
            "Meta learner fitted | feature_shape=%s | labels_shape=%s | modality_importance=%s",
            features.shape,
            labels.shape,
            np.round(weights, 6).tolist(),
        )
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.estimator is None:
            raise RuntimeError("Meta learner must be fitted before predict_proba.")
        probabilities = self.estimator.predict_proba(features)
        self.logger.info(
            "Meta learner predict | feature_shape=%s | output_shape=%s | probabilities=%s",
            features.shape,
            probabilities.shape,
            np.round(probabilities, 6).tolist(),
        )
        return probabilities

    @property
    def coef_(self) -> np.ndarray:
        if self.estimator is None:
            raise RuntimeError("Meta learner is not fitted.")
        return self.estimator.coef_


def build_meta_features(
    visual_probabilities: np.ndarray,
    signal_probabilities: np.ndarray,
) -> np.ndarray:
    visual = np.asarray(visual_probabilities, dtype=np.float32)
    signal = np.asarray(signal_probabilities, dtype=np.float32)
    if visual.ndim != 2 or signal.ndim != 2:
        raise ValueError("Probability inputs must be shaped (batch, classes).")
    if visual.shape != signal.shape:
        raise ValueError("Visual and signal probabilities must have the same shape.")

    confidence = np.stack([visual.max(axis=1), signal.max(axis=1)], axis=1)
    disagreement = np.abs(visual - signal)
    return np.concatenate([visual, signal, disagreement, confidence], axis=1)


def modality_weights_from_meta_features(meta_learner: Any, class_count: int) -> np.ndarray:
    if not hasattr(meta_learner, "coef_"):
        return np.array([0.5, 0.5], dtype=np.float32)

    coefficients = np.abs(np.asarray(meta_learner.coef_, dtype=np.float32))
    if coefficients.ndim == 1:
        coefficients = coefficients[None, :]
    feature_importance = coefficients.mean(axis=0)
    visual_importance = float(feature_importance[:class_count].sum())
    signal_importance = float(feature_importance[class_count : 2 * class_count].sum())
    weights = np.array([visual_importance, signal_importance], dtype=np.float32)
    if weights.sum() <= 1e-9:
        return np.array([0.5, 0.5], dtype=np.float32)
    return weights / weights.sum()


def _as_visual_batch(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        return image[None, None, :, :]
    if image.ndim == 3:
        if image.shape[0] in (1, 3):
            return image[None, :, :, :]
        return image[:, None, :, :]
    if image.ndim == 4:
        return image
    raise ValueError("Visual input must be 2D, 3D, or 4D.")


def _as_signal_batch(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim == 1:
        return signal[None, :]
    if signal.ndim == 2:
        return signal
    raise ValueError("Signal input must be 1D or 2D.")


def _predict_proba(model: Any, model_input: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(model_input), dtype=np.float32)
    if probabilities.ndim == 1:
        probabilities = probabilities[None, :]
    probabilities = probabilities / (probabilities.sum(axis=1, keepdims=True) + 1e-9)
    return probabilities


def _validate_probabilities(*probability_vectors: np.ndarray) -> None:
    class_count = probability_vectors[0].size
    for probabilities in probability_vectors:
        if probabilities.ndim != 1:
            raise ValueError("Each probability vector must be 1D after batch extraction.")
        if probabilities.size != class_count:
            raise ValueError("Both engines must output the same number of classes.")


def _certainty_score(probabilities: np.ndarray) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float32)
    class_count = probabilities.size
    top_two = np.sort(probabilities)[-2:] if class_count >= 2 else np.array([0.0, probabilities[0]])
    margin = float(top_two[-1] - top_two[-2])
    entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-9)))
    normalized_entropy = entropy / np.log(class_count) if class_count > 1 else 0.0
    entropy_certainty = 1.0 - normalized_entropy
    return max(1e-4, 0.55 * entropy_certainty + 0.45 * margin)
