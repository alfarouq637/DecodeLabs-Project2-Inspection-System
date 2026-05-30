from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


Array = np.ndarray
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def convolve2d(image: Array, kernel: Array) -> Array:
    """Valid 2D cross-correlation, the operation used by CNN layers."""
    image = np.asarray(image, dtype=np.float32)
    kernel = np.asarray(kernel, dtype=np.float32)
    if image.ndim != 2 or kernel.ndim != 2:
        raise ValueError("convolve2d expects a 2D image and a 2D kernel.")

    kh, kw = kernel.shape
    ih, iw = image.shape
    if kh > ih or kw > iw:
        raise ValueError("Kernel must be smaller than or equal to image.")

    windows = sliding_window_view(image, (kh, kw))
    return np.einsum("hwkl,kl->hw", windows, kernel)


class Layer:
    def forward(self, x: Array) -> Array:
        raise NotImplementedError

    def backward(self, grad: Array) -> Array:
        raise NotImplementedError

    def step(self, learning_rate: float) -> None:
        pass


class Conv2D(Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        rng: np.random.Generator,
    ) -> None:
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        kh, kw = kernel_size
        fan_in = in_channels * kh * kw
        scale = np.sqrt(2.0 / fan_in)
        self.weights = rng.normal(0.0, scale, (out_channels, in_channels, kh, kw)).astype(
            np.float32
        )
        self.bias = np.zeros(out_channels, dtype=np.float32)
        self._x: Array | None = None
        self._windows: Array | None = None
        self._grad_w = np.zeros_like(self.weights)
        self._grad_b = np.zeros_like(self.bias)

    def forward(self, x: Array) -> Array:
        if x.ndim != 4:
            raise ValueError("Conv2D expects input shaped (batch, channels, height, width).")
        kh, kw = self.weights.shape[-2:]
        if x.shape[2] < kh or x.shape[3] < kw:
            raise ValueError("Input is smaller than the convolution kernel.")

        self._x = x
        self._windows = sliding_window_view(x, (kh, kw), axis=(2, 3))
        out = np.einsum("nchwkl,fckl->nfhw", self._windows, self.weights)
        out += self.bias[None, :, None, None]
        return out.astype(np.float32, copy=False)

    def backward(self, grad: Array) -> Array:
        if self._x is None or self._windows is None:
            raise RuntimeError("Conv2D.backward called before forward.")

        kh, kw = self.weights.shape[-2:]
        self._grad_w = np.einsum("nfhw,nchwkl->fckl", grad, self._windows)
        self._grad_b = grad.sum(axis=(0, 2, 3))

        dx = np.zeros_like(self._x)
        for ky in range(kh):
            for kx in range(kw):
                dx[:, :, ky : ky + grad.shape[2], kx : kx + grad.shape[3]] += np.einsum(
                    "nfhw,fc->nchw", grad, self.weights[:, :, ky, kx]
                )
        return dx.astype(np.float32, copy=False)

    def step(self, learning_rate: float) -> None:
        self.weights -= learning_rate * self._grad_w
        self.bias -= learning_rate * self._grad_b


class ReLU(Layer):
    def __init__(self) -> None:
        self._mask: Array | None = None

    def forward(self, x: Array) -> Array:
        self._mask = x > 0
        return np.maximum(x, 0).astype(np.float32, copy=False)

    def backward(self, grad: Array) -> Array:
        if self._mask is None:
            raise RuntimeError("ReLU.backward called before forward.")
        return (grad * self._mask).astype(np.float32, copy=False)


class MaxPool2D(Layer):
    def __init__(self, pool_size: int = 2, stride: int | None = None) -> None:
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        self._input_shape: tuple[int, ...] | None = None
        self._argmax: Array | None = None

    def forward(self, x: Array) -> Array:
        if x.ndim != 4:
            raise ValueError("MaxPool2D expects input shaped (batch, channels, height, width).")
        n, c, h, w = x.shape
        k = self.pool_size
        s = self.stride
        out_h = (h - k) // s + 1
        out_w = (w - k) // s + 1
        if out_h <= 0 or out_w <= 0:
            raise ValueError("Input is smaller than the pooling window.")

        windows = sliding_window_view(x, (k, k), axis=(2, 3))[:, :, ::s, ::s, :, :]
        windows = windows[:, :, :out_h, :out_w, :, :]
        flat_windows = windows.reshape(n, c, out_h, out_w, k * k)
        self._input_shape = x.shape
        self._argmax = flat_windows.argmax(axis=-1)
        return flat_windows.max(axis=-1).astype(np.float32, copy=False)

    def backward(self, grad: Array) -> Array:
        if self._input_shape is None or self._argmax is None:
            raise RuntimeError("MaxPool2D.backward called before forward.")
        n, c, _, _ = self._input_shape
        k = self.pool_size
        s = self.stride
        dx = np.zeros(self._input_shape, dtype=np.float32)
        n_idx = np.arange(n)[:, None]
        c_idx = np.arange(c)[None, :]

        for oy in range(grad.shape[2]):
            for ox in range(grad.shape[3]):
                flat_index = self._argmax[:, :, oy, ox]
                yy = flat_index // k
                xx = flat_index % k
                np.add.at(dx, (n_idx, c_idx, oy * s + yy, ox * s + xx), grad[:, :, oy, ox])
        return dx


class Flatten(Layer):
    def __init__(self) -> None:
        self._input_shape: tuple[int, ...] | None = None

    def forward(self, x: Array) -> Array:
        self._input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad: Array) -> Array:
        if self._input_shape is None:
            raise RuntimeError("Flatten.backward called before forward.")
        return grad.reshape(self._input_shape)


class Dense(Layer):
    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator) -> None:
        scale = np.sqrt(2.0 / in_features)
        self.weights = rng.normal(0.0, scale, (in_features, out_features)).astype(np.float32)
        self.bias = np.zeros(out_features, dtype=np.float32)
        self._x: Array | None = None
        self._grad_w = np.zeros_like(self.weights)
        self._grad_b = np.zeros_like(self.bias)

    def forward(self, x: Array) -> Array:
        if x.ndim != 2:
            raise ValueError("Dense expects input shaped (batch, features).")
        self._x = x
        return (x @ self.weights + self.bias).astype(np.float32, copy=False)

    def backward(self, grad: Array) -> Array:
        if self._x is None:
            raise RuntimeError("Dense.backward called before forward.")
        self._grad_w = self._x.T @ grad
        self._grad_b = grad.sum(axis=0)
        return (grad @ self.weights.T).astype(np.float32, copy=False)

    def step(self, learning_rate: float) -> None:
        self.weights -= learning_rate * self._grad_w
        self.bias -= learning_rate * self._grad_b


class SoftmaxCrossEntropy:
    def __init__(self) -> None:
        self.probabilities: Array | None = None
        self.targets: Array | None = None

    def forward(self, logits: Array, targets: Array) -> float:
        targets = targets.astype(np.int64)
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        self.probabilities = exp / exp.sum(axis=1, keepdims=True)
        self.targets = targets
        losses = -np.log(self.probabilities[np.arange(targets.size), targets] + 1e-9)
        return float(losses.mean())

    def backward(self) -> Array:
        if self.probabilities is None or self.targets is None:
            raise RuntimeError("SoftmaxCrossEntropy.backward called before forward.")
        grad = self.probabilities.copy()
        grad[np.arange(self.targets.size), self.targets] -= 1.0
        return (grad / self.targets.size).astype(np.float32, copy=False)


@dataclass
class TrainingHistory:
    losses: list[float]
    accuracies: list[float]


@dataclass
class DataWorkspace:
    data_dir: Path
    labels_path: Path
    class_counts: dict[str, int]
    unlabeled_count: int
    imported_count: int

    @property
    def labeled_count(self) -> int:
        return sum(self.class_counts.values())


@dataclass(frozen=True)
class GearGeometry:
    center: tuple[float, float]
    max_radius: float
    mask: Array


@dataclass(frozen=True)
class PreprocessConfig:
    use_lab_lightness: bool = True
    use_text_mask: bool = True
    use_polar_unroll: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)
    text_response_percentile: float = 92.0
    polar_radius_margin: float = 1.03


class ScratchCNN:
    def __init__(self, layers: Sequence[Layer], class_names: Sequence[str]) -> None:
        self.layers = list(layers)
        self.class_names = list(class_names)
        self.loss = SoftmaxCrossEntropy()

    def forward(self, x: Array) -> Array:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: Array) -> None:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def step(self, learning_rate: float) -> None:
        for layer in self.layers:
            layer.step(learning_rate)

    def predict_proba(self, x: Array) -> Array:
        logits = self.forward(x)
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32, copy=False)

    def predict(self, x: Array) -> Array:
        return self.predict_proba(x).argmax(axis=1)

    def fit(
        self,
        x: Array,
        y: Array,
        *,
        epochs: int,
        learning_rate: float,
        batch_size: int,
        rng: np.random.Generator,
    ) -> TrainingHistory:
        losses: list[float] = []
        accuracies: list[float] = []
        sample_count = x.shape[0]

        for _ in range(epochs):
            order = rng.permutation(sample_count)
            epoch_losses: list[float] = []
            correct = 0

            for start in range(0, sample_count, batch_size):
                batch_index = order[start : start + batch_size]
                xb = x[batch_index]
                xb = augment_batch(xb, rng)
                yb = y[batch_index]

                logits = self.forward(xb)
                epoch_losses.append(self.loss.forward(logits, yb))
                correct += int((logits.argmax(axis=1) == yb).sum())

                self.backward(self.loss.backward())
                self.step(learning_rate)

            losses.append(float(np.mean(epoch_losses)))
            accuracies.append(correct / sample_count)

        return TrainingHistory(losses=losses, accuracies=accuracies)

    def save(self, path: str | Path) -> None:
        params: dict[str, Array] = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "weights"):
                params[f"{i}_weights"] = layer.weights
                params[f"{i}_bias"] = layer.bias
        np.savez_compressed(path, **params)

    def load(self, path: str | Path) -> None:
        data = np.load(path)
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "weights"):
                layer.weights[...] = data[f"{i}_weights"]
                layer.bias[...] = data[f"{i}_bias"]


def build_gear_cnn(
    input_shape: tuple[int, int, int] = (1, 64, 64),
    class_names: Sequence[str] = ("intact", "defective"),
    seed: int = 7,
) -> ScratchCNN:
    rng = np.random.default_rng(seed)
    layers: list[Layer] = [
        Conv2D(input_shape[0], 12, 5, rng), 
        ReLU(),
        MaxPool2D(2),
        Conv2D(12, 24, 3, rng), 
        ReLU(),
        MaxPool2D(2),
        Flatten(),
    ]

    dummy = np.zeros((1, *input_shape), dtype=np.float32)
    feature_count = dummy.shape[1]
    for layer in layers:
        dummy = layer.forward(dummy)
        feature_count = dummy.reshape(1, -1).shape[1]

    layers.extend(
        [
            Dense(feature_count, 64, rng), 
            ReLU(),
            Dense(64, len(class_names), rng),
        ]
    )
    return ScratchCNN(layers, class_names)


def preprocess_image(
    path: str | Path,
    image_size: tuple[int, int] = (64, 64),
    config: PreprocessConfig | None = None,
) -> Array:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not open image: {path}")
    return preprocess_bgr_for_cnn(image, image_size, config)[None, :, :]


def preprocess_bgr_for_cnn(
    image_bgr: Array,
    image_size: tuple[int, int] = (64, 64),
    config: PreprocessConfig | None = None,
) -> Array:
    """Prepare one gear image for the CNN using geometry-aware preprocessing."""
    config = config or PreprocessConfig()
    lightness = lab_lightness_channel(image_bgr, config)
    geometry = estimate_gear_geometry(lightness, config.polar_radius_margin)

    if config.use_text_mask:
        text_mask = detect_text_mask(lightness, geometry.mask, config)
        lightness = fill_mask_with_local_median(lightness, text_mask)

    if config.use_polar_unroll:
        prepared = unroll_gear_polar(
            lightness,
            geometry.center,
            geometry.max_radius,
            output_size=image_size,
        )
    else:
        prepared = cv2.resize(lightness, image_size, interpolation=cv2.INTER_AREA)

    prepared = prepared.astype(np.float32) / 255.0
    prepared = (prepared - prepared.mean()) / (prepared.std() + 1e-6)
    return prepared.astype(np.float32, copy=False)


def lab_lightness_channel(image_bgr: Array, config: PreprocessConfig) -> Array:
    """Use LAB L for shadow-normalized metal geometry."""
    if not config.use_lab_lightness:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    lightness, _, _ = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=config.clahe_clip_limit,
        tileGridSize=config.clahe_tile_grid_size,
    )
    return clahe.apply(lightness)


def estimate_gear_geometry(lightness: Array, radius_margin: float = 1.03) -> GearGeometry:
    """Estimate gear center/radius from the largest thresholded component."""
    height, width = lightness.shape[:2]
    blurred = cv2.GaussianBlur(lightness, (5, 5), 0)

    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, binary_inv = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = _best_component_mask([binary, binary_inv])

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        center = (width / 2.0, height / 2.0)
        max_radius = min(width, height) * 0.48
        fallback_mask = np.full((height, width), 255, dtype=np.uint8)
        return GearGeometry(center=center, max_radius=max_radius, mask=fallback_mask)

    main_contour = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(main_contour)
    radius *= radius_margin
    radius = min(radius, cx, cy, width - cx, height - cy)
    radius = max(4.0, float(radius))

    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, [main_contour], -1, 255, thickness=cv2.FILLED)
    return GearGeometry(center=(float(cx), float(cy)), max_radius=radius, mask=clean_mask)


def _best_component_mask(candidates: Sequence[Array]) -> Array:
    best_score = -1.0
    best_mask = np.zeros_like(candidates[0])
    image_height, image_width = candidates[0].shape[:2]
    image_area = image_height * image_width

    for binary in candidates:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < image_area * 0.01 or area > image_area * 0.95:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            touches_border = x <= 1 or y <= 1 or x + w >= image_width - 1 or y + h >= image_height - 1
            border_weight = 0.15 if touches_border else 1.0
            cx = x + w / 2.0
            cy = y + h / 2.0
            center_distance = np.hypot(cx - image_width / 2.0, cy - image_height / 2.0)
            center_weight = 1.0 - min(0.75, center_distance / max(image_width, image_height))
            score = area * border_weight * center_weight
            if score > best_score:
                best_score = score
                best_mask = np.zeros_like(binary)
                cv2.drawContours(best_mask, [contour], -1, 255, cv2.FILLED)

    return best_mask


def _largest_component_mask(binary: Array) -> Array:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = np.zeros_like(binary)
    if contours:
        cv2.drawContours(output, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    return output


def detect_text_mask(
    lightness: Array,
    gear_mask: Array | None,
    config: PreprocessConfig,
) -> Array:
    """Detect engraved alphanumeric strokes with small classical filters."""
    height, width = lightness.shape[:2]
    median_size = _odd_at_least(3, min(height, width) // 45)
    smooth = cv2.medianBlur(lightness, median_size)
    residual = cv2.absdiff(lightness, smooth)

    text_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(5, width // 32), max(3, height // 96)),
    )
    blackhat = cv2.morphologyEx(lightness, cv2.MORPH_BLACKHAT, text_kernel)
    tophat = cv2.morphologyEx(lightness, cv2.MORPH_TOPHAT, text_kernel)
    response = cv2.max(residual, cv2.max(blackhat, tophat))

    active_mask = None
    if gear_mask is not None and cv2.countNonZero(gear_mask) > 0:
        erode_size = max(3, _odd_at_least(3, min(height, width) // 24))
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))
        active_mask = cv2.erode(gear_mask, erode_kernel, iterations=1)
        response = cv2.bitwise_and(response, response, mask=active_mask)

    active_pixels = response[active_mask > 0] if active_mask is not None else response.reshape(-1)
    active_pixels = active_pixels[active_pixels > 0]
    if active_pixels.size == 0:
        return np.zeros_like(lightness, dtype=np.uint8)

    otsu_threshold, _ = cv2.threshold(response, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    percentile_threshold = np.percentile(active_pixels, config.text_response_percentile)
    threshold_value = max(float(otsu_threshold), float(percentile_threshold))
    binary = np.where(response >= threshold_value, 255, 0).astype(np.uint8)

    connect_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(3, width // 96), max(3, height // 96)),
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, connect_kernel, iterations=1)
    binary = cv2.dilate(binary, np.ones((3, 3), dtype=np.uint8), iterations=1)
    if active_mask is not None:
        binary = cv2.bitwise_and(binary, binary, mask=active_mask)

    return _filter_text_components(binary)


def _filter_text_components(binary: Array) -> Array:
    height, width = binary.shape[:2]
    image_area = height * width
    min_area = max(4, int(image_area * 0.00002))
    max_area = max(min_area + 1, int(image_area * 0.025))
    mask = np.zeros_like(binary)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        box_area = w * h
        if box_area < min_area or box_area > max_area:
            continue
        if w > width * 0.55 or h > height * 0.25:
            continue
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    return mask


def fill_mask_with_local_median(lightness: Array, mask: Array) -> Array:
    """Replace text pixels using the median value from a local surrounding ring."""
    if cv2.countNonZero(mask) == 0:
        return lightness

    result = lightness.copy()
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    fallback_pixels = lightness[mask == 0]
    fallback = int(np.median(fallback_pixels)) if fallback_pixels.size else int(np.median(lightness))
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    for label in range(1, labels_count):
        area = stats[label, cv2.CC_STAT_AREA]
        if area <= 0:
            continue
        component = (labels == label).astype(np.uint8) * 255
        ring = cv2.dilate(component, ring_kernel, iterations=1)
        ring = cv2.bitwise_and(ring, cv2.bitwise_not(component))
        ring_pixels = lightness[ring > 0]
        fill_value = int(np.median(ring_pixels)) if ring_pixels.size >= 8 else fallback
        result[component > 0] = fill_value

    return result


def unroll_gear_polar(
    lightness: Array,
    center: tuple[float, float],
    max_radius: float,
    output_size: tuple[int, int] = (64, 64),
) -> Array:
    """Unroll a circular gear into a rectangular radius-vs-angle strip.

    OpenCV's polar image has rows as angle and columns as radius. The transpose
    makes angle horizontal, then flipud places the outer teeth at the top.
    """
    output_width, output_height = output_size
    radial_bins = max(8, output_height)
    angular_bins = max(16, output_width)
    flags = cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LINEAR
    polar = cv2.warpPolar(
        lightness,
        (radial_bins, angular_bins),
        center,
        max_radius,
        flags,
    )
    strip = np.flipud(polar.T)
    if strip.shape[:2] != (output_height, output_width):
        strip = cv2.resize(strip, output_size, interpolation=cv2.INTER_AREA)
    return strip


def _odd_at_least(minimum: int, value: int) -> int:
    value = max(minimum, int(value))
    return value if value % 2 == 1 else value + 1



def load_labeled_dataset(
    dataset_dir: str | Path,
    image_size: tuple[int, int] = (64, 64),
    extensions: Iterable[str] = IMAGE_EXTENSIONS,
) -> tuple[Array, Array, list[str]]:
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_path}")

    class_dirs = sorted(path for path in dataset_path.iterdir() if path.is_dir())
    if len(class_dirs) < 2:
        raise ValueError("Expected at least two class folders, for example intact/ and defective/.")

    x_items: list[Array] = []
    y_items: list[int] = []
    extensions = tuple(ext.lower() for ext in extensions)

    for class_index, class_dir in enumerate(class_dirs):
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.suffix.lower() in extensions:
                x_items.append(preprocess_image(image_path, image_size))
                y_items.append(class_index)

    if not x_items:
        raise ValueError(f"No image files found under {dataset_path}.")

    x = np.stack(x_items).astype(np.float32)
    y = np.asarray(y_items, dtype=np.int64)
    return x, y, [path.name for path in class_dirs]


def load_csv_dataset(
    labels_path: str | Path,
    image_size: tuple[int, int] = (64, 64),
) -> tuple[Array, Array, list[str]]:
    labels_file = Path(labels_path)
    if not labels_file.exists():
        raise FileNotFoundError(f"Labels file does not exist: {labels_file}")

    x_items: list[Array] = []
    y_items: list[int] = []
    class_names: list[str] = []
    class_to_index: dict[str, int] = {}

    with labels_file.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        expected = {"filename", "label"}
        if reader.fieldnames is None or not expected.issubset(set(reader.fieldnames)):
            raise ValueError("Labels CSV must contain filename and label columns.")

        for row in reader:
            filename = (row.get("filename") or "").strip()
            label = (row.get("label") or "").strip()
            if not filename or not label:
                continue

            if label not in class_to_index:
                class_to_index[label] = len(class_names)
                class_names.append(label)

            image_path = labels_file.parent / filename
            x_items.append(preprocess_image(image_path, image_size))
            y_items.append(class_to_index[label])

    if len(class_names) < 2:
        raise ValueError("At least two labels are required, for example intact and defective.")
    if not x_items:
        raise ValueError(f"No labeled image rows found in {labels_file}.")

    x = np.stack(x_items).astype(np.float32)
    y = np.asarray(y_items, dtype=np.int64)
    return x, y, class_names


def prepare_data_workspace(
    data_dir: str | Path = "gear_data",
    *,
    class_names: Sequence[str] = ("intact", "defective"),
    unlabeled_name: str = "unlabeled",
    labels_filename: str = "labels.csv",
    import_from: str | Path | None = None,
    extensions: Iterable[str] = IMAGE_EXTENSIONS,
) -> DataWorkspace:
    """Create the managed data folders and regenerate labels.csv.

    Images placed directly inside data_dir are moved to the unlabeled folder.
    Images imported from another folder are copied only when that filename is
    not already present anywhere in the managed data workspace.
    """
    data_path = Path(data_dir)
    extensions = tuple(ext.lower() for ext in extensions)
    data_path.mkdir(parents=True, exist_ok=True)

    class_dirs = {name: data_path / name for name in class_names}
    unlabeled_dir = data_path / unlabeled_name
    for folder in [*class_dirs.values(), unlabeled_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    imported_count = 0
    if import_from is not None:
        source_dir = Path(import_from)
        if source_dir.exists():
            resolved_data_path = data_path.resolve()
            for image_path in sorted(source_dir.iterdir()):
                if not image_path.is_file() or image_path.suffix.lower() not in extensions:
                    continue
                if resolved_data_path in image_path.resolve().parents:
                    continue
                if _filename_exists(data_path, image_path.name, extensions):
                    continue
                destination = _unique_destination(unlabeled_dir / image_path.name)
                shutil.copy2(image_path, destination)
                imported_count += 1

    for image_path in sorted(data_path.iterdir()):
        if image_path.is_file() and image_path.suffix.lower() in extensions:
            destination = _unique_destination(unlabeled_dir / image_path.name)
            image_path.replace(destination)

    rows: list[tuple[str, str]] = []
    class_counts: dict[str, int] = {}
    for class_name, class_dir in class_dirs.items():
        image_paths = _image_files(class_dir, extensions)
        class_counts[class_name] = len(image_paths)
        rows.extend((path.relative_to(data_path).as_posix(), class_name) for path in image_paths)

    unlabeled_paths = _image_files(unlabeled_dir, extensions)
    rows.extend((path.relative_to(data_path).as_posix(), "") for path in unlabeled_paths)

    labels_path = data_path / labels_filename
    with labels_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "label"])
        writer.writerows(rows)

    return DataWorkspace(
        data_dir=data_path,
        labels_path=labels_path,
        class_counts=class_counts,
        unlabeled_count=len(unlabeled_paths),
        imported_count=imported_count,
    )


def _image_files(folder: Path, extensions: tuple[str, ...]) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def _unique_destination(destination: Path) -> Path:
    if not destination.exists():
        return destination

    stem = destination.stem
    suffix = destination.suffix
    parent = destination.parent
    index = 2
    while True:
        candidate = parent / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def _filename_exists(folder: Path, filename: str, extensions: tuple[str, ...]) -> bool:
    return any(
        path.name == filename
        for path in _image_files(folder, extensions)
    )


def make_toy_gear_dataset(
    samples_per_class: int = 24,
    image_size: int = 64,
    seed: int = 11,
) -> tuple[Array, Array, list[str]]:
    rng = np.random.default_rng(seed)
    images: list[Array] = []
    labels: list[int] = []

    for label in (0, 1):
        for _ in range(samples_per_class):
            images.append(_render_toy_gear(image_size, defective=bool(label), rng=rng))
            labels.append(label)

    x = np.stack(images).astype(np.float32)[:, None, :, :]
    y = np.asarray(labels, dtype=np.int64)
    order = rng.permutation(y.size)
    return x[order], y[order], ["intact", "defective"]


def _render_toy_gear(image_size: int, *, defective: bool, rng: np.random.Generator) -> np.ndarray:
    canvas_size = int(image_size * 1.5)
    yy, xx = np.mgrid[:canvas_size, :canvas_size]
    center = canvas_size / 2.0
    
    y = yy - center
    x = xx - center
    radius = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)

    
    tooth_count = rng.integers(16, 32)
    base_radius = canvas_size * rng.uniform(0.22, 0.28)
    tooth_length = canvas_size * rng.uniform(0.02, 0.05)
    outer_radius = base_radius + tooth_length

    
    tooth_wave = np.sin(tooth_count * theta)
    body = radius <= base_radius
    teeth = (radius <= outer_radius) & (tooth_wave > 0.3)
    gear_mask = (body | teeth).astype(np.float32)

    
    if defective:
        missing_tooth = rng.integers(0, tooth_count)
        tooth_angle = ((theta + np.pi) / (2 * np.pi) * tooth_count) % tooth_count
        distance = np.minimum(
            np.abs(tooth_angle - missing_tooth),
            tooth_count - np.abs(tooth_angle - missing_tooth),
        )
        defect_width = rng.uniform(0.3, 0.8) 
        defect_depth = rng.uniform(0.7, 0.95)
        gear_mask *= ~((distance < defect_width) & (radius > base_radius * defect_depth))

    
    inner_rad = base_radius * rng.uniform(0.3, 0.5)
    gear_mask *= (radius > inner_rad) 
    
    
    if rng.random() > 0.5:
        ring_rad = base_radius * rng.uniform(0.6, 0.8)
        gear_mask += (radius <= ring_rad) * (radius > ring_rad - 5) * rng.uniform(-0.2, 0.2)

    
    gear_mask = np.clip(gear_mask, 0.0, 1.0)
    gear_mask = cv2.GaussianBlur(gear_mask, (3, 3), 0)
    top_face = gear_mask.copy()

    
    if rng.random() > 0.3:
        texts = ["DL-P2", "STEEL 24T", "MOD 1.0", "UNIT 7", "EDGE AI"]
        txt = rng.choice(texts)
        cv2.putText(top_face, txt, (int(center - 45), int(center - base_radius * 0.5)), 
                    cv2.FONT_HERSHEY_SIMPLEX, rng.uniform(0.4, 0.6), rng.uniform(0.2, 0.6), 1, cv2.LINE_AA)

    
    tilt_top = canvas_size * rng.uniform(0.05, 0.20)
    tilt_side = canvas_size * rng.uniform(-0.15, 0.15)
    
    src_pts = np.float32([[0, 0], [canvas_size, 0], [canvas_size, canvas_size], [0, canvas_size]])
    dst_pts = np.float32([
        [abs(tilt_side), tilt_top], 
        [canvas_size - abs(tilt_side), tilt_top], 
        [canvas_size, canvas_size], 
        [0, canvas_size]
    ])
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_mask = cv2.warpPerspective(gear_mask, M, (canvas_size, canvas_size))
    warped_top = cv2.warpPerspective(top_face, M, (canvas_size, canvas_size))

    
    thickness = rng.integers(6, 16)
    canvas_3d = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    
    for d in range(thickness, 0, -1):
        M_trans = np.float32([[1, 0, 0], [0, 1, d]])
        shifted = cv2.warpAffine(warped_mask, M_trans, (canvas_size, canvas_size))
        shade = 0.3 + (0.4 * (thickness - d) / thickness) 
        canvas_3d = np.maximum(canvas_3d, shifted * shade)
        
   
    canvas_3d = np.where(warped_mask > 0.1, warped_top, canvas_3d)

    
    start = (canvas_size - image_size) // 2
    final_img = canvas_3d[start:start+image_size, start:start+image_size]

    
    is_extreme = rng.random() < 0.20 
    
    if is_extreme:
        base_bright = rng.uniform(0.3, 0.5)
        noise_level = rng.uniform(0.05, 0.15)
        glare_int = rng.uniform(0.2, 0.5)
    else:
        base_bright = rng.uniform(0.6, 0.9)
        noise_level = rng.uniform(0.01, 0.04)
        glare_int = rng.uniform(0.0, 0.15)

    yy, xx = np.mgrid[:image_size, :image_size]
    gradient = rng.uniform(-0.2, 0.2) * (xx / image_size) + rng.uniform(-0.2, 0.2) * (yy / image_size)
    noise = rng.normal(0, noise_level, (image_size, image_size))
    
    glare_angle = rng.uniform(-np.pi, np.pi)
    glare = np.exp(-((np.cos(glare_angle)*(xx-image_size/2) + np.sin(glare_angle)*(yy-image_size/2)) / 15.0) ** 2) * glare_int

    final_img = final_img * base_bright + gradient + noise + glare
    final_img = np.clip(final_img, 0.0, 1.0)
    final_img = (final_img - final_img.mean()) / (final_img.std() + 1e-6)
    
    return final_img.astype(np.float32)

def augment_batch(x_batch: Array, rng: np.random.Generator) -> Array:
    
    augmented = np.empty_like(x_batch)
    for i in range(x_batch.shape[0]):
        img = x_batch[i, 0].copy()
        
        
        if rng.random() > 0.5:
            img = np.fliplr(img)
        if rng.random() > 0.5:
            img = np.flipud(img)
        if rng.random() > 0.5:
            img = np.rot90(img, k=rng.integers(1, 4))
            
        
        brightness_shift = rng.uniform(-0.5, 0.5)
        img = img + brightness_shift
        
        
        augmented[i, 0] = np.clip(img, -3.0, 3.0)
        
    return augmented
