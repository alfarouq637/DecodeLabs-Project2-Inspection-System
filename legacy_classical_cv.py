from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def analyze_gear_topology(image_path: str | Path) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Legacy classical-CV inspection pipeline kept for comparison."""
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: could not open image: {image_path}")
        return None, None

    display_image = image.copy()
    height, width = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    _, threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return display_image, threshold

    main_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(main_contour)
    margin = 10

    is_partial = x < margin or y < margin or (x + w) > (width - margin) or (y + h) > (height - margin)
    status_text = "WARNING: PARTIAL GEAR DETECTED" if is_partial else "STATUS: COMPLETE GEAR"
    box_color = (0, 0, 255) if is_partial else (255, 0, 0)
    text_color = (0, 0, 255) if is_partial else (0, 255, 0)

    cv2.putText(display_image, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
    cv2.rectangle(display_image, (x, y), (x + w, y + h), box_color, 2)

    rect = cv2.minAreaRect(main_contour)
    angle = normalize_min_area_rect_angle(rect[2])
    cv2.putText(
        display_image,
        f"ANGLE: {abs(angle):.2f} DEG",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
    )

    cv2.drawContours(display_image, [main_contour], -1, (0, 255, 0), 2)
    hull_points = cv2.convexHull(main_contour)
    cv2.drawContours(display_image, [hull_points], -1, (0, 0, 255), 2)
    return display_image, threshold


def normalize_min_area_rect_angle(angle: float) -> float:
    """Normalize OpenCV minAreaRect angle into a compact -45..45-ish range."""
    if angle < -45:
        return angle + 90
    return angle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the legacy classical-CV gear pipeline.")
    parser.add_argument("image", nargs="?", default="test_gear5.png", help="Image to inspect.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyzed_image, binary_image = analyze_gear_topology(args.image)
    if analyzed_image is None or binary_image is None:
        return

    cv2.namedWindow("Topological Analysis", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Topological Analysis", 800, 600)
    cv2.imshow("Topological Analysis", analyzed_image)

    cv2.namedWindow("Computer Vision Threshold", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Computer Vision Threshold", 800, 600)
    cv2.imshow("Computer Vision Threshold", binary_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
