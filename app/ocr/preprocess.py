from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(slots=True)
class PreprocessResult:
    image: np.ndarray
    metadata: dict[str, Any]


class ReceiptPreprocessor:
    def __init__(self, use_perspective_fix: bool = True) -> None:
        self.use_perspective_fix = use_perspective_fix

    def load_image(self, file_path: str | Path) -> np.ndarray:
        file_path = str(file_path)
        image = cv2.imread(file_path)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {file_path}")
        return image

    def preprocess(self, image: np.ndarray) -> PreprocessResult:
        original_shape = image.shape
        image = self._resize(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast = self._boost_contrast(gray)
        denoised = cv2.fastNlMeansDenoising(contrast, None, 15, 7, 21)
        fixed = self._perspective_fix(denoised) if self.use_perspective_fix else denoised
        binary = self._adaptive_threshold(fixed)
        cleaned = self._morph_cleanup(binary)
        deskewed, angle = self._deskew(cleaned)
        cropped = self._crop_receipt(deskewed)
        return PreprocessResult(
            image=cropped,
            metadata={
                "original_shape": original_shape,
                "resized_shape": image.shape,
                "deskew_angle": angle,
            },
        )

    def _resize(self, image: np.ndarray, min_width: int = 1400) -> np.ndarray:
        h, w = image.shape[:2]
        if w >= min_width:
            return image
        scale = min_width / max(w, 1)
        return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    def _boost_contrast(self, gray: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def _adaptive_threshold(self, gray: np.ndarray) -> np.ndarray:
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )

    def _morph_cleanup(self, image: np.ndarray) -> np.ndarray:
        kernel = np.ones((1, 1), np.uint8)
        image = cv2.medianBlur(image, 3)
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        return image

    def _deskew(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        inv = cv2.bitwise_not(image)
        coords = np.column_stack(np.where(inv > 0))
        if coords.size == 0:
            return image, 0.0
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = 90 + angle
        angle = -angle
        if abs(angle) < 0.3:
            return image, 0.0
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image,
            matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated, float(angle)

    def _crop_receipt(self, image: np.ndarray) -> np.ndarray:
        inv = cv2.bitwise_not(image)
        coords = cv2.findNonZero(inv)
        if coords is None:
            return image
        x, y, w, h = cv2.boundingRect(coords)
        pad = 15
        x0 = max(x - pad, 0)
        y0 = max(y - pad, 0)
        x1 = min(x + w + pad, image.shape[1])
        y1 = min(y + h + pad, image.shape[0])
        return image[y0:y1, x0:x1]

    def _perspective_fix(self, gray: np.ndarray) -> np.ndarray:
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return gray
        contour = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) != 4:
            return gray
        pts = approx.reshape(4, 2).astype(np.float32)
        ordered = self._order_points(pts)
        width_a = np.linalg.norm(ordered[2] - ordered[3])
        width_b = np.linalg.norm(ordered[1] - ordered[0])
        height_a = np.linalg.norm(ordered[1] - ordered[2])
        height_b = np.linalg.norm(ordered[0] - ordered[3])
        max_w = int(max(width_a, width_b))
        max_h = int(max(height_a, height_b))
        if max_w < 200 or max_h < 300:
            return gray
        dst = np.array([[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(ordered, dst)
        return cv2.warpPerspective(gray, matrix, (max_w, max_h))

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect
