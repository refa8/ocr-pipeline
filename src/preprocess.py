"""
src/preprocess.py
Image preprocessing for historical OCR.
Pipeline: grayscale -> denoise -> deskew -> CLAHE -> resize
"""

import cv2
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:

    def __init__(self, target_height=64):
        # CRNN expects fixed height, variable width
        self.target_height = target_height

    def process(self, image_path: str):
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f'Cannot read: {image_path}')

        steps = {}

        # 1. Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        steps['gray'] = gray.copy()

        # 2. Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=10,
                                             templateWindowSize=7,
                                             searchWindowSize=21)
        steps['denoised'] = denoised.copy()

        # 3. Deskew
        deskewed = self._deskew(denoised)
        steps['deskewed'] = deskewed.copy()

        # 4. CLAHE contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(deskewed)
        steps['enhanced'] = enhanced.copy()

        return enhanced, steps

    def _deskew(self, image: np.ndarray) -> np.ndarray:
        _, binary = cv2.threshold(image, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(binary > 0))
        if len(coords) < 10:
            return image
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        if abs(angle) < 0.5:
            return image
        h, w = image.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(image, M, (w, h),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)

    def resize_for_crnn(self, image: np.ndarray) -> np.ndarray:
        """Resize to fixed height (CRNN input), preserve aspect ratio."""
        h, w = image.shape[:2]
        scale = self.target_height / h
        new_w = max(1, int(w * scale))
        return cv2.resize(image, (new_w, self.target_height),
                          interpolation=cv2.INTER_CUBIC)


print('✓ preprocess.py saved')
