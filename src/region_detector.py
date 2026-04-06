"""
src/region_detector.py
Detects the main text block in a page, ignoring marginalia.

Strategy:
- Project pixel density onto X and Y axes
- Find the largest contiguous dense region (main text block)
- Crop everything outside that region
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


class TextRegionDetector:

    def __init__(self, margin_ratio=0.15):
        """
        margin_ratio: fraction of page width to ignore on left/right
                      as potential marginalia zone
        """
        self.margin_ratio = margin_ratio

    def detect(self, image: np.ndarray, visualize=False):
        """
        Detect main text region.
        Returns cropped image containing only main text block.
        """
        h, w = image.shape[:2]

        # Binary threshold
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        _, binary = cv2.threshold(gray, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Morphological closing to connect text regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return image, (0, 0, w, h)

        # Filter: keep only contours in the central region (ignore margins)
        margin = int(w * self.margin_ratio)
        central_contours = []
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            # Must overlap with central area
            if x + cw > margin and x < w - margin:
                # Must be large enough to be text (not noise)
                if cw * ch > (w * h * 0.001):
                    central_contours.append(c)

        if not central_contours:
            return image, (0, 0, w, h)

        # Bounding box around all central contours = main text block
        all_pts = np.concatenate(central_contours)
        x, y, bw, bh = cv2.boundingRect(all_pts)

        # Add small padding
        pad = 10
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)

        cropped = image[y1:y2, x1:x2]

        if visualize:
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            plt.title('Detected Main Text Region (green box)')
            plt.axis('off')
            plt.show()

        return cropped, (x1, y1, x2, y2)


print('✓ region_detector.py saved')
