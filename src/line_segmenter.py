"""
src/line_segmenter.py
Splits a text block image into individual text lines.
Uses horizontal projection profile (pixel density per row).
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List


class LineSegmenter:

    def __init__(self, min_line_height=15, gap_threshold=5):
        self.min_line_height = min_line_height
        self.gap_threshold = gap_threshold

    def segment(self, image: np.ndarray, visualize=False) -> List[np.ndarray]:
        """
        Segment image into text line strips.
        Returns list of line images.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Binarize
        _, binary = cv2.threshold(gray, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Horizontal projection: sum of dark pixels per row
        h_proj = np.sum(binary, axis=1)

        # Smooth the projection
        h_proj_smooth = np.convolve(h_proj, np.ones(3)/3, mode='same')

        # Find text rows (where projection > threshold)
        threshold = np.max(h_proj_smooth) * 0.05
        is_text_row = h_proj_smooth > threshold

        # Find line boundaries
        lines = []
        in_line = False
        start = 0
        gap_count = 0

        for row_idx, is_text in enumerate(is_text_row):
            if is_text:
                if not in_line:
                    start = row_idx
                    in_line = True
                gap_count = 0
            else:
                if in_line:
                    gap_count += 1
                    if gap_count >= self.gap_threshold:
                        end = row_idx - gap_count
                        if end - start >= self.min_line_height:
                            lines.append((start, end))
                        in_line = False
                        gap_count = 0

        # Don't forget last line
        if in_line:
            end = len(is_text_row)
            if end - start >= self.min_line_height:
                lines.append((start, end))

        # Crop line images with small vertical padding
        pad = 3
        h, w = image.shape[:2]
        line_images = []
        for (y1, y2) in lines:
            y1p = max(0, y1 - pad)
            y2p = min(h, y2 + pad)
            line_img = gray[y1p:y2p, :]
            line_images.append(line_img)

        if visualize and lines:
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for (y1, y2) in lines:
                cv2.rectangle(vis, (0, y1), (w, y2), (0, 200, 0), 1)
            plt.figure(figsize=(14, 10))
            plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            plt.title(f'Line Segmentation: {len(lines)} lines detected')
            plt.axis('off')
            plt.show()

        return line_images, lines


print('✓ line_segmenter.py saved')
