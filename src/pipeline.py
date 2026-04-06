"""
src/pipeline.py
Full OCR pipeline: Preprocess -> Region Detect -> Line Segment -> CRNN -> LLM
"""

import sys
sys.path.insert(0, '/content/drive/MyDrive/OCR_Pipeline_Research')

import cv2
import torch
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict

from src.preprocess import ImagePreprocessor
from src.region_detector import TextRegionDetector
from src.line_segmenter import LineSegmenter
from src.crnn_model import CRNN
from src.charset import decode, VOCAB_SIZE
from src.llm_corrector import LLMCorrector

logger = logging.getLogger(__name__)


class OCRPipeline:

    def __init__(self, model_path: str, use_llm=True,
                 llm_provider='gemini', device=None):

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.preprocessor = ImagePreprocessor(target_height=64)
        self.region_detector = TextRegionDetector(margin_ratio=0.15)
        self.line_segmenter = LineSegmenter(min_line_height=15, gap_threshold=5)

        # Load trained CRNN
        self.model = CRNN(vocab_size=VOCAB_SIZE).to(self.device)
        if Path(model_path).exists():
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
            print(f'✓ CRNN model loaded from {model_path}')
        else:
            print(f'⚠ Model not found at {model_path} — using untrained model')
        self.model.eval()

        self.llm = LLMCorrector(provider=llm_provider) if use_llm else None

    def process_page(self, image_path: str, visualize=False) -> Dict:
        """
        Process a full page image through the complete pipeline.
        Returns dict with per-stage text output.
        """
        # Stage 1: Preprocess
        preprocessed, _ = self.preprocessor.process(image_path)

        # Stage 2: Detect main text region (ignore marginalia)
        text_region, bbox = self.region_detector.detect(
            preprocessed, visualize=visualize
        )

        # Stage 3: Segment into lines
        line_images, line_coords = self.line_segmenter.segment(
            text_region, visualize=visualize
        )

        if not line_images:
            return {'raw': '', 'llm': '', 'lines': []}

        # Stage 4: CRNN inference on each line
        raw_lines = []
        for line_img in line_images:
            text = self._recognize_line(line_img)
            raw_lines.append(text)

        raw_text = '\n'.join(raw_lines)

        # Stage 5: LLM post-correction
        llm_text = raw_text
        if self.llm:
            llm_text = self.llm.correct(raw_text)

        return {
            'raw': raw_text,
            'llm': llm_text,
            'lines': raw_lines,
            'num_lines': len(line_images),
            'text_bbox': bbox,
        }

    def _recognize_line(self, line_img: np.ndarray) -> str:
        """Run CRNN inference on a single text line image."""
        # Resize to CRNN input height
        line_img = self.preprocessor.resize_for_crnn(line_img)

        # Normalize + tensorize
        img_tensor = torch.tensor(
            line_img.astype(np.float32) / 255.0
        ).unsqueeze(0).unsqueeze(0).to(self.device)   # (1, 1, H, W)

        with torch.no_grad():
            log_probs = self.model(img_tensor)   # (T, 1, vocab)
            pred = log_probs.argmax(2).squeeze(1)  # (T,)

        return decode(pred.cpu().tolist())


print('✓ pipeline.py saved')
