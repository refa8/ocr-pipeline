"""
src/dataset.py
PyTorch Dataset for CRNN training.
Expects: data/lines/<image_name>.png + data/annotations/labels.txt

labels.txt format (tab-separated):
    line_001.png\tEl rey don Alfonso
    line_002.png\tque era muy pequeño
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from src.charset import encode, VOCAB_SIZE
import sys
sys.path.insert(0, '/content/drive/MyDrive/OCR_Pipeline_Research')


class OCRDataset(Dataset):

    def __init__(self, labels_file: str, image_dir: str,
                 target_height=64, target_width=512, augment=False):
        self.image_dir = Path(image_dir)
        self.target_height = target_height
        self.target_width = target_width
        self.augment = augment
        self.samples = []

        with open(labels_file, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    self.samples.append((parts[0].strip(), parts[1].strip()))

        print(f'Dataset loaded: {len(self.samples)} samples')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, text = self.samples[idx]
        if filename.startswith("/") or filename.startswith("content"):
            img_path = Path("/" + filename.lstrip("/"))  # normalize path
        else:
            img_path = self.image_dir / filename
        print("Loading:", img_path)

        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            # Return blank image if file missing
            image = np.ones((self.target_height, self.target_width),
                            dtype=np.uint8) * 255

        image = self._resize(image)

        if self.augment:
            image = self._augment(image)

        # Normalize to [0, 1], add channel dim
        image = image.astype(np.float32) / 255.0
        image = torch.tensor(image).unsqueeze(0)   # (1, H, W)

        label = encode(text)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return image, label_tensor, text

    def _resize(self, image):
        h, w = image.shape
        scale = self.target_height / h
        new_w = int(w * scale)
        image = cv2.resize(image, (new_w, self.target_height))

        # Pad or crop to target width
        if new_w < self.target_width:
            pad = self.target_width - new_w
            image = np.pad(image, ((0, 0), (0, pad)),
                           constant_values=255)
        else:
            image = image[:, :self.target_width]
        return image

    def _augment(self, image):
        """Light augmentation for historical docs — no flipping."""
        import random
        # Random brightness
        delta = random.randint(-20, 20)
        image = np.clip(image.astype(np.int32) + delta, 0, 255).astype(np.uint8)
        # Random Gaussian noise
        if random.random() < 0.3:
            noise = np.random.normal(0, 5, image.shape).astype(np.int32)
            image = np.clip(image.astype(np.int32) + noise, 0, 255).astype(np.uint8)
        return image


def collate_fn(batch):
    """Custom collate to handle variable-length labels for CTC."""
    images, labels, texts = zip(*batch)
    images = torch.stack(images, 0)
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels_concat = torch.cat(labels)
    return images, labels_concat, label_lengths, texts


print('✓ dataset.py saved')
