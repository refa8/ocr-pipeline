"""
src/crnn_model.py
CRNN: Convolutional Recurrent Neural Network for OCR

Architecture:
  Input image (1 x H x W)  — grayscale, fixed height=64
       ↓
  CNN Feature Extractor     — VGG-style: Conv→BN→ReLU→Pool (x5)
       ↓
  Feature Map (C x 1 x T)   — T = W/16 time steps
       ↓
  Reshape → Sequence (T x C)
       ↓
  BiLSTM (2 layers)          — captures left+right context
       ↓
  Linear → logits (T x vocab)
       ↓
  CTC Decoder                — produces final character sequence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ConvBlock(nn.Module):
    """Conv2d + BatchNorm + ReLU block"""

    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class CNN(nn.Module):
    """
    VGG-style CNN feature extractor.
    Input:  (B, 1, 64, W)
    Output: (B, 512, 1, W//16)
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1 — (B,1,64,W) -> (B,64,32,W/2)
            ConvBlock(1, 64),
            nn.MaxPool2d(2, 2),

            # Block 2 — (B,64,32,W/2) -> (B,128,16,W/4)
            ConvBlock(64, 128),
            nn.MaxPool2d(2, 2),

            # Block 3 — (B,128,16,W/4) -> (B,256,8,W/4)
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            nn.MaxPool2d((2, 1), (2, 1)),   # pool only height

            # Block 4 — (B,256,8,W/4) -> (B,512,4,W/4)
            ConvBlock(256, 512),
            ConvBlock(512, 512),
            nn.MaxPool2d((2, 1), (2, 1)),   # pool only height

            # Block 5 — (B,512,4,W/4) -> (B,512,1,W/4)
            ConvBlock(512, 512, kernel=(4, 1), padding=0),
        )

    def forward(self, x):
        return self.features(x)


class CRNN(nn.Module):
    """
    Full CRNN model.

    Args:
        vocab_size: number of unique characters + 1 (CTC blank)
        hidden_size: BiLSTM hidden units (per direction)
        num_rnn_layers: number of stacked BiLSTM layers
    """

    def __init__(self, vocab_size: int, hidden_size: int = 256,
                 num_rnn_layers: int = 2, dropout: float = 0.3):
        super().__init__()

        self.cnn = CNN()
        cnn_out_channels = 512

        self.rnn = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=hidden_size,
            num_layers=num_rnn_layers,
            bidirectional=True,
            dropout=dropout if num_rnn_layers > 1 else 0,
            batch_first=False        # (T, B, C) convention
        )

        # Output: BiLSTM gives hidden_size*2
        self.classifier = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, 64, W)
        returns: (T, B, vocab_size) log-softmax output for CTC
        """
        # CNN features: (B, 512, 1, T)
        features = self.cnn(x)

        # Squeeze height dim: (B, 512, T)
        features = features.squeeze(2)

        # Permute to (T, B, 512) for LSTM
        features = features.permute(2, 0, 1)

        # BiLSTM: (T, B, hidden*2)
        rnn_out, _ = self.rnn(features)

        # Classifier: (T, B, vocab_size)
        logits = self.classifier(rnn_out)

        # Log-softmax for CTC loss
        return F.log_softmax(logits, dim=2)


print('✓ crnn_model.py saved')
