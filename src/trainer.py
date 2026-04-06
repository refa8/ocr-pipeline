"""
src/trainer.py
Training loop for CRNN with Weighted CTC loss.
"""

import sys
sys.path.insert(0, '/content/drive/MyDrive/OCR_Pipeline_Research')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import editdistance
import matplotlib.pyplot as plt
from pathlib import Path
import logging

from src.crnn_model import CRNN
from src.dataset import OCRDataset, collate_fn
from src.charset import decode, VOCAB_SIZE, char_to_idx
from src.weighted_ctc import WeightedCTCLoss, compute_char_weights
from src.beam_decoder import BeamDecoder, build_spanish_lexicon

logger = logging.getLogger(__name__)


class CRNNTrainer:

    def __init__(self, labels_file, image_dir, save_dir,
                 hidden_size=256, num_rnn_layers=2,
                 batch_size=16, lr=1e-3, device=None,
                 use_weighted_loss=True, use_beam_decode=True):

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.use_beam_decode = use_beam_decode

        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f'Using device: {self.device}')

        # Dataset
        full_dataset = OCRDataset(labels_file, image_dir, augment=True)
        val_size = max(1, int(0.2 * len(full_dataset)))
        train_size = len(full_dataset) - val_size
        train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
        val_ds.dataset.augment = False

        self.train_loader = DataLoader(
            train_ds, batch_size=batch_size,
            shuffle=True, collate_fn=collate_fn, num_workers=0
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=batch_size,
            shuffle=False, collate_fn=collate_fn, num_workers=0
        )
        print(f'Train: {train_size} | Val: {val_size} samples')

        # Model
        self.model = CRNN(
            vocab_size=VOCAB_SIZE,
            hidden_size=hidden_size,
            num_rnn_layers=num_rnn_layers
        ).to(self.device)

        # ── Weighted CTC Loss ──────────────────────────────────────────
        if use_weighted_loss:
            print('Computing character weights for rare letterforms...')
            char_weights = compute_char_weights(labels_file, char_to_idx)
            self.criterion = WeightedCTCLoss(char_weights).to(self.device)
            print('✓ Using WeightedCTCLoss (rare chars boosted)')
        else:
            self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)
            print('✓ Using standard CTCLoss')

        # ── Spanish Lexicon for Beam Decoder ──────────────────────────
        if use_beam_decode:
            print('Building Renaissance Spanish lexicon...')
            self.vocabulary = build_spanish_lexicon(labels_file)
            from src.charset import idx_to_char
            self.beam_decoder = BeamDecoder(
                idx_to_char=idx_to_char,
                vocabulary=self.vocabulary,
                beam_width=10,
                lm_weight=0.5
            )
            print('✓ Beam decoder ready')
        else:
            self.beam_decoder = None

        # Optimizer + scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=1e-4
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5
        )
        self.history = {'train_loss': [], 'val_cer_greedy': [], 'val_cer_beam': []}

    def train(self, num_epochs=50):
        best_cer = float('inf')

        for epoch in range(1, num_epochs + 1):
            train_loss = self._train_epoch()
            greedy_cer, beam_cer = self._validate()

            self.history['train_loss'].append(train_loss)
            self.history['val_cer_greedy'].append(greedy_cer)
            self.history['val_cer_beam'].append(beam_cer)
            self.scheduler.step(beam_cer)

            print(f'Epoch [{epoch:3d}/{num_epochs}] '
                  f'Loss: {train_loss:.4f} | '
                  f'Greedy CER: {greedy_cer:.4f} | '
                  f'Beam CER: {beam_cer:.4f}')

            # Save best on beam CER
            if beam_cer < best_cer:
                best_cer = beam_cer
                torch.save(self.model.state_dict(),
                           self.save_dir / 'best_crnn.pth')
                print(f'  ✓ Saved best model (Beam CER: {best_cer:.4f})')

        self._plot_training()
        return best_cer

    def _train_epoch(self):
        print("🚀 Entered training loop")

        self.model.train()
        total_loss = 0
        print("➡️ Starting batch loop")
        for images, labels, label_lengths, _ in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            log_probs = self.model(images)
            T, B, _ = log_probs.shape
            input_lengths = torch.full((B,), T, dtype=torch.long).to(self.device)

            loss = self.criterion(log_probs, labels, input_lengths, label_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def _validate(self):
        self.model.eval()
        greedy_cer_total = 0
        beam_cer_total = 0
        count = 0

        with torch.no_grad():
            for images, _, _, texts in self.val_loader:
                images = images.to(self.device)
                log_probs = self.model(images)   # (T, B, vocab)

                # Greedy decode
                preds_greedy = log_probs.argmax(2).permute(1, 0)
                for pred, gt in zip(preds_greedy, texts):
                    pred_text = decode(pred.cpu().tolist())
                    greedy_cer_total += editdistance.eval(pred_text, gt) / max(len(gt), 1)

                # Beam decode (with lexicon constraint)
                if self.beam_decoder:
                    beam_preds = self.beam_decoder.decode_batch(log_probs)
                    for pred_text, gt in zip(beam_preds, texts):
                        beam_cer_total += editdistance.eval(pred_text, gt) / max(len(gt), 1)
                else:
                    beam_cer_total = greedy_cer_total

                count += len(texts)

        n = max(count, 1)
        return greedy_cer_total / n, beam_cer_total / n

    def _plot_training(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.set_title('Training Loss (Weighted CTC)')
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
        ax1.legend(); ax1.grid(True)

        ax2.plot(self.history['val_cer_greedy'], label='Greedy CER', linestyle='--')
        ax2.plot(self.history['val_cer_beam'],   label='Beam CER (lexicon)', linewidth=2)
        ax2.set_title('Validation CER: Greedy vs Beam Search')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('CER')
        ax2.legend(); ax2.grid(True)

        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=150)
        plt.show()
        print('✓ Training curves saved')


print('✓ trainer.py saved')
