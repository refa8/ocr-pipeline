"""
src/metrics.py
OCR Evaluation Metrics.

CER (Character Error Rate):
    CER = edit_distance(predicted, ground_truth) / len(ground_truth)
    Range: 0.0 (perfect) to 1.0+ (completely wrong)

WER (Word Error Rate):
    WER = edit_distance(pred_words, gt_words) / len(gt_words)

Both based on Levenshtein edit distance (insertions, deletions, substitutions).
"""

import editdistance


class Metrics:

    @staticmethod
    def cer(predicted: str, ground_truth: str) -> float:
        if not ground_truth:
            return 0.0 if not predicted else 1.0
        return editdistance.eval(predicted, ground_truth) / len(ground_truth)

    @staticmethod
    def wer(predicted: str, ground_truth: str) -> float:
        pred_words = predicted.split()
        gt_words = ground_truth.split()
        if not gt_words:
            return 0.0 if not pred_words else 1.0
        return editdistance.eval(pred_words, gt_words) / len(gt_words)

    @staticmethod
    def accuracy(predicted: str, ground_truth: str) -> float:
        """Character-level accuracy (complement of CER, capped at 0)."""
        return max(0.0, 1.0 - Metrics.cer(predicted, ground_truth))


print('✓ metrics.py saved')
