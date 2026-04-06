"""
src/beam_decoder.py
Constrained Beam Search Decoder with Renaissance Spanish Lexicon.

Why: Greedy CTC decoding picks the single most likely char at each step,
     which ignores word-level context and produces hallucinated sequences.
     Beam search explores multiple hypotheses simultaneously and uses a
     lexicon to constrain outputs to real Spanish words.

Strategy:
  1. Run standard CTC beam search (width=10) on CRNN log-probs
  2. At word boundaries, score each hypothesis against the Spanish lexicon
  3. Heavily penalize hypotheses containing out-of-vocabulary words
  4. Return the highest-scoring valid sequence
"""

import torch
import math
from typing import List, Set, Dict, Tuple
from collections import defaultdict


class BeamDecoder:
    """
    CTC Beam Search Decoder with lexicon constraint.

    Args:
        idx_to_char:    mapping from index to character
        vocabulary:     set of valid Spanish words (lexicon)
        beam_width:     number of beams to maintain
        lm_weight:      weight for lexicon scoring (0 = pure CTC)
        blank_idx:      CTC blank token index
    """

    def __init__(self, idx_to_char: dict, vocabulary: Set[str],
                 beam_width: int = 10, lm_weight: float = 0.5,
                 blank_idx: int = 0):
        self.idx_to_char = idx_to_char
        self.vocabulary = {w.lower() for w in vocabulary}
        self.beam_width = beam_width
        self.lm_weight = lm_weight
        self.blank_idx = blank_idx

    def decode(self, log_probs: torch.Tensor) -> str:
        """
        Decode a single sequence.

        Args:
            log_probs: (T, vocab_size) log-softmax probabilities

        Returns:
            decoded string
        """
        T, vocab_size = log_probs.shape
        probs = log_probs.exp().cpu().numpy()

        # Beam: dict of {sequence: (score, last_char)}
        # sequence is tuple of char indices (no blanks, no repeats yet)
        beams = {(): (0.0, self.blank_idx)}

        for t in range(T):
            new_beams = defaultdict(lambda: -float("inf"))

            for seq, (score, last_char) in beams.items():
                for c in range(vocab_size):
                    p = float(probs[t, c])
                    if p < 1e-10:
                        continue
                    log_p = math.log(p)

                    if c == self.blank_idx:
                        # Blank: keep sequence, reset last_char
                        key = (seq, self.blank_idx)
                        candidate = score + log_p
                        if candidate > new_beams[seq]:
                            new_beams[seq] = candidate
                    elif c == last_char:
                        # Repeat without blank: skip (CTC rule)
                        continue
                    else:
                        new_seq = seq + (c,)
                        candidate = score + log_p
                        if candidate > new_beams[new_seq]:
                            new_beams[new_seq] = candidate

            # Keep top beam_width beams
            sorted_beams = sorted(new_beams.items(),
                                  key=lambda x: x[1], reverse=True)
            beams = {}
            for seq, score in sorted_beams[:self.beam_width]:
                last = seq[-1] if seq else self.blank_idx
                beams[seq] = (score, last)

        # Re-score with lexicon constraint
        best_seq, best_score = self._rescore(beams)
        return self._indices_to_text(best_seq)

    def _rescore(self, beams: dict) -> Tuple:
        """Apply lexicon penalty to final beams."""
        best_seq = ()
        best_score = -float("inf")

        for seq, (score, _) in beams.items():
            text = self._indices_to_text(seq)
            lex_score = self._lexicon_score(text)
            total = score + self.lm_weight * lex_score
            if total > best_score:
                best_score = total
                best_seq = seq

        return best_seq, best_score

    def _lexicon_score(self, text: str) -> float:
        """
        Score based on fraction of words in vocabulary.
        Returns value in [-1, 0]: 0 = all words valid, -1 = none valid.
        """
        words = text.lower().split()
        if not words:
            return 0.0
        in_vocab = sum(1 for w in words if w in self.vocabulary)
        return (in_vocab / len(words)) - 1.0

    def _indices_to_text(self, indices: tuple) -> str:
        chars = []
        for idx in indices:
            ch = self.idx_to_char.get(idx, "")
            chars.append(ch)
        return "".join(chars)

    def decode_batch(self, log_probs: torch.Tensor) -> List[str]:
        """
        Decode a batch.
        log_probs: (T, B, vocab_size)
        """
        T, B, V = log_probs.shape
        results = []
        for b in range(B):
            results.append(self.decode(log_probs[:, b, :]))
        return results


def build_spanish_lexicon(labels_file: str,
                           extra_words: List[str] = None) -> Set[str]:
    """
    Build a Renaissance Spanish lexicon from training transcriptions.
    Optionally augment with a provided word list.
    """
    import re
    vocab = set()

    with open(labels_file, encoding="utf-8") as f:
        for line in f:
            if "	" not in line:
                continue
            text = line.strip().split("	", 1)[1]
            words = re.findall(r"[\wÀ-ɏ]+", text.lower())
            vocab.update(words)

    if extra_words:
        vocab.update(w.lower() for w in extra_words)

    print(f"✓ Spanish lexicon built: {len(vocab)} unique words")
    return vocab


print("✓ beam_decoder.py saved")
