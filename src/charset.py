"""
src/charset.py
Character set for 17th century Spanish printed sources.
Includes standard Latin chars + Spanish diacritics + historical symbols.
"""

# Spanish Renaissance character set
CHARS = (
    ' !\"#&\'()*+,-./:;?'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    'abcdefghijklmnopqrstuvwxyz'
    '0123456789'
    'ÁÉÍÓÚÜÑáéíóúüñ'
    'ÀÈÌÒÙàèìòù'
    'ÂÊÎÔÛâêîôû'
    'çÇ'
    # Common in early modern Spanish print
    'ÿœæ£§†‡'
)

# CTC blank token is index 0
BLANK_IDX = 0

# Build char -> index and index -> char mappings
# Index 0 reserved for CTC blank
char_to_idx = {c: i + 1 for i, c in enumerate(CHARS)}
idx_to_char = {i + 1: c for i, c in enumerate(CHARS)}
idx_to_char[BLANK_IDX] = ''   # blank decodes to empty

VOCAB_SIZE = len(CHARS) + 1   # +1 for blank


def encode(text: str):
    """Encode text string to list of indices. Unknown chars skipped."""
    return [char_to_idx[c] for c in text if c in char_to_idx]


def decode(indices):
    """Decode list of indices to string, collapsing CTC repeats."""
    result = []
    prev = None
    for idx in indices:
        if idx != BLANK_IDX and idx != prev:
            result.append(idx_to_char.get(idx, ''))
        prev = idx
    return ''.join(result)


print(f'✓ charset.py saved — vocab size: {VOCAB_SIZE}')
