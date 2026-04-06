#  Historical OCR Pipeline 

## Overview
This project implements an end-to-end Optical Character Recognition (OCR) system for transcribing 17th-century Spanish printed texts using a CRNN (Convolutional Recurrent Neural Network) architecture with CTC loss.

The system is trained on the **RODRIGO corpus** (~15,000 line images) and is designed to handle historical typography, ligatures, and diacritics found in early modern manuscripts.

---

##  Key Features
- End-to-end OCR pipeline using CNN + BiLSTM + CTC Loss
- No character-level annotations required (sequence learning via CTC)
- Handles historical Spanish text with diacritics and ligatures
- Beam search decoding with optional lexicon constraints
- LLM-based post-processing for error correction (Gemini / GPT-4o-mini)
- Evaluation using CER (Character Error Rate) and WER (Word Error Rate)

---

##  Architecture

### CRNN (CNN + RNN + CTC)
- **CNN (VGG-style):** Extracts spatial features from text-line images
- **BiLSTM:** Models sequential dependencies in both directions
- **CTC Loss:** Enables alignment-free training for variable-length sequences

---

##  Design Choices

### Architecture Choice: CRNN with CTC
A CRNN architecture is used because it enables sequence recognition from images without requiring character-level segmentation:
- CNN extracts column-wise visual features (strokes, shapes, typography)
- BiLSTM captures contextual dependencies and ligatures
- CTC loss aligns predictions with text sequences without explicit alignment

---

### LLM Post-Correction
A lightweight LLM (Gemini / GPT-4o-mini) is used as a **post-processing step**:
- Corrects common OCR errors (e.g., rn → m, 0 → o, I → l)
- Improves linguistic coherence and spelling
- Does not replace the OCR model — only refines outputs

---

##  Evaluation Metrics

- **CER (Character Error Rate):**  
  Measures character-level edit distance  
  → Primary metric for OCR performance  

- **WER (Word Error Rate):**  
  Measures word-level errors  
  → Sensitive to spacing and segmentation  

Evaluation is performed:
- Before LLM correction  
- After LLM correction  

---

##  Text Region Detection
A morphology + contour-based approach is used to:
- Detect main text regions
- Remove marginalia, headers, and noise
- Ensure model processes only relevant content

---

##  Dataset
- **RODRIGO OCR Dataset** (Zenodo)
- ~15,000 line images from 16th–17th century Spanish texts

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
