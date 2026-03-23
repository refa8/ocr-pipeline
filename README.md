# OCR Pipeline for Historical Documents

An end-to-end Optical Character Recognition (OCR) pipeline built in Python, designed to extract and correct text from degraded historical documents. Developed and run on Google Colab.

## Overview

This project implements a complete OCR pipeline that handles synthetic document generation, multi-stage image preprocessing, text recognition using multiple OCR engines, and post-processing correction — including LLM-based correction using GPT-4o-mini.

## Features

- **Synthetic Document Generation** — Creates realistic degraded historical documents with noise, stains, aging effects, perspective distortion, and blur for training/testing
- **PDF Support** — Extracts pages from uploaded PDF documents at 300 DPI
- **Multi-Stage Preprocessing** — Grayscale conversion → Denoising (Non-Local Means) → Deskewing → CLAHE contrast enhancement → Aspect-ratio-preserving resize
- **Multiple OCR Engines** — Benchmarks three engines side by side:
  - **TrOCR** (Microsoft) — Transformer-based, state-of-the-art
  - **Tesseract** — Traditional rule-based baseline
  - **EasyOCR** — Deep learning based, multilingual
- **Post-processing Correction**
  - Lexicon-based correction using edit distance (for Spanish text)
  - LLM-based correction using GPT-4o-mini
- **Evaluation Metrics** — Character Error Rate (CER) and Word Error Rate (WER) computed across all engines

## Project Structure
```
OCR_Pipeline_Research/
├── data/
│   ├── images/          # Synthetic, real, and PDF-derived document images
│   └── annotations/     # Ground truth text labels
├── src/
│   ├── preprocess.py    # ImagePreprocessor class (5-stage pipeline)
│   ├── ocr_engines.py   # TrOCR, Tesseract, EasyOCR wrappers
│   ├── metrics.py       # CER and WER computation
│   ├── corrector.py     # LexiconCorrector (edit distance)
│   └── vocab_builder.py # Vocabulary builder from dataset texts
├── results/             # Output visualizations and evaluation reports
├── models/              # Saved model weights
└── OCR_Pipeline_Complete.ipynb  # Main notebook
```

## Tech Stack

- **Python** (Google Colab)
- **PyTorch** + **HuggingFace Transformers** (TrOCR)
- **OpenCV** (image preprocessing)
- **Pillow** (image generation)
- **Tesseract OCR** + **EasyOCR**
- **OpenAI GPT-4o-mini** (LLM-based correction)
- **editdistance** (lexicon correction and metrics)
- **pdf2image** (PDF extraction)

## How to Run

1. Open `OCR_Pipeline_Complete.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Mount your Google Drive when prompted
3. Add your OpenAI API key to Colab Secrets (🔑 icon in the left sidebar) with the name `OPENAI_API_KEY`
4. Run all cells in order

## Notes

- GPU runtime recommended (Runtime → Change runtime type → T4 GPU)
- First-time setup installs dependencies and downloads TrOCR model (~3–5 minutes)
- Pipeline is configured for Spanish historical documents but can be adapted for other languages
