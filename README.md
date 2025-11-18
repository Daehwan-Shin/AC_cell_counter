# IOLMaster Anterior Chamber Cell Quantification (Annulus-like ROI)

This repository contains a Streamlit application for **semi-automatic quantification of inflammatory cells in the anterior chamber** using **IOLMaster B-scan images**.

The app:
- Applies **NL-means denoising** to the raw B-scan.
- Automatically detects the **cornea** and **lens/iris** as two bright blobs.
- Extracts the **posterior corneal surface** and **anterior lens/iris surface**.
- Builds an **annulus-like anterior chamber (AC) ROI polygon** between those arcs.
- Performs **thresholding** within the AC ROI and **detects circular bright spots** as “cells”.
- Overlays the detected cells and AC ROI onto the original / processed images.

> ⚠️ This tool is intended for research/prototyping use only and is **not** a medical device.

---
## Acknowledgment & Inspiration

This project is an homage to the anterior chamber cell quantification workflow pioneered by Francesco Pichi, M.D.**, who introduced a widely adopted **ImageJ macro–based method** for AC inflammation analysis in swept-source optical biometry and OCT systems.

His approach to:
- defining reproducible ROI geometry,
- isolating AC inflammatory cells via thresholding,
- and standardizing quantification across devices

served as the conceptual foundation for this Streamlit-based, fully Python-implemented workflow.

While this project does **not** reuse ImageJ code, the overall analytical logic and ROI philosophy are derived from **Dr. Pichi’s elegant and influential ImageJ macro methodology**, adapted into a modern, interactive, real-time Python pipeline.


## Features

- 📥 Upload IOLMaster B-scan images (`PNG/JPG/TIF`).
- 🧽 **NL-means denoising** with adjustable parameters:
  - `h factor`
  - `patch size`
  - `patch distance`
- 💡 **Central beam removal** (bright central row suppression).
- 🌓 Automatic **cornea / lens blob detection** based on brightness.
- 🎯 Posterior cornea & anterior lens/iris **arc extraction** and smoothing.
- 🟢 **Annulus-like anterior chamber ROI** generation using:
  - cornea arc
  - lens arc
  - top/bottom chords
- ⚖️ **Otsu / manual thresholding** within AC ROI.
- 🔴 **Cell detection** based on:
  - area range
  - circularity
- 👁 Multiple visualization modes:
  - Original
  - NLM denoised
  - Beam removed
  - AC ROI overlay
  - Thresholded AC ROI
  - AC ROI + cells
  - Full B-scan + cells

---

## Requirements

- Python 3.9–3.12
- See `requirements.txt` for Python dependencies.

---

## Installation

1. **Clone this repository:**

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
# AC_cell_counter
