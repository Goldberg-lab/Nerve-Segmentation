# Optic Nerve Segmentation & Diameter Analysis

This app allows you to upload optic nerve images, run segmentation, select chiasm points, and analyze nerve diameters. This is useful for numerous practical applications requiring nerve morphology measurements.

A hosted version is available here: **https://optic-nerve-segmentation.streamlit.app/**. No installation required if you just want to try the app.

The instructions below are for running the app locally.

## Features

- Upload multiple optic nerve images for batch processing
- Automatic segmentation using a pretrained Roboflow SAM model
- Interactive point selection for chiasm analysis
- Constant-interval perpendicular diameter sampling along the nerve contour
- Visualization and CSV export of nerve diameter measurements

## Requirements

- **Miniconda** — installs its own isolated Python automatically. You do **not** need Python already installed on your computer.
  Download: https://docs.conda.io/en/latest/miniconda.html
- **Git** (optional) — used to download the code. If you'd rather not install git, you can instead click the green **Code -> Download ZIP** button on the GitHub repo page and unzip it manually.

## Installation

### 1. Install Miniconda

Download the installer for your operating system from the link above and run it like any normal installer (accept the defaults).

### 2. Get the code

Using git:

```bash
git clone https://github.com/Goldberg-lab/Nerve-Segmentation
cd Nerve-Segmentation
```

Or without git: download the ZIP from GitHub, unzip it, and open a terminal in that folder.

### 3. Create the environment

From inside the project folder, run:

```bash
conda env create -f environment.yml
conda activate optic-nerve-app
```

This creates a self-contained environment with the correct Python version and all required packages.

## Running the App

With the environment activated (`conda activate optic-nerve-app`), run:

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

### Updating later

If you pull new code changes and the environment needs updating:

```bash
conda env update -f environment.yml --prune
```

## Using the App

1. **Upload**: Set the image scale (microns per pixel) and measurement interval, then upload one or more optic nerve images.
2. **Segmentation**: The app runs Roboflow inference automatically and displays the segmented nerve mask.
3. **Select Points**: Click three points on the mask. First the rightmost point (just left of the chiasm), then the two leftmost points (one for the top leg, and one for the bottom leg). Make sure points are within the nerve's bounds.
4. **Diameter Analysis**: The app automatically samples diameters at constant intervals along the nerve contour and displays:
   - Annotated nerve image with diameter overlays
   - Diameter vs. position graph
   - Downloadable CSV of all measurements
5. For batch uploads, repeat steps 2 - 4 for each image; a combined ZIP of all CSVs is available after the last image.

## General Reminders

- An internet connection is required for model inference (Roboflow API).
- Results and CSVs are downloadable directly from the app after processing.
- To batch-upload, select multiple image files at once in the upload dialog.
- When selecting the left/right chiasm points, ensure all points are within the nerve's length to prevent measurement errors.
