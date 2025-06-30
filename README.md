# Optic Nerve Mask Segmentation App

This app allows you to upload optic nerve images, run segmentation, select chiasm points, and analyze nerve diameters. This is useful for tumor analysis and numerous other practical applications.

## Features

- Upload multiple optic nerve images for batch processing
- Automatic segmentation using a pretrained model
- Interactive point selection for chiasm analysis
- Visualization and CSV export of nerve diameter measurements

## Requirements

- Python 3.8–3.11 (`inference` does not support versions above 3.11). Please install the appropriate Python version from [python.org](https://www.python.org).
- See [`requirements.txt`](requirements.txt) for dependencies

## Installation

1. Clone this repository:

   ```sh
   git clone <repo-url>
   cd OpticNerve-Analysis
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Running the App

Start the Streamlit app with:

```sh
streamlit run app.py
```

The app will open in your browser. Follow the instructions to upload images and analyze them.

## Notes

- You need an internet connection for model inference.
- Results and CSVs can be downloaded after processing images.
- If you want multi-file upload, simply select numerous files in your desktop and upload them/
- When you are selecting the left and right points for the diameter interval, please sure that both sides of the nerve are fully encompassed by the left bound.
