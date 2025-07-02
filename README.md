# Optic Nerve Segmentation App

This app allows you to upload optic nerve images, run segmentation, select chiasm points, and analyze nerve diameters. This is useful for tumor analysis and numerous other practical applications.

## Features

- Upload multiple optic nerve images for batch processing
- Automatic segmentation using a pretrained model
- Interactive point selection for chiasm analysis
- Visualization and CSV export of nerve diameter measurements

## Requirements

- Python 3.8–3.11 (`inference` does not support versions above 3.11). Please install the appropriate Python version from [python.org](https://www.python.org).
- See [`requirements.txt`](requirements.txt) for dependencies
- If you want to maintain support for this app with new updates, please install `git` from [git.scm.com](https://git-scm.com/downloads).

## Installation

1. Clone this repository:
   

   Run the following in your local terminal:
   ```sh
   git clone <repo-url>
   ```
   *Note: If you do not have git installed, simply download this reposity and unzip it.*

   From there, either open this repository in a code editor/IDE like Visual Studio Code and open the terminal, or simply navigate to its location in your local terminal using the following command (assuming your folder is stored inside Desktop):

   ```sh
   cd OpticNerve-Analysis
   ``` 
   **If your file is not stored in your main desktop, please put the actual path to the app's location after** `cd`

2. Install dependencies:

   Run this in the terminal:

   ```sh
   pip install -r requirements.txt
   ```

   Ensure that the dependencies are installing correctly. 


## Running the App

Start the Streamlit app with:

```sh
streamlit run app.py
```

The app will open in your browser. Follow the instructions to upload images and analyze them.

## Platform-Specific Notes

### Mac
- On mac, the local terminal app is called "terminal"
- If you see a security warning about running Python or Streamlit, you may need to allow the app in **System Preferences > Security & Privacy**.

### Windows

- On windows, the local terminal app is called "command prompt"
- You may need to add python certain modules to your environment variables on your device to ensure everything runs properly
- If you get a warning about running Python, you may need to allow it through Windows Defender or your antivirus.

## General Reminders

- You need an internet connection for model inference.
- Results and CSVs can be downloaded after processing images.
- If you want multi-file upload, simply select numerous files in your desktop and upload them/
- When you are selecting the left and right points for the diameter interval, please sure that both sides of the nerve are fully encompassed by the left bound.
