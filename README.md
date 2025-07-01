# RawImageBatchLab

A minimal GUI application for applying common RAW photo adjustments and batch processing files.

This project is currently a work in progress. It features a Tkinter-based interface for tweaking exposure, contrast and other controls with a live preview, and a small processing pipeline built on `rawpy` and `OpenCV`.

RawImageBatchLab is open source software released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Usage

1. Install the dependencies:

   ```bash
   pip install rawpy opencv-python Pillow
   ```

   (Tkinter is included with most Python installations.)

2. Launch the application:

   ```bash
   python ui.py
   ```

3. Click **Add Images** to select RAW files, adjust the sliders while viewing the live preview, and use **Save Options** to set the output destination, file naming and other preferences. When satisfied, choose **Process Images** to save the results.

