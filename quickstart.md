# Quickstart Guide

This guide will walk you through a complete analysis workflow, from loading a video to viewing the results.

## Prerequisites

- You have successfully installed the Behavioral Maze Analyzer by following the instructions in the [README.md](./README.md) file.
- You have one or more maze videos ready for analysis.

---

## Step-by-Step Analysis

### 1. Launch the Application

Open a terminal or command prompt, navigate to the project directory, activate your virtual environment, and run:

```bash
python gui_main.py
```

### 2. Add Videos

- Click the **"Add Videos"** button.
- A file dialog will open. Select one or more video files (.mp4, .avi, etc.).
- The selected videos will appear in the list. The output files will be saved in the same directory as the first video you added.

### 3. Start the Configuration Workflow

Click the **"Configure & Start Analysis"** button.

### 4. Select Maze Type

- A dialog box will appear asking you to select the maze type (e.g., "Y-Maze", "Elevated Plus Maze").
- Choose the correct type for your videos and click **"OK"**.

### 5. Define Regions of Interest (ROIs)

- An ROI selection window will appear, displaying the first frame of your reference video.
- Follow the on-screen instructions at the top of the window.
- **Left-click** to add points to form a polygon for the current zone (e.g., "Arm A").
- **Right-click** to finish the current shape. You can draw multiple shapes for a single zone if needed.
- Click **"Next Zone"** to save the ROIs for the current zone and move to the next.
- After defining all ROIs, you will be asked if you want to draw a reference line for distance calibration. This is highly recommended for accurate distance measurements.

### 6. Tune Detection Parameters

The Interactive Detection Tuner window will appear. This is where you ensure the software can accurately track your animal.

- Use the sliders on the right to adjust parameters:
  - **Threshold**: The most important parameter. Adjust it until the white mask in the "Threshold View" neatly covers the animal without picking up too much background noise.
  - **Weight Omega / Window Size**: These control the spatial weighting. Generally, the defaults work well.
- Use the frame slider or keyboard shortcuts (**N/P** for next/previous frame, **+/-** for larger jumps) to check the detection on different frames.
- Once you are satisfied with the tracking, click the **"Accept Parameters"** button.

### 7. Finalize Analysis Settings

A final settings dialog will appear. Here you can:

- Adjust the **Start Time** and **End Time** for the analysis.
- Enter the **Real-world length** (in cm) corresponding to the reference line you drew in Step 5.
- Check the box to **"Create Timelapse Videos"**.
- Click **"OK"** (or **"Accept"** in batch mode).

### 8. Run the Analysis

- If you are analyzing multiple videos, you will be asked if you want to verify the settings for each video.
  - **Yes**: You will repeat Step 7 for each video, allowing you to set unique start/end times or redraw ROIs if needed.
  - **No**: The settings you just configured will be applied to all videos in the batch.
- A final confirmation will appear. Click **"Yes"** to start the analysis.
- The progress bar and log window will show the analysis progress. You can click **"Cancel Analysis"** to stop the process.

### 9. Review Your Results

- Once the analysis is complete, a confirmation message will appear.
- Navigate to the output directory (the same folder as your first video). You will find all the generated CSV files, plots, and timelapse videos, clearly named after the source video file.

---

**That's it!** You have successfully analyzed your first video.