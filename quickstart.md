# Quickstart Guide

This guide will walk you through a complete analysis workflow, from loading your first video to reviewing the generated heatmaps and data.

## 1. Launch the Application
- **If using the executable (.exe):** Double-click the downloaded file.
- **If using Python:** Open your terminal, activate your virtual environment, and run `python gui_main.py`.

## 2. Load Your Videos
1. Click the **"Add Videos"** or **"Add Folder"** button at the top of the main window.
2. Select the video files (.mp4, .avi, etc.) you want to analyze. 
   *Note: All output files (CSVs, plots, timelapses) will be saved in the same directory as the first video in your list, unless you change it via `Tools -> Set Output Directory`.*

## 3. Start the Configuration Workflow
Click the **"Configure & Start Analysis"** button. 

### Step A: Select Maze Type
A dialog will appear asking you to select the paradigm (e.g., "Y-Maze", "Elevated Plus Maze", "Tail Suspension Test", or "Freestyle / Open Field"). Select the appropriate option.

### Step B: Draw Regions of Interest (ROIs)
1. You will see a frame from the middle of your video.
2. Follow the prompt at the top left to draw the current zone (e.g., "Arm A").
3. **Left-click** to drop points and draw a polygon around the arm.
4. **Right-click** to close and finish the shape.
5. Click **"Next Zone"** and repeat until all zones are drawn.
6. *Calibration:* You will be asked to draw a reference line. Draw a line along an object of known length in the real world (e.g., a 30cm maze arm). This allows the software to output distances in centimeters instead of pixels.

### Step C: Tune Detection Parameters
The Interactive Detection Tuner ensures the software cleanly separates the animal from the background.
1. Use the **Frame Navigation** slider (or the `N` and `P` keys) to look at different parts of the video.
2. Adjust the **Threshold** slider until the white mask in the "Threshold View" neatly covers the animal without picking up background noise (like reflections or shadows).
3. Check the "Detection Quality" panel on the right. If it says "Excellent" or "Fair", you are good to go.
4. *(Optional)* Click **Quick Scan Test** to automatically test the settings across 50 random frames.
5. Click **Accept Parameters**.

### Step D: Finalize Time & Settings
The final review dialog allows you to:
- Adjust the **Start Time** and **End Time** (useful for trimming setup/handling time).
- Enter the real-world length of the calibration line you drew earlier.
- Check the box to generate a **Validation/Timelapse Video** (highly recommended for verifying results).

## 4. Run the Batch Analysis
- If you have multiple videos loaded, the software will ask if you want to **Verify Each** or **Run All Now**.
  - **Verify Each:** Allows you to adjust start/end times or redraw ROIs slightly if the camera moved between recordings.
  - **Run All Now:** Applies the exact same ROIs and settings to every video in the list.
- Click **Yes** to begin. The progress bar will show the estimated time remaining.
- *Note: If you need to stop, click **Cancel Analysis**. The software will save a checkpoint, and you can resume from the exact same frame later!*

## 5. Review Your Results
Once complete, navigate to your output folder. For every video, you will find:
1. `[video_name]_tracking_data.csv`: Frame-by-frame X/Y coordinates and current zone.
2. `[video_name]_analysis_summary.csv`: Total distance, time in zones, alternation scores, etc.
3. `[video_name]_heatmap.png`: A visual density map of where the animal spent the most time.
4. `[video_name]_timelapse.mp4` (if enabled): A sped-up video showing the tracking dot, drawn ROIs, and text indicating the recognized zone.

You will also find a `_BATCH_SUMMARY.csv` containing the final aggregated statistics for all processed videos in one easy-to-copy spreadsheet.