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
2. Adjust the **Threshold**, minimum component area, and morphology controls
   until the green selected component in the "Threshold View" covers the
   animal without selecting reflections or shadows.
3. Check the evidence-based confidence and quality panel. “Review” or “Poor”
   means the validation output must be inspected and the settings may need
   adjustment.
4. *(Optional)* Click **Quick Scan Test** to test deterministic sequential
   windows from the beginning, middle, and end of the video.
5. Click **Accept Parameters**.

For TST, the evidence tuner is different: select each mouse in turn and review
the state overlay across the scan. Purple is coherent passive swing, red is
static immobility, green is active escape, and orange/yellow is uncertain and
excluded from scoring. The tuner reviews up to five minutes to remain
responsive; the actual analysis uses the entire selected interval.

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

When reusing a saved configuration, the target videos must have the same frame
dimensions as the video used to draw the ROIs. The preflight check will stop
the run if the geometry is incompatible.

## 5. Review Your Results
Once complete, open the new `analysis_run_...` folder inside your chosen
output directory. It contains `batch_summary.csv`, a `manifest.json` run index,
and one folder per video. Each video folder separates:

1. `data/`: frame-level data, human-readable summary, complete JSON, and any
   paradigm-specific event tables.
2. `plots/`: heatmaps, trajectories, time series, and paradigm-specific plots.
3. `media/`: optional annotated validation video; this folder is omitted when
   validation media was not requested.

The batch summary always includes the shared duration, total-distance/unit,
excluded-jump, and tracking-quality columns before paradigm-specific metrics.
Unsupported fields, such as centroid distance for TST, are left blank rather
than reported as zero. See `OUTPUTS.md` for the full per-paradigm output
matrix.

In tracking CSVs, `zone=undetected` means no reliable position was available
for that frame. These frames are excluded from zone occupancy, distance,
transition, and immobility calculations. The same file includes `confidence`
and `failure_reason` columns for quality review.

Configurations saved by releases before 3.0 cannot be used for TST. Re-open
the source video, review every mouse with the new temporal evidence tuner, and
save a new configuration.
