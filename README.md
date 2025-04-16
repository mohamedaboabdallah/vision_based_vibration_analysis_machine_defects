# 🔧 **Vision-Based Vibration Analysis for Machine Defect Detection**

## 🚀 **Project Overview**

This project aims to **automatically detect and classify defects in machines** based on **vibration patterns** observed in videos. The approach uses **computer vision** techniques to analyze vibration data captured via video. The project focuses on analyzing three types of machine conditions:

- **Bearing Fault**
- **Normal State**
- **Unbalanced Weight**

Each condition is represented by two different video recordings:
1. **Front view at 40 cm distance** (Captured at **250 RPM**).
2. **Angled view** (Also captured at **250 RPM**).

The goal is to process these videos and use them to identify the type of defect by analyzing the vibrations visually. The final deliverable will be an automated system that can classify these defects in real-time or in post-processing.

---

## 📹 **Step 1: Video Collection**

### **Data Collection**
- **6 videos** collected for the project (2 videos for each defect type):
  - **Bearing Fault** (2 videos)
  - **Normal State** (2 videos)
  - **Unbalanced Weight** (2 videos)
- **Video Specifications**:
  - **Duration**: Each video is approximately **1 minute** long.
  - **Frame Rate**: **250 RPM** for all videos.
  - **Views**: 
    - One recorded from the **front view** at **40 cm** distance.
    - One recorded from an **angled view** at a small angle.
- **Format**: Each video captures a real-time machine operation, with **vibrations** visible as disturbances or oscillations on the machine's surface.

---

## 🧹 Step 2: Video Preprocessing

This step prepares your raw machine vibration videos for analysis by enhancing stability, isolating relevant areas, and improving visual clarity. Below are the key tasks performed:

### 📥 1. **Video Loading**
- Each video is loaded frame by frame using OpenCV (`cv2.VideoCapture`).
- All frames are stored in a list for further processing.



### 🎯 2. **Video Stabilization**
- Vibration videos may include small camera shakes. Stabilization reduces these to focus only on machine motion.
- **Method used**: Optical Flow (Farneback algorithm).
  - Calculates motion between frames.
  - Applies cumulative transformations to keep the video steady.
- Output: A stabilized list of frames aligned to the first frame.


### 🎬 3. **Motion-based ROI Detection**
- The goal is to **automatically find the region of interest (ROI)** where meaningful machine motion occurs.
- **How it's done:**
  - For each frame pair: calculate absolute difference between consecutive grayscale frames.
  - Threshold the difference to detect motion.
  - Accumulate motion over all frames to build a heatmap of where movement happens.
  - Find contours on this motion heatmap.
  - Compute bounding box around all detected motion areas.
- Output: A bounding box `(x, y, w, h)` defining the active motion zone.


### 🔍 4. **ROI Preview**
- A few sample frames are selected.
- Green rectangle is drawn on each to show the detected ROI.
- Frames are displayed using `matplotlib` to allow manual inspection.
- This ensures the ROI detection step worked correctly.


### 🧮 5. **Histogram Equalization (Optional but Applied)**
- Converts each frame to grayscale.
- Applies `cv2.equalizeHist()` to enhance contrast:
  - This brings out more detail in low-light or low-contrast regions.
- Helps improve feature extraction and analysis later.


### ✂️ 6. **Cropping**
- Using the ROI `(x, y, w, h)`, each equalized grayscale frame is cropped to isolate only the vibrating part of the machine.
- This reduces noise from static parts of the scene.

### 💾 7. **Saving Preprocessed Frames**
- All cropped grayscale frames are saved as `.png` images in the specified `output_dir`.
- File names follow a sequence: `frame_0000.png`, `frame_0001.png`, etc.
- These frames can now be used for:
  - Feature extraction
  - Deep learning models
  - Vibration pattern analysis



### ✅ Summary of Output
| Output Item           | Description                                  |
|-----------------------|----------------------------------------------|
| Stabilized Frames     | Aligned frames reducing camera shake         |
| ROI Bounding Box      | Coordinates of the moving machine region     |
| Equalized Grayscale   | Contrast-enhanced version of the ROI         |
| Saved Cropped Frames  | Final processed frames ready for analysis    |
| ROI Preview           | Visual check of correctness of region        |

---

## Step3: Preprocessing and Video Generation

### Overview
In this step, we process and generate synchronized output videos from multiple pre-recorded video frames. These frames represent different machine states (e.g., Unbalanced Weight, Bearing Fault, Normal State), with each state having two types of views (front and angle). The goal is to ensure that all videos have the same number of frames and resolution before saving them as video files.

### Workflow and Key Tasks

### Step 1: Loading Preprocessed Frames
- **Input**: The frames for each condition (Unbalanced Weight, Bearing Fault, Normal State) are stored in six separate folders: 
  - `Unbalance_weight/front`
  - `Unbalance_weight/angle`
  - `Bearing_fault/front`
  - `Bearing_fault/angle`
  - `Normal_state/front`
  - `Normal_state/angle`
  
- **Action**: Frames are loaded from each folder into a list for further processing.

### Step 2: Interpolation of Frames
- **Problem**: Different videos may have varying lengths, resulting in a different number of frames.
  
- **Action**: To ensure all videos have the same number of frames, the frames are interpolated using the `interpolate_frames()` function. This function blends two consecutive frames to create intermediate frames. The number of frames in each video is adjusted to match the maximum frame count found across all videos.

- **Result**: Each video now contains the same number of frames.

### Step 3: Resizing Frames to a Common Size
- **Problem**: The videos may have different resolutions, causing inconsistency when combining them later.

- **Action**: All frames are resized to a target resolution (based on the first video frame's size) using the `resize_frames_to_target_size()` function. This ensures that all videos have the same resolution.

- **Result**: All frames now have the same resolution, making it possible to combine them into synchronized videos.

### Step 4: Video Writing
- **Action**: Using OpenCV’s `cv2.VideoWriter`, a video file is created for each condition, and the resized frames are written to the corresponding video. This is done for each of the six conditions: 
  - `Unbalance_weight/front`
  - `Unbalance_weight/angle`
  - `Bearing_fault/front`
  - `Bearing_fault/angle`
  - `Normal_state/front`
  - `Normal_state/angle`

- **Result**: Six synchronized videos are created and saved in the `merged_preprocessed_videos` directory.

### Output
- Six output videos are generated and saved in `.avi` or `.mp4` format, depending on the specified codec.
- Each video corresponds to one of the six conditions: `Unbalance_weight/front`, `Unbalance_weight/angle`, `Bearing_fault/front`, `Bearing_fault/angle`, `Normal_state/front`, `Normal_state/angle`.

These output videos have the same resolution and frame count, making them ready for further analysis or processing.

### Detailed Code Walkthrough
- **Loading Frames**: The `load_frames_from_folder()` function is used to load frames from each folder.
- **Removing Duplicate Frames**: The `remove_duplicate_frames()` function ensures no duplicate frames are included by comparing consecutive frames.
- **Interpolation**: The `interpolate_frames()` function handles frame duplication or interpolation to ensure all videos have the same number of frames.
- **Resizing**: The `resize_frames_to_target_size()` function resizes all frames to a common resolution.
- **Video Writing**: The `cv2.VideoWriter` class is used to write frames to output videos for each condition.

This step standardizes the input videos by ensuring they have the same resolution and frame count, enabling further analysis or processing.

---

## Step 4: Eulerian Video Magnification (EVM)

## Overview
Eulerian Video Magnification (EVM) is a technique used to amplify subtle motion or changes in videos that are typically imperceptible to the human eye. In this step, we apply EVM to enhance the vibration patterns from the preprocessed synchronized videos. The process involves several key operations, including Laplacian pyramid construction, temporal bandpass filtering, amplification of the desired frequency range, and reconstruction of the modified video.

## Workflow and Key Tasks

### Step 1: Build Laplacian Pyramid for Each Frame
- **Input**: The frames of the synchronized video (from Step 3).
- **Action**: Each frame is transformed into a Laplacian pyramid, which decomposes the image into several levels, each representing different frequency details at various scales.
- **Function Used**: `build_laplacian_pyramid(frame, levels)`.
- **Output**: A list of pyramids, where each pyramid contains multiple levels for a frame.

### Step 2: Convert Pyramid Level to Array
- **Action**: Extract the middle level of the Laplacian pyramid for each frame to represent the primary frequency content of the image.
- **Output**: An array of frames representing the middle level of the pyramid for each video frame.

### Step 3: Temporal Bandpass Filtering
- **Input**: The pyramid levels extracted from the frames.
- **Action**: A temporal bandpass filter is applied to the pyramid frames. The filter is designed to isolate the specific frequencies related to the vibrations or motion of interest (between `freq_min` and `freq_max`).
- **Function Used**: `temporal_bandpass_filter(frames, freq_min, freq_max, fps)`.
- **Output**: The frames are filtered to retain only the desired frequencies.

### Step 4: Amplification and Reconstruction
- **Action**: The filtered frames are amplified to exaggerate the vibration or motion that corresponds to the target frequencies. The amplified frames are then added back to the Laplacian pyramid.
- **Function Used**: `reconstruct_from_laplacian_pyramid(pyramid)`.
- **Output**: A new set of frames with enhanced motion or vibrations, reconstructed from the modified Laplacian pyramids.

### Step 5: Save the Enhanced Video
- **Action**: The processed frames are converted back to BGR format and written to an output video file.
- **Function Used**: `cv2.VideoWriter()` is used to create a video file.
- **Output**: A video where the amplified vibrations or motion are clearly visible.

## Output
- The final output is a video where subtle vibrations or changes in the original video are amplified, making them more visible. The output video will be saved in the specified `output_path` in `.mp4` format.

## Detailed Code Walkthrough
- **Laplacian Pyramid Construction**: The `build_laplacian_pyramid()` function decomposes each frame into several levels, allowing us to isolate different frequency components.
- **Temporal Bandpass Filtering**: The `temporal_bandpass_filter()` function is used to isolate the desired frequency range in the temporal domain, allowing us to focus on the vibrations that occur at specific frequencies.
- **Amplification**: The amplified frames are created by applying a scaling factor to the filtered frames, making subtle motions more pronounced.
- **Reconstruction**: The `reconstruct_from_laplacian_pyramid()` function reconstructs the modified frames by adding the amplified frequencies back to the Laplacian pyramid and then upscaling the pyramid levels.
- **Video Saving**: Finally, the frames are saved as a video using OpenCV's `cv2.VideoWriter()`.

This step enhances subtle motions, such as vibrations or fluctuations, which are otherwise imperceptible, enabling better analysis and detection of machine defects.

---

## Step 5: Video Segmentation into Time Slices with Overlap

This step involves splitting vibration videos into smaller time slices (segments) to allow detailed analysis of vibration patterns. Segments can either be non-overlapping or have a defined overlap between consecutive segments.

### 1. Process Overview

- **Segment Duration:** Each segment is defined by a fixed duration (e.g., 5, 10, or 15 seconds).
- **Overlap Ratio:** An optional overlap between segments (e.g., 50%) to capture smoother transitions.
- **Input Videos:** Videos from three categories (`Normal_state`, `Bearing_fault`, `Unbalance_weight`) with two views: `front_evm.avi` and `angle_evm.avi`.

### 2. Segmentation Process

1. **Read Video:** The video is loaded, and its total number of frames is determined.
2. **Calculate Segments:** Segments are created based on the segment duration and overlap ratio. The step size between consecutive segments is adjusted for overlap.
3. **Save Segments:** Each segment is saved as an individual video file with a name indicating segment number, duration, and overlap type.
4. **Directory Organization:** Segments are organized by category and overlap type in directories like `segmented_5`, `segmented_10_overlap`, etc.

### 3. Output Structure

Segments are saved in structured directories for each category:

- `segmented_5/`, `segmented_10/`, `segmented_15/` (non-overlapping)
- `segmented_5_overlap/`, `segmented_10_overlap/`, `segmented_15_overlap/` (overlapping)

Example output filenames:
- `front_evm_segment_1_5s.avi`
- `angle_evm_segment_1_10s_50_overlap.avi`

### 4. Benefits of Segmentation

- **Granular Analysis:** Breaks videos into smaller segments for more detailed analysis.
- **Overlap Option:** Ensures smoother transitions between segments, preventing data loss at boundaries.
- **Customizable:** Flexible segment durations and overlap ratios for tailored analysis.

This step prepares the video data for the next stages of processing, ensuring that relevant features can be extracted from distinct time intervals.

---

