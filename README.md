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

## 🧹 Step 2: Video Preprocessing

This step prepares your raw machine vibration videos for analysis by enhancing stability, isolating relevant areas, and improving visual clarity. Below are the key tasks performed:

---

### 📥 1. **Video Loading**
- Each video is loaded frame by frame using OpenCV (`cv2.VideoCapture`).
- All frames are stored in a list for further processing.

---

### 🎯 2. **Video Stabilization**
- Vibration videos may include small camera shakes. Stabilization reduces these to focus only on machine motion.
- **Method used**: Optical Flow (Farneback algorithm).
  - Calculates motion between frames.
  - Applies cumulative transformations to keep the video steady.
- Output: A stabilized list of frames aligned to the first frame.

---

### 🎬 3. **Motion-based ROI Detection**
- The goal is to **automatically find the region of interest (ROI)** where meaningful machine motion occurs.
- **How it's done:**
  - For each frame pair: calculate absolute difference between consecutive grayscale frames.
  - Threshold the difference to detect motion.
  - Accumulate motion over all frames to build a heatmap of where movement happens.
  - Find contours on this motion heatmap.
  - Compute bounding box around all detected motion areas.
- Output: A bounding box `(x, y, w, h)` defining the active motion zone.

---

### 🔍 4. **ROI Preview**
- A few sample frames are selected.
- Green rectangle is drawn on each to show the detected ROI.
- Frames are displayed using `matplotlib` to allow manual inspection.
- This ensures the ROI detection step worked correctly.

---

### 🧮 5. **Histogram Equalization (Optional but Applied)**
- Converts each frame to grayscale.
- Applies `cv2.equalizeHist()` to enhance contrast:
  - This brings out more detail in low-light or low-contrast regions.
- Helps improve feature extraction and analysis later.

---

### ✂️ 6. **Cropping**
- Using the ROI `(x, y, w, h)`, each equalized grayscale frame is cropped to isolate only the vibrating part of the machine.
- This reduces noise from static parts of the scene.

---

### 💾 7. **Saving Preprocessed Frames**
- All cropped grayscale frames are saved as `.png` images in the specified `output_dir`.
- File names follow a sequence: `frame_0000.png`, `frame_0001.png`, etc.
- These frames can now be used for:
  - Feature extraction
  - Deep learning models
  - Vibration pattern analysis

---

### ✅ Summary of Output
| Output Item           | Description                                  |
|-----------------------|----------------------------------------------|
| Stabilized Frames     | Aligned frames reducing camera shake         |
| ROI Bounding Box      | Coordinates of the moving machine region     |
| Equalized Grayscale   | Contrast-enhanced version of the ROI         |
| Saved Cropped Frames  | Final processed frames ready for analysis    |
| ROI Preview           | Visual check of correctness of region        |

## 🧩 Step 3: Frame Synchronization and Video Reconstruction

After preprocessing the six input videos and extracting their frames, the next step is to **synchronize the frame lengths** and **reconstruct each video** from its processed frames.

### 🎯 Objective

Ensure all videos have the same number of frames by trimming longer sequences, then convert these synchronized frame sequences back into video files for consistent analysis and further processing.

### 🗂️ Input

- 6 folders of preprocessed frames:
  - `Bearing_fault/front`
  - `Normal_state/front`
  - `Unbalance_weight/front`
  - `Bearing_fault/angle`
  - `Normal_state/angle`
  - `Unbalance_weight/angle`

Each folder contains a sequence of preprocessed frames (images).

### ⚙️ Process

1. **Load frames** from each of the six folders.
2. **Determine the minimum number of frames** across all sequences.
3. **Trim all frame sequences** to this minimum length to ensure uniformity.
4. **Reconstruct videos** from the trimmed frames using a fixed frame rate (e.g., 30 FPS).
5. **Save output videos** with names that reflect their original folders.

### 🧪 Why This Step is Important

- Ensures that all videos are temporally aligned for consistent comparison.
- Necessary before applying algorithms like **Eulerian Video Magnification (EVM)** which require consistent frame rates and counts.
- Prepares clean and synchronized video input for defect classification or visualization.

### 💾 Output

- 6 synchronized video files:
  - `Bearing_fault/front.mp4`
  - `Normal_state/front.mp4`
  - `Unbalance_weight/front.mp4`
  - `Bearing_fault/angle..mp4`
  - `Normal_state/angle.mp4`
  - `Unbalance_weight/angle.mp4`
