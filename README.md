# Pothole Detection Software using YOLOv8 and OpenCV

!Pothole Detection

## Overview

This repository contains the implementation of a pothole detection system using YOLOv8 (You Only Look Once version 8), a state-of-the-art object detection algorithm. The software detects potholes in images and videos, making it useful for road maintenance and safety.

## Features

- Real-time pothole detection
- Easy integration with existing projects
- Customizable model training using your own dataset

## Getting Started

1. **Dataset Preparation**:
   - Download and prepare the pothole detection dataset in the required YOLOv8 format. Organize the dataset into training and validation sets.

2. **Model Training**:
   - Download the YOLOv8 pre-trained weights.
   - Modify the configuration file for YOLOv8 to carry out multi-resolution training.
   - Train the model using your prepared dataset.
   - Save the best model weights as `best.pt`.

3. **Inference on Videos**:
   - Use the trained model (`best.pt`) to run inference on real-world pothole detection videos.
   - Analyze the results and visualize the detected potholes.
   - WHILE USING CODE READ README FILE AND DOWNLOAD ZIP FILE FROM THE GOOGLE DRIVE LINK
