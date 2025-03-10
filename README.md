
# Autonomous RC Vehicle with Computer Vision

This project implements a **Python-based computer vision system** designed to enhance a remote-controlled vehicle to an **autonomous navigation system** using **OpenCV**. The system begins by **calibrating the camera** to correct for lens distortion, ensuring accurate image processing. 

The **vehicle detects our target** object (in this case is a hawk photo; logo.png) **via detection algorithms** like ORB, SIFT, and AKAZE. Computing **real-time positional data**, allowing the vehicle to dynamically **adjust steering and speed** based on the target's location.

The communication between the vision system and RC car is handled via Arduino serial communication, enabling a seamless integration between computer vision and motor control. All together the system allows the RC car to **autonomously track and navigate towards its target** within its environment.

## Features

- **Camera Calibration & Undistortion:**
  - Calibrate cameras using chessboard images.
  - Undistort images and live video feeds with the saved calibration data.
- **Feature Detection:**
  - Utilize ORB, SIFT, and AKAZE algorithms to detect a template object (e.g. logo.png).
  - Compute the object's position relative to the video frame (e.g., left/right and distance from the center).
- **RC Vehicle Control:**
  - Communicate with an Arduino via serial to send speed and steering commands.
- **Utility Tools:**
  - Camera capture, testing, and selection utilities.

## Requirements

- Python 3.x
- [OpenCV](https://opencv.org/) (cv2)
- [NumPy](https://numpy.org/)
- [PySerial](https://pythonhosted.org/pyserial/)


## Usage
  - [Camera Calibration](#camera-calibration)
  - [Image Undistortion](#image-undistortion)
  - [Feature Detection](#feature-detection)
  - [RC Vehicle Control](#rc-vehicle-control)

---

#### **Camera Calibration**

- Place chessboard images in the project directory
- Run the calibration script
````
python CameraCal.py
````
- calibration data will be saved in calibration_data.npz
- undistortion.py: transfers the captured image based on calibration
- CamCalErr.py: Compares the two types of pictures
- The distance and positon code is DisPos.py
<img src="https://i.postimg.cc/5tKrJK2R/Screenshot-2025-02-23-165740.png" width="350" alt="Process Flow">

---

#### **Image Undistortion**
- To Undistort images
````
python undistortion.py
````
- Provided opencv_frames inside directory have been altered to protect individual's identity

---
#### **RC Vehicle Control**
To control an RC vehicle based on detected object position:

- Test Serial Communication:
````
python RCDriveTest.py
````

- Run the Full Prototype:
````
python Prototype.py
````

### Feature Detection
Feature detection is used to recognize and track an object (such as a template image) in the video stream. This project supports multiple feature detection algorithms, each with different trade-offs in speed and accuracy. Below are the available methods:


- ORB (Oriented FAST and Rotated BRIEF). ORB is an efficient, fast, and rotation-invariant feature detection algorithm. It is well-suited for real-time applications due to its speed.

- AKAZE (Accelerated-KAZE) is an advanced feature detection method that provides better performance than ORB for detecting objects with varying lighting conditions and texture details.

- SIFT (Scale-Invariant Feature Transform) with Brute-Force Matcher is a more robust algorithm for detecting objects under various transformations (e.g., scale, rotation). The brute-force matcher compares all features exhaustively, making it accurate but computationally expensive.

- SIFT with FLANN (Fast Library for Approximate Nearest Neighbors) Matcher. FLANN is an optimized nearest-neighbor search algorithm that improves the efficiency of feature matching compared to brute-force matching.


| Method              | Speed   | Accuracy     | Best Use Case |
|---------------------|---------|--------------| --------------|
|ORB |Fast |Moderate |Real-time applications with limited computational resources |
|AKAZE  |Medium |High |Detailed objects with variations in texture and lighting |
|SIFT + Brute-Force |Slow |Very High |High-accuracy applications where performance is not a constraint |
|SIFT + FLANN |Medium |High |Real-time applications needing a balance between accuracy and speed|


## Known Issues & Limitations
- Fixed Turn Angle: The prototype uses hard-coded turn angles.
- Brute-Force Matching: Some scripts use brute-force matching, which may be suboptimal for real-time performance.
- Detection Handling: The system may not gracefully handle situations where the target object is not detected.

## Pictures + Video testing out the RC Car 
<img src="https://i.postimg.cc/BZTNJWWj/image.jpg" width="300" alt="Process Flow">
<img src="https://i.postimg.cc/mgDS3tP9/IMG-6314.jpg" width="300" alt="Process Flow">
<img src="https://i.postimg.cc/RVGRncZ9/IMG-6315.jpg" width="300" alt="Process Flow">

https://github.com/user-attachments/assets/3f39d3c9-b5c5-4461-b503-fb1aa25277b1

