# GSU_CV
Computer Vision CSC 8830

# CSc 8830 - Computer Vision
## Module 2 Assignment

This project performs camera calibration, object width and height measurement, validation using 20 objects, and statistical error calculation.

## How to Run

Create a virtual environment:

python3 -m venv .venv

Activate the virtual environment:

source .venv/bin/activate

Install the required packages:

pip install opencv-python==4.10.0.84 numpy==1.26.4

Run the program:

cd module2

python finalweek1.py

## Step 1 - Camera Calibration

Show the checkerboard to the camera.

Controls:

SPACE = Capture checkerboard  
C = Calibrate  
ESC = Exit

Move and tilt the checkerboard between captures.

## Step 2 - Object Measurement

Enter the distance between the camera and the object in meters.

Example:

1.5

Press SPACE to capture the object.

Then click the four corners in this order:

1. Top-left  
2. Top-right  
3. Bottom-right  
4. Bottom-left  

The program will display the estimated width and height of the object in centimeters.

## Step 3 - Validation

The validation distance used is 2.10 meters.

The program will ask:

1. Predefined objects  
2. Manual entry  

Predefined mode uses the 20 objects already stored in the program.

Manual mode asks for:

Object name  
Actual width  
Actual height  

After completing the measurements, the program calculates the error statistics and saves the results in:

step3_validation_results.csv

## Project Files

finalweek1.py  
camera_calibration.npz  
step3_validation_results.csv  

Georgia State University  
CSc 8830 - Computer Vision  
Module 2 Assignment
