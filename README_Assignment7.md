# CSc 8830 - Assignment 7 Web Demo

This page implements the two coded parts of Assignment 7:

- Problem 1: calibrated stereo object-size estimation
- Problem 3: room dimension estimation from stereo videos

Problem 2 remains a handwritten derivation and is intentionally not implemented on the webpage.

## Files

- `assignment7.html`
- `assignment7.css`
- `assignment7.js`
- `assets/left_speaker.jpg`
- `assets/right_speaker.jpg`
- `assets/left_circle.jpg`
- `assets/right_circle.jpg`
- `assets/left_polygon.jpg`
- `assets/right_polygon.jpg`
- `assets/room_left.mp4`
- `assets/room_right.mp4`

The uploaded iPhone MOV room videos were converted to browser-friendly H.264 MP4 files for the webpage.

## Problem 1

The user can choose:

1. Preloaded stereo pair
2. Upload a new LEFT and RIGHT image pair
3. Use a live browser camera and capture LEFT and RIGHT views

Supported geometry:

- Rectangle: TL, TR, BR, BL
- Circle: D1, D2 across the diameter
- Polygon: visible vertices P1...Pn in order

The page corrects clicked coordinates using the supplied lens-distortion model, estimates disparity, computes Z first, reconstructs X/Y/Z, then computes the requested physical dimensions.

## Problem 3

The page can use:

1. The included LEFT and RIGHT room videos
2. User-uploaded LEFT and RIGHT videos

The user selects a frame time from each video, captures those frames, chooses Width / Height / Length, and clicks the same two physical endpoints in both frames.

## Calibration

At the top of the page, replace the default values with the latest iPhone Continuity Camera calibration values produced by the Python calibration step. Use the same calibration for both views because the same iPhone camera is used at both positions.

The calibration values can be saved in browser local storage with the `Save Calibration in Browser` button.

## GitHub / Vercel

Copy the entire folder contents into the repository, preserving the `assets/` folder. Add a homepage link to `assignment7.html`.
