"""
================================================================================
CSc 8830 - Computer Vision | Assignment 6 | Problem 1
Sparse Optical Flow (Lucas-Kanade) with AUTOMATIC motion-region tracking
================================================================================

WHAT THIS SCRIPT DOES
----------------------
1. Loads a video (file dialog) and processes the first PROCESS_SECONDS seconds.
2. Automatically finds the MOVING region(s) of the scene using background
   subtraction (MOG2) -- no manual box-drawing. On a mostly static scene
   (parked-camera footage of a house, a street, etc.), the strongest corners
   by contrast/gradient alone are almost always static architecture (window
   and door frames, brick lines), so plain Shi-Tomasi run on the whole frame
   tends to lock onto the background and report ~0 pixel displacement every
   frame. Restricting feature detection to the foreground motion mask fixes
   this without any user interaction.
3. Detects Shi-Tomasi corners inside that automatically-computed motion mask.
4. Tracks them frame-to-frame with pyramidal Lucas-Kanade optical flow.
5. Draws per-point motion trails + the instantaneous flow vectors.
6. Writes an annotated output video.
7. Saves ONE validation pair of consecutive frames (frame N, frame N+1),
   BOTH as annotated figures (numbered points, for the report) and as CLEAN
   unannotated copies (for the bilinear-interpolation / brightness-constancy
   check in bilinear_validation.py), plus a CSV of tracked coordinates.

HOW THE AUTOMATIC MOTION MASK WORKS
-------------------------------------
`cv2.createBackgroundSubtractorMOG2` models each pixel's intensity history
as a mixture of Gaussians and flags pixels that deviate from that learned
background as foreground (i.e., moving). Every frame:
    fg_mask = background_subtractor.apply(frame_gray)
This mask is thresholded and cleaned with morphological opening/closing to
remove sensor noise and small false positives, then dilated slightly so
Shi-Tomasi has a few pixels of margin around each moving object's silhouette.
`cv2.goodFeaturesToTrack(..., mask=fg_mask)` then only returns corners that
fall inside that foreground region, so the tracker locks onto the moving
subject(s) automatically, on any input video, with no manual step.

The background subtractor needs a short learning period (a couple of
seconds) before its foreground mask is reliable, so feature (re)detection
does not start until BACKGROUND_LEARNING_FRAMES has elapsed.

HOW TO RUN
----------
    pip install opencv-python numpy
    python optical_flow_tracking.py

CONTROLS
--------
- Fully automatic -- no windows to interact with beyond the live preview.
- During processing: ESC stops early.

OUTPUTS (written to the working directory)
-------------------------------------------
- optical_flow_output.mp4      : annotated video with trails + vectors
- validation_frame_1.png       : annotated earlier frame (numbered points)
- validation_frame_2.png       : annotated later frame (numbered points)
- raw_frame_1.png               : CLEAN earlier frame (no drawing)
- raw_frame_2.png               : CLEAN later frame (no drawing)
- tracking_validation.csv      : point id, frame ids, x1,y1,x2,y2,dx,dy

Author: [YOUR NAME] -- CSc 8830, Assignment 6
================================================================================
"""

import cv2
import numpy as np
import csv
from tkinter import Tk, filedialog


# ============================================================
# SETTINGS
# ============================================================

PROCESS_SECONDS = 30

OUTPUT_VIDEO = "optical_flow_output.mp4"
VALIDATION_FRAME_1 = "validation_frame_1.png"          # annotated, for the report figure
VALIDATION_FRAME_2 = "validation_frame_2.png"          # annotated, for the report figure
RAW_FRAME_1 = "raw_frame_1.png"                        # CLEAN copy, no drawing -- use this
RAW_FRAME_2 = "raw_frame_2.png"                        # for the bilinear-interpolation check
VALIDATION_CSV = "tracking_validation.csv"

# Which frame pair (relative to start) to freeze for the hand-validation figure.
VALIDATION_SECONDS_IN = 5

# Background subtractor needs this many frames to build a stable background
# model before its foreground mask is trustworthy.
BACKGROUND_LEARNING_FRAMES = 20

# Minimum foreground blob area (pixels) to be considered real motion rather
# than noise -- filters speckle out of the MOG2 mask.
MIN_MOTION_AREA = 400


# ============================================================
# SELECT VIDEO
# ============================================================

def select_video():
    root = Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select Video",
        filetypes=[("Video Files", "*.mp4 *.mov *.avi *.mkv"), ("All Files", "*.*")]
    )
    root.destroy()
    return path


# ============================================================
# AUTOMATIC MOTION MASK
# ============================================================

def compute_motion_mask(back_sub, frame):
    """Foreground (moving-pixel) mask via MOG2 background subtraction,
    cleaned with morphology and filtered by minimum blob area."""
    fg = back_sub.apply(frame)

    # Drop shadow label (MOG2 marks shadows as 127); keep only strong foreground.
    _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
    fg = cv2.dilate(fg, kernel, iterations=2)

    # Remove small blobs (noise) -- keep only sufficiently large moving regions.
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(fg)
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area >= MIN_MOTION_AREA:
            cv2.drawContours(clean_mask, [c], -1, 255, thickness=cv2.FILLED)
            boxes.append(cv2.boundingRect(c))

    return clean_mask, boxes


# ============================================================
# MAIN
# ============================================================

def main():
    video_path = select_video()
    if not video_path:
        print("No video selected.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print("\n=======================================")
    print("VIDEO INFORMATION")
    print("=======================================")
    print("FPS:", fps)
    print("Resolution:", width, "x", height)
    print("Duration:", duration, "seconds")
    if duration < 30:
        print("\nWARNING: Assignment asks for at least 30 seconds.")

    max_frames = int(fps * PROCESS_SECONDS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    back_sub = cv2.createBackgroundSubtractorMOG2(
        history=200, varThreshold=32, detectShadows=True
    )

    ret, old_frame = cap.read()
    if not ret:
        print("Could not read first frame.")
        return
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # --------------------------------------------------------
    # LEARN THE BACKGROUND MODEL (no feature detection yet)
    # --------------------------------------------------------
    print(f"\nLearning background model over {BACKGROUND_LEARNING_FRAMES} frames...")
    back_sub.apply(old_frame)
    for _ in range(BACKGROUND_LEARNING_FRAMES - 1):
        ret, f = cap.read()
        if not ret:
            break
        back_sub.apply(f)
        old_frame = f
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # --------------------------------------------------------
    # SHI-TOMASI CORNERS -- restricted to the automatic motion mask
    # --------------------------------------------------------
    feature_params = dict(maxCorners=100, qualityLevel=0.1, minDistance=5, blockSize=7)

    motion_mask, motion_boxes = compute_motion_mask(back_sub, old_frame)
    old_points = cv2.goodFeaturesToTrack(old_gray, mask=motion_mask, **feature_params)

    if old_points is None or len(old_points) == 0:
        print("No motion detected yet at start -- will keep looking as frames arrive.")
        old_points = np.empty((0, 1, 2), dtype=np.float32)

    # --------------------------------------------------------
    # LUCAS-KANADE PARAMETERS
    # --------------------------------------------------------
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    trail_mask = np.zeros_like(old_frame)
    frame_number = 1
    validation_done = False
    validation_target = int(fps * VALIDATION_SECONDS_IN)
    validation_rows = []

    while True:
        ret, frame = cap.read()
        if not ret or frame_number >= max_frames:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Recompute the motion mask every frame (background/foreground
        # separation is inherently a per-frame operation).
        motion_mask, motion_boxes = compute_motion_mask(back_sub, frame)

        # Re-seed features inside the current motion mask if too few remain
        # (points get lost as the subject moves, or the mask shifts).
        if old_points is None or len(old_points) < 6:
            fresh = cv2.goodFeaturesToTrack(old_gray, mask=motion_mask, **feature_params)
            if fresh is not None:
                old_points = fresh
                trail_mask = np.zeros_like(frame)
            else:
                old_gray = frame_gray.copy()
                old_frame = frame.copy()
                frame_number += 1
                continue

        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, old_points, None, **lk_params
        )

        if new_points is None:
            old_gray = frame_gray.copy()
            old_points = cv2.goodFeaturesToTrack(old_gray, mask=motion_mask, **feature_params)
            continue

        good_new = new_points[status.flatten() == 1]
        good_old = old_points[status.flatten() == 1]

        # ----------------------------------------------------
        # DRAW MOTION (trails + instantaneous vectors)
        # ----------------------------------------------------
        for new, old in zip(good_new, good_old):
            x_new, y_new = new.ravel()
            x_old, y_old = old.ravel()
            cv2.line(trail_mask, (int(x_new), int(y_new)), (int(x_old), int(y_old)), (0, 255, 0), 2)
            cv2.arrowedLine(frame, (int(x_old), int(y_old)), (int(x_new), int(y_new)), (0, 200, 255), 2, tipLength=0.4)
            cv2.circle(frame, (int(x_new), int(y_new)), 4, (0, 0, 255), -1)

        # Draw the automatically detected motion bounding box(es) for reference.
        for (bx, by, bw, bh) in motion_boxes:
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)

        output = cv2.add(frame, trail_mask)

        # ----------------------------------------------------
        # SAVE VALIDATION FRAME PAIR
        # ----------------------------------------------------
        if not validation_done and frame_number >= validation_target and len(good_new) > 0:
            # Save CLEAN (unannotated) copies first -- these are what the
            # bilinear-interpolation brightness-constancy check must sample.
            # Sampling the annotated figures instead just reads the marker
            # circle's own color, not the underlying scene.
            cv2.imwrite(RAW_FRAME_1, old_frame)
            cv2.imwrite(RAW_FRAME_2, frame)

            frame1 = old_frame.copy()
            frame2 = frame.copy()
            number_points = min(10, len(good_new))

            for i in range(number_points):
                old_x, old_y = good_old[i].ravel()
                new_x, new_y = good_new[i].ravel()
                dx = new_x - old_x
                dy = new_y - old_y

                cv2.circle(frame1, (int(old_x), int(old_y)), 6, (0, 255, 0), -1)
                cv2.putText(frame1, str(i + 1), (int(old_x) + 8, int(old_y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.circle(frame2, (int(new_x), int(new_y)), 6, (0, 0, 255), -1)
                cv2.putText(frame2, str(i + 1), (int(new_x) + 8, int(new_y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                validation_rows.append([
                    i + 1, frame_number - 1, frame_number,
                    float(old_x), float(old_y), float(new_x), float(new_y),
                    float(dx), float(dy)
                ])

            cv2.imwrite(VALIDATION_FRAME_1, frame1)
            cv2.imwrite(VALIDATION_FRAME_2, frame2)
            validation_done = True

        cv2.putText(output, f"Frame: {frame_number}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(output, f"Tracked Points: {len(good_new)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        writer.write(output)

        old_gray = frame_gray.copy()
        old_frame = frame.copy()
        old_points = good_new.reshape(-1, 1, 2)
        frame_number += 1

    with open(VALIDATION_CSV, "w", newline="") as file:
        writer_csv = csv.writer(file)
        writer_csv.writerow(["Point", "Frame 1", "Frame 2", "x1", "y1", "x2", "y2", "dx", "dy"])
        writer_csv.writerows(validation_rows)

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print("\n=======================================")
    print("OPTICAL FLOW COMPLETE")
    print("=======================================")
    print("Saved:", OUTPUT_VIDEO)
    print("Saved:", VALIDATION_FRAME_1, "(annotated, for figures)")
    print("Saved:", VALIDATION_FRAME_2, "(annotated, for figures)")
    print("Saved:", RAW_FRAME_1, "(clean, for bilinear_validation.py)")
    print("Saved:", RAW_FRAME_2, "(clean, for bilinear_validation.py)")
    print("Saved:", VALIDATION_CSV)

    print("\nTRACKING VALIDATION")
    for row in validation_rows:
        print(f"Point {row[0]}: ({row[3]:.2f}, {row[4]:.2f}) -> ({row[5]:.2f}, {row[6]:.2f})  "
              f"dx = {row[7]:.2f}, dy = {row[8]:.2f}")


if __name__ == "__main__":
    main()