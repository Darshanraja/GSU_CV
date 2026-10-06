# Module 5 - Optical Flow and Planar Multi-View Reconstruction

## Objective

Module 5 contains two experiments. Problem 1 computes and visualizes optical flow in videos and validates tracked feature displacement using consecutive frames. Problem 2 uses four viewpoints of a planar book cover to rectify the views, reconstruct a common planar appearance, and estimate calibrated camera poses.

## Problem 1 - Optical Flow

The browser implementation uses a motion mask to concentrate feature detection on moving regions, Shi-Tomasi feature points, and pyramidal Lucas-Kanade optical flow. For each tracked point, the implementation reports x1, y1, x2, y2, dx, dy, and displacement magnitude. The best valid validation pair is retained for each video during the browser session.

## Problem 2 - Planar Multi-View Reconstruction

The same four book-cover corners are selected in each of four viewpoints. Homographies rectify the views to a common planar reference, and median fusion combines the rectified images. The calibrated camera matrix is also used with planar pose estimation to obtain rotation, translation, and relative camera centers.

The base OpenCV calibration used in the experiment is:

- fx = 2301.4244
- fy = 2304.4655
- cx = 973.7959
- cy = 494.0908
- Calibration resolution = 1920 x 1080

The saved camera-center results shown in the experiment interface are approximately:

- View 1: C = (1.9987, 2.6245, -2.0226)
- View 2: C = (0.0583, 2.5356, -2.3937)
- View 3: C = (-1.1584, 1.0725, -2.7148)
- View 4: C = (1.9379, 0.6807, -2.8876)

The camera positions are reported in normalized cover-width units because physical book dimensions were not supplied.

## Website Demo

Open `module5.html`. Problem 1 contains the optical-flow video demo and tracking validation. Problem 2 contains the four-view planar reconstruction, selected point coordinates, homographies, camera calibration information, and estimated camera poses.

## Saved Output Images

The permanent website result gallery is stored in `module5_results/`. It contains the optical-flow validation result, all four book viewpoints with selected corners, rectified views, reconstructed object, and camera-parameter/pose output.

## Files

- `module5.html` - web interface
- `module5.js` - optical flow, planar reconstruction, and pose estimation
- `module5.css` - Module 5 styles and saved-result gallery styles
- `module5_results/` - saved experiment outputs

## Deployment Note

Keep the existing `module5_assets/` folder in the repository because the interactive Module 5 demo uses the preloaded videos and book images from that folder.
