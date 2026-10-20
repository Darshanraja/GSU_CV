"""
CSc 8830 - Computer Vision
Assignment 7

ONE PROGRAM FOR:

1. iPhone Camera Calibration
2. Problem 1 - Calibrated Stereo Object Measurement
3. Problem 3 - Room Measurement Using Stereo Videos

Install:

    pip install opencv-python numpy

Run:

    python assignment7_all.py


============================================================
GENERAL IDEA
============================================================

The same iPhone is used at two positions:

LEFT POSITION

        camera

          |
          |---------- measured baseline ---------->
                                             camera

                                         RIGHT POSITION


The scene must remain stationary.

Try to keep:

- same phone orientation
- same phone height
- same zoom
- same camera mode
- same resolution


============================================================
PROBLEM 1
============================================================

LEFT photo
RIGHT photo

d = abs(x_left - x_right)

Z = fx * B / d

Then reconstruct X, Y, Z and calculate object dimensions.


============================================================
PROBLEM 3
============================================================

Record:

LEFT room video

then move the same iPhone by the known baseline

RIGHT room video

A frame is extracted from each video.

Corresponding room points are clicked and reconstructed in 3D.
"""

import cv2
import numpy as np
import math
from pathlib import Path


# ============================================================
# GLOBAL SETTINGS
# ============================================================

IPHONE_CAMERA_INDEX = 1

DEFAULT_BASELINE_CM = 15.0

CALIBRATION_FILE = "iphone_calibration.npz"

OUTPUT_DIR = Path("assignment7_outputs")

OUTPUT_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# OPEN IPHONE
# ============================================================

def open_iphone():

    cap = cv2.VideoCapture(
        IPHONE_CAMERA_INDEX,
        cv2.CAP_AVFOUNDATION
    )

    if not cap.isOpened():

        raise RuntimeError(
            "\nCould not open iPhone camera.\n"
            "Try changing IPHONE_CAMERA_INDEX at the top "
            "of the program to 0, 2, or 3."
        )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1920
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        1080
    )

    return cap


# ============================================================
# LOAD CALIBRATION
# ============================================================

def load_calibration():

    path = Path(
        CALIBRATION_FILE
    )

    if not path.exists():

        raise RuntimeError(
            "\nCamera calibration not found.\n\n"
            "Run option 1 first."
        )

    data = np.load(
        CALIBRATION_FILE
    )

    K = data[
        "camera_matrix"
    ].astype(
        np.float64
    )

    distortion = data[
        "distortion_coefficients"
    ].astype(
        np.float64
    )

    calibration_width = int(
        data["image_width"]
    )

    calibration_height = int(
        data["image_height"]
    )

    return (
        K,
        distortion,
        calibration_width,
        calibration_height
    )


# ============================================================
# SCALE CALIBRATION IF RESOLUTION CHANGES
# ============================================================

def scale_camera_matrix(
    K,
    calibration_width,
    calibration_height,
    width,
    height
):

    if (
        width == calibration_width
        and
        height == calibration_height
    ):

        return K.copy()

    sx = (
        width
        /
        calibration_width
    )

    sy = (
        height
        /
        calibration_height
    )

    scaled = K.copy()

    scaled[0, 0] *= sx
    scaled[0, 2] *= sx

    scaled[1, 1] *= sy
    scaled[1, 2] *= sy

    return scaled


# ============================================================
# CAMERA CALIBRATION
# ============================================================

def calibrate_iphone():

    print()
    print("=" * 70)
    print("IPHONE CAMERA CALIBRATION")
    print("=" * 70)

    print()
    print(
        "Enter the number of INNER checkerboard corners."
    )

    print(
        "Example: if OpenCV sees 9 x 6 corners, "
        "enter 9 and 6."
    )

    cols = int(
        input(
            "\nCheckerboard inner columns: "
        )
    )

    rows = int(
        input(
            "Checkerboard inner rows: "
        )
    )

    square_size_cm = float(
        input(
            "Size of one square in cm: "
        )
    )

    minimum_samples = 12


    # --------------------------------------------------------
    # 3D CHECKERBOARD POINTS
    # --------------------------------------------------------

    template = np.zeros(
        (
            rows * cols,
            3
        ),
        np.float32
    )

    template[:, :2] = (
        np.mgrid[
            0:cols,
            0:rows
        ]
        .T
        .reshape(
            -1,
            2
        )
    )

    template *= (
        square_size_cm
    )


    object_points = []
    image_points = []


    cap = open_iphone()


    window = (
        "iPhone Calibration"
    )

    cv2.namedWindow(
        window,
        cv2.WINDOW_NORMAL
    )


    print()
    print(
        "C = capture checkerboard"
    )

    print(
        "R = reset samples"
    )

    print(
        "ENTER = calibrate"
    )

    print(
        "ESC = quit"
    )

    print()
    print(
        "Capture about 15-20 different views."
    )

    print(
        "Move the board around the image and "
        "change distance and angle."
    )


    frame_size = None
    current_corners = None


    while True:

        success, frame = (
            cap.read()
        )

        if not success:
            continue


        height, width = (
            frame.shape[:2]
        )

        frame_size = (
            width,
            height
        )


        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        found, corners = (
            cv2.findChessboardCorners(

                gray,

                (
                    cols,
                    rows
                ),

                flags=(
                    cv2.CALIB_CB_ADAPTIVE_THRESH
                    +
                    cv2.CALIB_CB_NORMALIZE_IMAGE
                )
            )
        )


        display = (
            frame.copy()
        )


        if found:

            corners = (
                cv2.cornerSubPix(

                    gray,

                    corners,

                    (
                        11,
                        11
                    ),

                    (
                        -1,
                        -1
                    ),

                    (
                        cv2.TERM_CRITERIA_EPS
                        +
                        cv2.TERM_CRITERIA_MAX_ITER,

                        30,

                        0.001
                    )
                )
            )


            current_corners = (
                corners.copy()
            )


            cv2.drawChessboardCorners(

                display,

                (
                    cols,
                    rows
                ),

                corners,

                True
            )


            status = (
                "CHECKERBOARD FOUND - PRESS C"
            )

            status_color = (
                0,
                255,
                0
            )


        else:

            current_corners = None

            status = (
                "Checkerboard not detected"
            )

            status_color = (
                0,
                0,
                255
            )


        cv2.rectangle(

            display,

            (
                0,
                0
            ),

            (
                width,
                120
            ),

            (
                0,
                0,
                0
            ),

            -1
        )


        cv2.putText(

            display,

            status,

            (
                20,
                35
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            status_color,

            2
        )


        cv2.putText(

            display,

            f"Samples: {len(object_points)}",

            (
                20,
                70
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (
                0,
                255,
                255
            ),

            2
        )


        cv2.putText(

            display,

            "C=capture R=reset ENTER=calibrate",

            (
                20,
                105
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (
                255,
                255,
                255
            ),

            1
        )


        cv2.imshow(
            window,
            display
        )


        key = (
            cv2.waitKey(1)
            &
            0xFF
        )


        # ----------------------------------------------------
        # CAPTURE
        # ----------------------------------------------------

        if key in [
            ord("c"),
            ord("C")
        ]:

            if current_corners is None:

                print(
                    "Checkerboard not detected."
                )

                continue


            object_points.append(
                template.copy()
            )

            image_points.append(
                current_corners.copy()
            )


            number = len(
                object_points
            )


            cv2.imwrite(

                str(
                    OUTPUT_DIR
                    /
                    f"calibration_{number:02d}.jpg"
                ),

                frame
            )


            print(
                f"Captured sample {number}"
            )


        # ----------------------------------------------------
        # RESET
        # ----------------------------------------------------

        elif key in [
            ord("r"),
            ord("R")
        ]:

            object_points.clear()
            image_points.clear()

            print(
                "Calibration samples reset."
            )


        # ----------------------------------------------------
        # ENTER
        # ----------------------------------------------------

        elif key in [
            13,
            10
        ]:

            if (
                len(object_points)
                <
                minimum_samples
            ):

                print(
                    f"Need at least "
                    f"{minimum_samples} samples."
                )

                continue

            break


        elif key == 27:

            cap.release()

            cv2.destroyAllWindows()

            return


    cap.release()

    cv2.destroyAllWindows()


    # --------------------------------------------------------
    # CALIBRATE
    # --------------------------------------------------------

    print()
    print(
        "Calculating calibration..."
    )


    rms, K, distortion, rvecs, tvecs = (
        cv2.calibrateCamera(

            object_points,

            image_points,

            frame_size,

            None,

            None
        )
    )


    # --------------------------------------------------------
    # REPROJECTION ERROR
    # --------------------------------------------------------

    errors = []


    for i in range(
        len(object_points)
    ):

        projected, _ = (
            cv2.projectPoints(

                object_points[i],

                rvecs[i],

                tvecs[i],

                K,

                distortion
            )
        )


        error = cv2.norm(

            image_points[i],

            projected,

            cv2.NORM_L2

        ) / len(
            projected
        )


        errors.append(
            error
        )


    mean_error = float(
        np.mean(
            errors
        )
    )


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    np.savez(

        CALIBRATION_FILE,

        camera_matrix=K,

        distortion_coefficients=distortion,

        image_width=frame_size[0],

        image_height=frame_size[1],

        rms_error=rms,

        mean_reprojection_error=mean_error,

        checkerboard_cols=cols,

        checkerboard_rows=rows,

        square_size_cm=square_size_cm
    )


    print()
    print("=" * 70)
    print("CALIBRATION COMPLETE")
    print("=" * 70)

    print()
    print(
        "Resolution:",
        frame_size[0],
        "x",
        frame_size[1]
    )

    print()
    print(
        "RMS error =",
        rms
    )

    print(
        "Mean reprojection error =",
        mean_error
    )


    print()
    print(
        "Camera matrix:"
    )

    print(
        K
    )


    print()
    print(
        "fx =",
        K[0, 0]
    )

    print(
        "fy =",
        K[1, 1]
    )

    print(
        "cx =",
        K[0, 2]
    )

    print(
        "cy =",
        K[1, 2]
    )


    print()
    print(
        "Distortion:"
    )

    print(
        distortion.flatten()
    )


    print()
    print(
        "Saved to:"
    )

    print(
        CALIBRATION_FILE
    )


# ============================================================
# CAPTURE LEFT + RIGHT PHOTOS
# ============================================================

def capture_stereo_photos():

    cap = open_iphone()


    left = None
    right = None


    window = (
        "iPhone Stereo Capture"
    )


    cv2.namedWindow(
        window,
        cv2.WINDOW_NORMAL
    )


    print()
    print(
        "L = capture LEFT"
    )

    print(
        "Move iPhone horizontally."
    )

    print(
        "R = capture RIGHT"
    )

    print(
        "ENTER = continue"
    )


    while True:

        success, frame = (
            cap.read()
        )

        if not success:
            continue


        display = (
            frame.copy()
        )


        if left is None:

            text = (
                "LEFT POSITION - PRESS L"
            )


        elif right is None:

            text = (
                "MOVE RIGHT - PRESS R"
            )


        else:

            text = (
                "BOTH CAPTURED - PRESS ENTER"
            )


        cv2.rectangle(

            display,

            (
                0,
                0
            ),

            (
                display.shape[1],
                80
            ),

            (
                0,
                0,
                0
            ),

            -1
        )


        cv2.putText(

            display,

            text,

            (
                25,
                50
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.9,

            (
                0,
                255,
                255
            ),

            2
        )


        cv2.imshow(
            window,
            display
        )


        key = (
            cv2.waitKey(1)
            &
            0xFF
        )


        if key in [
            ord("l"),
            ord("L")
        ]:

            left = (
                frame.copy()
            )


            cv2.imwrite(

                str(
                    OUTPUT_DIR
                    /
                    "left.jpg"
                ),

                left
            )


            print(
                "LEFT captured."
            )


        elif key in [
            ord("r"),
            ord("R")
        ]:

            if left is None:

                print(
                    "Capture LEFT first."
                )

                continue


            right = (
                frame.copy()
            )


            cv2.imwrite(

                str(
                    OUTPUT_DIR
                    /
                    "right.jpg"
                ),

                right
            )


            print(
                "RIGHT captured."
            )


        elif key in [
            13,
            10
        ]:

            if (
                left is not None
                and
                right is not None
            ):

                break


        elif key == 27:

            cap.release()

            cv2.destroyAllWindows()

            raise RuntimeError(
                "Capture cancelled."
            )


    cap.release()

    cv2.destroyAllWindows()


    return (
        left,
        right
    )


# ============================================================
# IMAGE DISPLAY RESIZE
# ============================================================

def resize_for_display(
    image,
    max_width=1000,
    max_height=750
):

    height, width = (
        image.shape[:2]
    )


    scale = min(

        max_width / width,

        max_height / height,

        1.0
    )


    display_width = int(
        width * scale
    )

    display_height = int(
        height * scale
    )


    image_display = cv2.resize(

        image,

        (
            display_width,
            display_height
        )
    )


    return (

        image_display,

        width / display_width,

        height / display_height
    )


# ============================================================
# CLICK POINTS
# ============================================================

def select_points(
    image,
    count,
    window_name,
    labels,
    close_shape=False
):

    display, sx, sy = (
        resize_for_display(
            image
        )
    )


    points = []


    def redraw():

        canvas = (
            display.copy()
        )


        for i, point in enumerate(
            points
        ):

            x, y = point


            cv2.circle(

                canvas,

                (
                    x,
                    y
                ),

                8,

                (
                    0,
                    255,
                    0
                ),

                -1
            )


            cv2.putText(

                canvas,

                labels[i],

                (
                    x + 10,
                    y - 10
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (
                    255,
                    255,
                    255
                ),

                2
            )


        if len(points) > 1:

            for i in range(
                len(points) - 1
            ):

                cv2.line(

                    canvas,

                    points[i],

                    points[i + 1],

                    (
                        0,
                        255,
                        255
                    ),

                    2
                )


        if (
            close_shape
            and
            len(points) == count
        ):

            cv2.line(

                canvas,

                points[-1],

                points[0],

                (
                    0,
                    255,
                    255
                ),

                2
            )


        cv2.rectangle(

            canvas,

            (
                0,
                0
            ),

            (
                canvas.shape[1],
                75
            ),

            (
                0,
                0,
                0
            ),

            -1
        )


        cv2.putText(

            canvas,

            f"Selected {len(points)}/{count}",

            (
                20,
                30
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (
                0,
                255,
                255
            ),

            2
        )


        cv2.putText(

            canvas,

            "Click | R=reset | ENTER=continue",

            (
                20,
                60
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (
                255,
                255,
                255
            ),

            1
        )


        cv2.imshow(
            window_name,
            canvas
        )


    def mouse(
        event,
        x,
        y,
        flags,
        param
    ):

        if (
            event
            ==
            cv2.EVENT_LBUTTONDOWN

            and

            len(points)
            <
            count
        ):

            points.append(
                (
                    x,
                    y
                )
            )

            redraw()


    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )


    cv2.setMouseCallback(
        window_name,
        mouse
    )


    redraw()


    while True:

        key = (
            cv2.waitKey(20)
            &
            0xFF
        )


        if key in [
            13,
            10
        ]:

            if len(points) == count:

                break


        elif key in [
            ord("r"),
            ord("R")
        ]:

            points.clear()

            redraw()


        elif key == 27:

            cv2.destroyAllWindows()

            raise RuntimeError(
                "Selection cancelled."
            )


    cv2.destroyWindow(
        window_name
    )


    return [

        (
            x * sx,
            y * sy
        )

        for x, y
        in points
    ]


# ============================================================
# UNDISTORT CLICKED PIXELS
# ============================================================

def undistort_pixel_points(
    points,
    K,
    distortion
):

    points_array = np.array(

        points,

        dtype=np.float64

    ).reshape(
        -1,
        1,
        2
    )


    corrected = cv2.undistortPoints(

        points_array,

        K,

        distortion,

        P=K
    )


    return corrected.reshape(
        -1,
        2
    )


# ============================================================
# STEREO 3D RECONSTRUCTION
# ============================================================

def reconstruct_points(
    left_points,
    right_points,
    K,
    distortion,
    baseline_cm
):

    left_corrected = (
        undistort_pixel_points(

            left_points,

            K,

            distortion
        )
    )


    right_corrected = (
        undistort_pixel_points(

            right_points,

            K,

            distortion
        )
    )


    fx = K[0, 0]
    fy = K[1, 1]

    cx = K[0, 2]
    cy = K[1, 2]


    results = []


    for left, right in zip(

        left_corrected,

        right_corrected
    ):

        x_left = left[0]
        y_left = left[1]

        x_right = right[0]


        disparity = abs(

            x_left
            -
            x_right
        )


        if disparity < 0.01:

            raise RuntimeError(
                "Stereo disparity is too small."
            )


        Z = (

            fx
            *
            baseline_cm

        ) / disparity


        X = (

            (
                x_left
                -
                cx
            )

            *
            Z

        ) / fx


        Y = (

            (
                y_left
                -
                cy
            )

            *
            Z

        ) / fy


        results.append(

            np.array(

                [
                    X,
                    Y,
                    Z
                ],

                dtype=np.float64
            )
        )


    return (

        np.array(
            results
        ),

        left_corrected,

        right_corrected
    )


# ============================================================
# DISTANCE
# ============================================================

def distance_3d(
    a,
    b
):

    return float(

        np.linalg.norm(
            a - b
        )
    )


# ============================================================
# PROBLEM 1
# ============================================================

def problem_1():

    print()
    print("=" * 70)
    print("PROBLEM 1")
    print("CALIBRATED STEREO OBJECT SIZE")
    print("=" * 70)


    K_original, distortion, cal_w, cal_h = (
        load_calibration()
    )


    baseline_text = input(

        f"\nBaseline in cm "
        f"[{DEFAULT_BASELINE_CM}]: "

    ).strip()


    baseline_cm = (

        float(
            baseline_text
        )

        if baseline_text

        else DEFAULT_BASELINE_CM
    )


    left_image, right_image = (
        capture_stereo_photos()
    )


    height, width = (
        left_image.shape[:2]
    )


    K = scale_camera_matrix(

        K_original,

        cal_w,

        cal_h,

        width,

        height
    )


    print()
    print(
        "Image resolution:",
        width,
        "x",
        height
    )


    print()
    print(
        "Using fx =",
        K[0, 0]
    )


    print()
    print(
        "1 = Rectangle"
    )

    print(
        "2 = Circle"
    )

    print(
        "3 = Polygon"
    )


    shape = input(
        "Shape: "
    ).strip()


    if shape == "1":

        name = "rectangle"

        count = 4

        labels = [

            "TL",
            "TR",
            "BR",
            "BL"
        ]

        close = True


    elif shape == "2":

        name = "circle"

        count = 2

        labels = [

            "D1",
            "D2"
        ]

        close = False


    elif shape == "3":

        name = "polygon"

        count = int(

            input(
                "Number of vertices: "
            )
        )

        labels = [

            f"P{i + 1}"

            for i in range(
                count
            )
        ]

        close = True


    else:

        raise RuntimeError(
            "Invalid shape."
        )


    print()
    print(
        "Select points on LEFT."
    )


    left_points = select_points(

        left_image,

        count,

        "LEFT OBJECT",

        labels,

        close
    )


    print()
    print(
        "Select SAME points on RIGHT."
    )


    right_points = select_points(

        right_image,

        count,

        "RIGHT OBJECT",

        labels,

        close
    )


    points_3d, corrected_left, corrected_right = (
        reconstruct_points(

            left_points,

            right_points,

            K,

            distortion,

            baseline_cm
        )
    )


    print()
    print("=" * 85)

    print(
        "3D POINT RESULTS"
    )

    print("=" * 85)


    print(

        f"{'Point':<10}"

        f"{'d px':>12}"

        f"{'X cm':>15}"

        f"{'Y cm':>15}"

        f"{'Z cm':>15}"
    )


    print("-" * 85)


    for i in range(
        count
    ):

        disparity = abs(

            corrected_left[i][0]

            -

            corrected_right[i][0]
        )


        print(

            f"{labels[i]:<10}"

            f"{disparity:>12.3f}"

            f"{points_3d[i][0]:>15.3f}"

            f"{points_3d[i][1]:>15.3f}"

            f"{points_3d[i][2]:>15.3f}"
        )


    mean_z = float(

        np.mean(
            points_3d[:, 2]
        )
    )


    print()
    print("=" * 70)

    print(
        "DEPTH"
    )

    print("=" * 70)


    print(

        f"Estimated Z = "

        f"{mean_z:.3f} cm"
    )


    print(

        f"Estimated Z = "

        f"{mean_z / 100:.3f} m"
    )


    # --------------------------------------------------------
    # RECTANGLE
    # --------------------------------------------------------

    if name == "rectangle":

        top = distance_3d(

            points_3d[0],

            points_3d[1]
        )


        right_edge = distance_3d(

            points_3d[1],

            points_3d[2]
        )


        bottom = distance_3d(

            points_3d[3],

            points_3d[2]
        )


        left_edge = distance_3d(

            points_3d[0],

            points_3d[3]
        )


        object_width = (

            top
            +
            bottom

        ) / 2


        object_height = (

            left_edge
            +
            right_edge

        ) / 2


        print()
        print("=" * 70)

        print(
            "RECTANGLE DIMENSIONS"
        )

        print("=" * 70)


        print(

            f"Top    = "
            f"{top:.3f} cm"
        )

        print(

            f"Bottom = "
            f"{bottom:.3f} cm"
        )

        print(

            f"Left   = "
            f"{left_edge:.3f} cm"
        )

        print(

            f"Right  = "
            f"{right_edge:.3f} cm"
        )


        print()

        print(

            f"WIDTH  = "
            f"{object_width:.3f} cm"
        )

        print(

            f"HEIGHT = "
            f"{object_height:.3f} cm"
        )


    # --------------------------------------------------------
    # CIRCLE
    # --------------------------------------------------------

    elif name == "circle":

        diameter = distance_3d(

            points_3d[0],

            points_3d[1]
        )


        print()
        print(

            f"DIAMETER = "
            f"{diameter:.3f} cm"
        )


    # --------------------------------------------------------
    # POLYGON
    # --------------------------------------------------------

    else:

        print()
        print(
            "POLYGON EDGE DIMENSIONS"
        )


        for i in range(
            count
        ):

            j = (

                i + 1

            ) % count


            edge = distance_3d(

                points_3d[i],

                points_3d[j]
            )


            print(

                f"{labels[i]} -> "
                f"{labels[j]} = "

                f"{edge:.3f} cm"
            )


# ============================================================
# RECORD VIDEO FROM IPHONE
# ============================================================

def record_video(
    filename,
    title
):

    cap = open_iphone()


    success, frame = (
        cap.read()
    )


    if not success:

        cap.release()

        raise RuntimeError(
            "Could not read camera."
        )


    height, width = (
        frame.shape[:2]
    )


    fps = 30.0


    writer = cv2.VideoWriter(

        str(filename),

        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),

        fps,

        (
            width,
            height
        )
    )


    print()
    print(
        title
    )

    print(
        "Press S to START recording."
    )

    print(
        "Press E to END recording."
    )


    recording = False


    while True:

        success, frame = (
            cap.read()
        )

        if not success:
            continue


        display = (
            frame.copy()
        )


        if recording:

            writer.write(
                frame
            )

            text = (
                "RECORDING - PRESS E TO STOP"
            )

            color = (
                0,
                0,
                255
            )


        else:

            text = (
                "PRESS S TO START"
            )

            color = (
                0,
                255,
                255
            )


        cv2.putText(

            display,

            text,

            (
                30,
                50
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.9,

            color,

            2
        )


        cv2.imshow(

            title,

            display
        )


        key = (
            cv2.waitKey(1)
            &
            0xFF
        )


        if key in [
            ord("s"),
            ord("S")
        ]:

            recording = True


        elif key in [
            ord("e"),
            ord("E")
        ]:

            if recording:

                break


        elif key == 27:

            break


    cap.release()

    writer.release()

    cv2.destroyAllWindows()


    return str(
        filename
    )


# ============================================================
# EXTRACT MIDDLE VIDEO FRAME
# ============================================================

def extract_middle_frame(
    video_path
):

    cap = cv2.VideoCapture(
        video_path
    )


    if not cap.isOpened():

        raise RuntimeError(
            "Could not open video."
        )


    count = int(

        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    middle = (
        count // 2
    )


    cap.set(

        cv2.CAP_PROP_POS_FRAMES,

        middle
    )


    success, frame = (
        cap.read()
    )


    cap.release()


    if not success:

        raise RuntimeError(
            "Could not extract video frame."
        )


    return frame


def problem_3():

    print()
    print("=" * 70)

    print(
        "PROBLEM 3"
    )

    print(
        "ROOM / WALL DIMENSION USING STEREO VIDEOS"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # LOAD CALIBRATION
    # --------------------------------------------------------

    K_original, distortion, cal_w, cal_h = (
        load_calibration()
    )


    # --------------------------------------------------------
    # VIDEO FILES
    # --------------------------------------------------------

    left_video_path = Path("wall/left.MOV")
    right_video_path = Path("wall/right.MOV")


    if not left_video_path.exists():

        raise RuntimeError(
            f"Left video not found: {left_video_path}"
        )


    if not right_video_path.exists():

        raise RuntimeError(
            f"Right video not found: {right_video_path}"
        )


    print()
    print(
        "Using videos:"
    )

    print(
        "LEFT  =",
        left_video_path
    )

    print(
        "RIGHT =",
        right_video_path
    )


    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    baseline_text = input(

        f"\nBaseline in cm "
        f"[{DEFAULT_BASELINE_CM}]: "

    ).strip()


    baseline_cm = (

        float(
            baseline_text
        )

        if baseline_text

        else DEFAULT_BASELINE_CM
    )


    print()
    print(
        f"Baseline = {baseline_cm:.2f} cm"
    )


    # --------------------------------------------------------
    # OPEN LEFT VIDEO
    # --------------------------------------------------------

    left_cap = cv2.VideoCapture(
        str(
            left_video_path
        )
    )


    if not left_cap.isOpened():

        raise RuntimeError(
            "Could not open wall/left.mp4"
        )


    left_frame_count = int(

        left_cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    left_fps = left_cap.get(
        cv2.CAP_PROP_FPS
    )


    left_width = int(

        left_cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )


    left_height = int(

        left_cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )


    # --------------------------------------------------------
    # OPEN RIGHT VIDEO
    # --------------------------------------------------------

    right_cap = cv2.VideoCapture(
        str(
            right_video_path
        )
    )


    if not right_cap.isOpened():

        left_cap.release()

        raise RuntimeError(
            "Could not open wall/right.mp4"
        )


    right_frame_count = int(

        right_cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    right_fps = right_cap.get(
        cv2.CAP_PROP_FPS
    )


    right_width = int(

        right_cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )


    right_height = int(

        right_cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )


    # --------------------------------------------------------
    # CHECK RESOLUTION
    # --------------------------------------------------------

    if (
        left_width != right_width
        or
        left_height != right_height
    ):

        left_cap.release()
        right_cap.release()

        raise RuntimeError(
            "LEFT and RIGHT videos must have the same resolution."
        )


    print()
    print(
        "LEFT VIDEO"
    )

    print(
        f"Frames = {left_frame_count}"
    )

    print(
        f"FPS = {left_fps:.2f}"
    )

    print(
        f"Resolution = {left_width} x {left_height}"
    )


    print()
    print(
        "RIGHT VIDEO"
    )

    print(
        f"Frames = {right_frame_count}"
    )

    print(
        f"FPS = {right_fps:.2f}"
    )

    print(
        f"Resolution = {right_width} x {right_height}"
    )


    # --------------------------------------------------------
    # USE MIDDLE FRAME
    # --------------------------------------------------------

    left_middle = (
        left_frame_count // 2
    )

    right_middle = (
        right_frame_count // 2
    )


    left_cap.set(

        cv2.CAP_PROP_POS_FRAMES,

        left_middle
    )


    success_left, left_image = (
        left_cap.read()
    )


    right_cap.set(

        cv2.CAP_PROP_POS_FRAMES,

        right_middle
    )


    success_right, right_image = (
        right_cap.read()
    )


    left_cap.release()
    right_cap.release()


    if not success_left:

        raise RuntimeError(
            "Could not extract frame from LEFT video."
        )


    if not success_right:

        raise RuntimeError(
            "Could not extract frame from RIGHT video."
        )


    # --------------------------------------------------------
    # SAVE EXTRACTED FRAMES
    # --------------------------------------------------------

    cv2.imwrite(

        str(
            OUTPUT_DIR
            /
            "wall_left_frame.jpg"
        ),

        left_image
    )


    cv2.imwrite(

        str(
            OUTPUT_DIR
            /
            "wall_right_frame.jpg"
        ),

        right_image
    )


    print()
    print(
        "Extracted middle frames:"
    )

    print(
        OUTPUT_DIR
        /
        "wall_left_frame.jpg"
    )

    print(
        OUTPUT_DIR
        /
        "wall_right_frame.jpg"
    )


    # --------------------------------------------------------
    # SCALE CAMERA MATRIX
    # --------------------------------------------------------

    K = scale_camera_matrix(

        K_original,

        cal_w,

        cal_h,

        left_width,

        left_height
    )


    print()
    print(
        "Using calibrated fx =",
        K[0, 0]
    )


    # --------------------------------------------------------
    # CHOOSE ROOM DIMENSION
    # --------------------------------------------------------

    print()
    print(
        "What are you measuring?"
    )

    print(
        "1 = Wall / Room Width"
    )

    print(
        "2 = Wall / Room Height"
    )

    print(
        "3 = Room Length / Visible Edge"
    )


    measurement_choice = input(

        "Choose 1 / 2 / 3: "

    ).strip()


    if measurement_choice == "1":

        measurement_name = (
            "WIDTH"
        )

        labels = [
            "LEFT",
            "RIGHT"
        ]


    elif measurement_choice == "2":

        measurement_name = (
            "HEIGHT"
        )

        labels = [
            "BOTTOM",
            "TOP"
        ]


    elif measurement_choice == "3":

        measurement_name = (
            "LENGTH"
        )

        labels = [
            "NEAR",
            "FAR"
        ]


    else:

        raise RuntimeError(
            "Invalid measurement option."
        )


    # --------------------------------------------------------
    # SELECT TWO POINTS IN LEFT FRAME
    # --------------------------------------------------------

    print()
    print(
        f"Select the two endpoints for {measurement_name}"
    )

    print(
        "on the LEFT video frame."
    )


    left_points = select_points(

        left_image,

        2,

        f"LEFT FRAME - {measurement_name}",

        labels,

        False
    )


    # --------------------------------------------------------
    # SELECT SAME TWO POINTS IN RIGHT FRAME
    # --------------------------------------------------------

    print()
    print(
        "Now select the SAME two physical points"
    )

    print(
        "in the SAME order on the RIGHT frame."
    )


    right_points = select_points(

        right_image,

        2,

        f"RIGHT FRAME - {measurement_name}",

        labels,

        False
    )


    # --------------------------------------------------------
    # RECONSTRUCT 3D POINTS
    # --------------------------------------------------------

    points_3d, corrected_left, corrected_right = (
        reconstruct_points(

            left_points,

            right_points,

            K,

            distortion,

            baseline_cm
        )
    )


    # --------------------------------------------------------
    # DISPARITIES
    # --------------------------------------------------------

    disparity_1 = abs(

        corrected_left[0][0]

        -

        corrected_right[0][0]
    )


    disparity_2 = abs(

        corrected_left[1][0]

        -

        corrected_right[1][0]
    )


    # --------------------------------------------------------
    # DEPTHS
    # --------------------------------------------------------

    z1 = points_3d[0][2]
    z2 = points_3d[1][2]


    # --------------------------------------------------------
    # DIMENSION
    # --------------------------------------------------------

    estimated_dimension = distance_3d(

        points_3d[0],

        points_3d[1]
    )


    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    print()
    print("=" * 80)

    print(
        "STEREO RECONSTRUCTION"
    )

    print("=" * 80)


    print()

    print(
        f"{labels[0]} point:"
    )

    print(
        f"Disparity = {disparity_1:.3f} pixels"
    )

    print(
        f"X = {points_3d[0][0]:.3f} cm"
    )

    print(
        f"Y = {points_3d[0][1]:.3f} cm"
    )

    print(
        f"Z = {z1:.3f} cm"
    )


    print()

    print(
        f"{labels[1]} point:"
    )

    print(
        f"Disparity = {disparity_2:.3f} pixels"
    )

    print(
        f"X = {points_3d[1][0]:.3f} cm"
    )

    print(
        f"Y = {points_3d[1][1]:.3f} cm"
    )

    print(
        f"Z = {z2:.3f} cm"
    )


    print()
    print("=" * 70)

    print(
        f"ESTIMATED ROOM {measurement_name}"
    )

    print("=" * 70)


    print()

    print(
        f"{measurement_name} = "
        f"{estimated_dimension:.2f} cm"
    )


    print(
        f"{measurement_name} = "
        f"{estimated_dimension / 100:.3f} m"
    )


    print(
        f"{measurement_name} = "
        f"{estimated_dimension / 30.48:.3f} ft"
    )


    # --------------------------------------------------------
    # OPTIONAL ACTUAL VALUE
    # --------------------------------------------------------

    actual_text = input(

        f"\nActual {measurement_name.lower()} in cm "
        f"(Enter to skip): "

    ).strip()


    if actual_text:

        actual_value = float(
            actual_text
        )


        error = (

            abs(
                estimated_dimension
                -
                actual_value
            )

            /
            actual_value

            *
            100
        )


        print()

        print(
            f"Actual {measurement_name} = "
            f"{actual_value:.2f} cm"
        )

        print(
            f"Absolute error = "
            f"{abs(estimated_dimension - actual_value):.2f} cm"
        )

        print(
            f"Percentage error = "
            f"{error:.2f}%"
        )


    print()
    print(
        "Problem 3 finished."
    )


# ============================================================
# MAIN MENU
# ============================================================

def main():

    while True:

        print()
        print("=" * 70)

        print(
            "CSc 8830 - ASSIGNMENT 7"
        )

        print("=" * 70)

        print()
        print(
            "1 - Calibrate iPhone Camera"
        )

        print(
            "2 - Problem 1: Object Size"
        )

        print(
            "3 - Problem 3: Room Dimensions"
        )

        print(
            "Q - Quit"
        )


        choice = input(
            "\nChoose: "
        ).strip().lower()


        try:

            if choice == "1":

                calibrate_iphone()


            elif choice == "2":

                problem_1()


            elif choice == "3":

                problem_3()


            elif choice == "q":

                print(
                    "Finished."
                )

                break


            else:

                print(
                    "Invalid option."
                )


        except Exception as error:

            print()
            print("=" * 70)

            print(
                "ERROR"
            )

            print("=" * 70)

            print(
                error
            )


        finally:

            cv2.destroyAllWindows()


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()