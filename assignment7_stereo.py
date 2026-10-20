"""
CSc 8830 - Computer Vision
Assignment 7

Problem 1:
Calibrated Stereo Object Size Estimation

Problem 3:
Room Dimension Estimation Using Stereo Videos


PROBLEM 1 WORKFLOW

1. Open iPhone camera.
2. Press L to capture LEFT image.
3. Move iPhone horizontally to the right by the measured baseline.
4. Press R to capture RIGHT image.
5. Press ENTER.
6. Select matching object points.
7. Calculate disparity.
8. Calculate Z.
9. Calculate physical object dimensions.


STEREO EQUATIONS

d = x_left - x_right

Z = fx * B / d

X = (x - cx) * Z / fx

Y = (y - cy) * Z / fy


3D distance:

distance = sqrt(
    (X2-X1)^2 +
    (Y2-Y1)^2 +
    (Z2-Z1)^2
)
"""

import cv2
import numpy as np
import math
from pathlib import Path


# ============================================================
# CAMERA SETTINGS
# ============================================================

# Change ONLY this if your iPhone uses another camera index.
IPHONE_CAMERA_INDEX = 1


# Existing camera calibration
FX = 2301.4244
FY = 2304.4655

CX = 973.7959
CY = 494.0908

CALIB_WIDTH = 1920
CALIB_HEIGHT = 1080


# Default stereo baseline
DEFAULT_BASELINE_CM = 15.0


# Output folder
OUTPUT_DIR = Path("assignment7_outputs")

OUTPUT_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# SCALE CAMERA CALIBRATION
# ============================================================

def scaled_intrinsics(
    width,
    height
):

    scale_x = width / CALIB_WIDTH
    scale_y = height / CALIB_HEIGHT

    fx = FX * scale_x
    fy = FY * scale_y

    cx = CX * scale_x
    cy = CY * scale_y

    return (
        fx,
        fy,
        cx,
        cy
    )


# ============================================================
# CAPTURE LEFT AND RIGHT IMAGES FROM IPHONE
# ============================================================

def capture_stereo_pair_from_iphone():

    print()
    print("=" * 70)
    print("OPENING IPHONE CAMERA")
    print("=" * 70)

    cap = cv2.VideoCapture(
        IPHONE_CAMERA_INDEX,
        cv2.CAP_AVFOUNDATION
    )


    if not cap.isOpened():

        raise RuntimeError(

            "Could not open iPhone camera.\n\n"

            "Try changing:\n"

            "IPHONE_CAMERA_INDEX = 1\n\n"

            "to 0, 2, or 3."
        )


    # Ask for 1920 x 1080
    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1920
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        1080
    )


    left_image = None
    right_image = None


    window_name = "iPhone Stereo Camera"


    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )


    print()
    print("STEP 1")
    print("Put the iPhone at the LEFT position.")
    print()
    print("Press L to take the LEFT photo.")
    print()
    print("Then move the iPhone horizontally to the RIGHT.")
    print()
    print("Press R to take the RIGHT photo.")
    print()
    print("Press ENTER after both images are captured.")
    print()


    while True:

        success, frame = cap.read()


        if not success:
            continue


        display = frame.copy()


        # ----------------------------------------------------
        # DISPLAY INSTRUCTIONS
        # ----------------------------------------------------

        if left_image is None:

            instruction = (
                "LEFT POSITION - Press L"
            )


        elif right_image is None:

            instruction = (
                "Move camera RIGHT - Press R"
            )


        else:

            instruction = (
                "LEFT + RIGHT SAVED - Press ENTER"
            )


        cv2.rectangle(

            display,

            (0, 0),

            (
                display.shape[1],
                100
            ),

            (0, 0, 0),

            -1
        )


        cv2.putText(

            display,

            instruction,

            (30, 45),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            (0, 255, 255),

            2
        )


        if left_image is not None:

            cv2.putText(

                display,

                "LEFT SAVED",

                (30, 80),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0, 255, 0),

                2
            )


        if right_image is not None:

            cv2.putText(

                display,

                "RIGHT SAVED",

                (220, 80),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0, 0, 255),

                2
            )


        cv2.imshow(
            window_name,
            display
        )


        key = cv2.waitKey(1) & 0xFF


        # ----------------------------------------------------
        # CAPTURE LEFT
        # ----------------------------------------------------

        if key in [
            ord("l"),
            ord("L")
        ]:

            left_image = frame.copy()


            cv2.imwrite(

                str(
                    OUTPUT_DIR
                    /
                    "left.jpg"
                ),

                left_image
            )


            print()
            print("LEFT PHOTO CAPTURED.")
            print()
            print(
                "Now move the iPhone horizontally "
                "to the RIGHT."
            )


        # ----------------------------------------------------
        # CAPTURE RIGHT
        # ----------------------------------------------------

        elif key in [
            ord("r"),
            ord("R")
        ]:

            if left_image is None:

                print(
                    "Capture LEFT image first."
                )

                continue


            right_image = frame.copy()


            cv2.imwrite(

                str(
                    OUTPUT_DIR
                    /
                    "right.jpg"
                ),

                right_image
            )


            print()
            print("RIGHT PHOTO CAPTURED.")
            print()
            print("Press ENTER.")


        # ----------------------------------------------------
        # FINISH
        # ----------------------------------------------------

        elif key in [
            13,
            10
        ]:

            if (
                left_image is not None
                and
                right_image is not None
            ):

                break


        # ----------------------------------------------------
        # ESC TO CANCEL
        # ----------------------------------------------------

        elif key == 27:

            cap.release()

            cv2.destroyAllWindows()

            raise RuntimeError(
                "Capture cancelled."
            )


    cap.release()

    cv2.destroyAllWindows()


    print()
    print("Images saved:")
    print(
        OUTPUT_DIR
        /
        "left.jpg"
    )

    print(
        OUTPUT_DIR
        /
        "right.jpg"
    )


    return (
        left_image,
        right_image
    )


# ============================================================
# RESIZE IMAGE FOR SCREEN
# ============================================================

def resize_for_display(
    image,
    max_width=1000,
    max_height=750
):

    height, width = image.shape[:2]


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


    resized = cv2.resize(

        image,

        (
            display_width,
            display_height
        ),

        interpolation=cv2.INTER_AREA
    )


    scale_x = (
        width
        /
        display_width
    )

    scale_y = (
        height
        /
        display_height
    )


    return (
        resized,
        scale_x,
        scale_y
    )


# ============================================================
# SELECT POINTS
# ============================================================

def select_points(
    image,
    number_of_points,
    window_name,
    labels,
    close_shape=False
):

    display_image, scale_x, scale_y = (
        resize_for_display(
            image
        )
    )


    selected_points = []


    def redraw():

        canvas = display_image.copy()


        for index, point in enumerate(
            selected_points
        ):

            x, y = point


            cv2.circle(

                canvas,

                (x, y),

                8,

                (0, 255, 0),

                -1
            )


            cv2.circle(

                canvas,

                (x, y),

                10,

                (255, 255, 255),

                2
            )


            cv2.putText(

                canvas,

                labels[index],

                (
                    x + 12,
                    y - 12
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.75,

                (255, 255, 255),

                2
            )


        # Draw lines
        if len(selected_points) > 1:

            for i in range(
                len(selected_points) - 1
            ):

                cv2.line(

                    canvas,

                    selected_points[i],

                    selected_points[i + 1],

                    (0, 255, 255),

                    2
                )


        if (
            close_shape
            and
            len(selected_points)
            ==
            number_of_points
        ):

            cv2.line(

                canvas,

                selected_points[-1],

                selected_points[0],

                (0, 255, 255),

                2
            )


        cv2.rectangle(

            canvas,

            (0, 0),

            (
                canvas.shape[1],
                80
            ),

            (0, 0, 0),

            -1
        )


        cv2.putText(

            canvas,

            f"Selected: "
            f"{len(selected_points)}"
            f"/{number_of_points}",

            (20, 30),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (0, 255, 255),

            2
        )


        cv2.putText(

            canvas,

            "Click = add | R = reset | ENTER = continue",

            (20, 60),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (255, 255, 255),

            1
        )


        cv2.imshow(
            window_name,
            canvas
        )


    # --------------------------------------------------------
    # MOUSE CALLBACK
    # --------------------------------------------------------

    def mouse_callback(
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
        ):

            if (
                len(selected_points)
                <
                number_of_points
            ):

                selected_points.append(
                    (x, y)
                )

                redraw()


    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )


    cv2.setMouseCallback(
        window_name,
        mouse_callback
    )


    redraw()


    while True:

        key = cv2.waitKey(20) & 0xFF


        if key in [
            13,
            10
        ]:

            if (
                len(selected_points)
                ==
                number_of_points
            ):

                break


        elif key in [
            ord("r"),
            ord("R")
        ]:

            selected_points.clear()

            redraw()


        elif key == 27:

            cv2.destroyAllWindows()

            raise RuntimeError(
                "Point selection cancelled."
            )


    cv2.destroyWindow(
        window_name
    )


    # Convert screen coordinates
    # back into original image pixels
    original_points = []


    for x, y in selected_points:

        original_points.append(

            (

                x * scale_x,

                y * scale_y

            )
        )


    return original_points


# ============================================================
# RECONSTRUCT ONE 3D POINT
# ============================================================

def reconstruct_3d(
    left_point,
    right_point,
    fx,
    fy,
    cx,
    cy,
    baseline_cm
):

    x_left, y_left = (
        left_point
    )

    x_right, y_right = (
        right_point
    )


    # --------------------------------------------------------
    # DISPARITY
    # --------------------------------------------------------

    disparity = (
        x_left
        -
        x_right
    )


    if disparity <= 0.01:

        raise ValueError(

            "\nInvalid disparity.\n\n"

            f"x_left  = {x_left:.2f}\n"

            f"x_right = {x_right:.2f}\n"

            f"d = {disparity:.2f}\n\n"

            "Check that:\n"

            "1. You clicked the same point.\n"

            "2. You moved the camera to the RIGHT.\n"

            "3. You did not rotate the phone."
        )


    # --------------------------------------------------------
    # Z
    # --------------------------------------------------------

    Z = (

        fx
        *
        baseline_cm

    ) / disparity


    # --------------------------------------------------------
    # X
    # --------------------------------------------------------

    X = (

        (
            x_left
            -
            cx
        )

        *
        Z

    ) / fx


    # --------------------------------------------------------
    # Y
    # --------------------------------------------------------

    Y = (

        (
            y_left
            -
            cy
        )

        *
        Z

    ) / fy


    return {

        "X": X,

        "Y": Y,

        "Z": Z,

        "disparity": disparity
    }


# ============================================================
# 3D DISTANCE
# ============================================================

def distance_3d(
    point1,
    point2
):

    return math.sqrt(

        (
            point2["X"]
            -
            point1["X"]
        ) ** 2

        +

        (
            point2["Y"]
            -
            point1["Y"]
        ) ** 2

        +

        (
            point2["Z"]
            -
            point1["Z"]
        ) ** 2
    )


# ============================================================
# ERROR
# ============================================================

def percentage_error(
    estimated,
    actual
):

    return (

        abs(
            estimated
            -
            actual
        )

        /
        actual

        *
        100
    )


# ============================================================
# PROBLEM 1
# ============================================================

def problem_1():

    print()
    print("=" * 75)

    print(
        "PROBLEM 1 - CALIBRATED STEREO OBJECT SIZE ESTIMATION"
    )

    print("=" * 75)


    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    baseline_input = input(

        "\nHow far will you move the iPhone "
        "to the right in cm [15]: "

    ).strip()


    if baseline_input:

        baseline_cm = float(
            baseline_input
        )

    else:

        baseline_cm = (
            DEFAULT_BASELINE_CM
        )


    print()
    print(
        "Baseline B =",
        baseline_cm,
        "cm"
    )


    # --------------------------------------------------------
    # TAKE PHOTOS
    # --------------------------------------------------------

    left_image, right_image = (
        capture_stereo_pair_from_iphone()
    )


    if (
        left_image.shape
        !=
        right_image.shape
    ):

        raise RuntimeError(

            "Left and right images "
            "do not have the same resolution."
        )


    image_height, image_width = (
        left_image.shape[:2]
    )


    print()
    print(
        "Image resolution =",
        image_width,
        "x",
        image_height
    )


    # --------------------------------------------------------
    # CAMERA CALIBRATION
    # --------------------------------------------------------

    fx, fy, cx, cy = (
        scaled_intrinsics(

            image_width,

            image_height
        )
    )


    print()
    print("Camera Parameters")

    print(
        "fx =",
        round(
            fx,
            4
        )
    )

    print(
        "fy =",
        round(
            fy,
            4
        )
    )

    print(
        "cx =",
        round(
            cx,
            4
        )
    )

    print(
        "cy =",
        round(
            cy,
            4
        )
    )


    # --------------------------------------------------------
    # OBJECT NAME
    # --------------------------------------------------------

    object_name = input(
        "\nObject name: "
    ).strip()


    # --------------------------------------------------------
    # SHAPE
    # --------------------------------------------------------

    print()
    print(
        "1 - Rectangle"
    )

    print(
        "2 - Circle"
    )

    print(
        "3 - Polygon"
    )


    choice = input(
        "Object shape: "
    ).strip()


    # --------------------------------------------------------
    # RECTANGLE
    # --------------------------------------------------------

    if choice == "1":

        shape = "rectangle"

        number_of_points = 4

        labels = [

            "TL",

            "TR",

            "BR",

            "BL"
        ]

        close_shape = True


    # --------------------------------------------------------
    # CIRCLE
    # --------------------------------------------------------

    elif choice == "2":

        shape = "circle"

        number_of_points = 2

        labels = [

            "D1",

            "D2"
        ]

        close_shape = False


    # --------------------------------------------------------
    # POLYGON
    # --------------------------------------------------------

    elif choice == "3":

        shape = "polygon"

        number_of_points = int(

            input(
                "Number of visible polygon vertices: "
            )
        )


        labels = [

            f"P{i + 1}"

            for i
            in range(
                number_of_points
            )
        ]


        close_shape = True


    else:

        raise ValueError(
            "Invalid shape choice."
        )


    # --------------------------------------------------------
    # LEFT POINTS
    # --------------------------------------------------------

    print()
    print(
        "Select points on LEFT image."
    )


    left_points = select_points(

        left_image,

        number_of_points,

        "LEFT IMAGE",

        labels,

        close_shape
    )


    # --------------------------------------------------------
    # RIGHT POINTS
    # --------------------------------------------------------

    print()
    print(
        "Select the EXACT SAME points "
        "in the SAME order on RIGHT image."
    )


    right_points = select_points(

        right_image,

        number_of_points,

        "RIGHT IMAGE",

        labels,

        close_shape
    )


    # --------------------------------------------------------
    # RECONSTRUCT POINTS
    # --------------------------------------------------------

    points_3d = []


    for left_point, right_point in zip(

        left_points,

        right_points
    ):

        result = reconstruct_3d(

            left_point,

            right_point,

            fx,

            fy,

            cx,

            cy,

            baseline_cm
        )


        points_3d.append(
            result
        )


    # --------------------------------------------------------
    # PRINT EACH POINT
    # --------------------------------------------------------

    print()
    print("=" * 95)

    print(
        "STEREO POINT RESULTS"
    )

    print("=" * 95)


    print(

        f"{'Point':<8}"

        f"{'xL':>10}"

        f"{'xR':>10}"

        f"{'d':>10}"

        f"{'X cm':>12}"

        f"{'Y cm':>12}"

        f"{'Z cm':>12}"
    )


    print("-" * 95)


    for index, point in enumerate(
        points_3d
    ):

        print(

            f"{labels[index]:<8}"

            f"{left_points[index][0]:>10.2f}"

            f"{right_points[index][0]:>10.2f}"

            f"{point['disparity']:>10.2f}"

            f"{point['X']:>12.2f}"

            f"{point['Y']:>12.2f}"

            f"{point['Z']:>12.2f}"
        )


    # --------------------------------------------------------
    # MEAN DEPTH
    # --------------------------------------------------------

    mean_disparity = np.mean(

        [

            point["disparity"]

            for point
            in points_3d

        ]
    )


    mean_z = np.mean(

        [

            point["Z"]

            for point
            in points_3d

        ]
    )


    print()
    print("=" * 70)

    print(
        "DEPTH ESTIMATION"
    )

    print("=" * 70)


    print(

        "Mean disparity d =",

        round(
            mean_disparity,
            3
        ),

        "pixels"
    )


    print(

        "Estimated Z =",

        round(
            mean_z,
            3
        ),

        "cm"
    )


    print(

        "Estimated Z =",

        round(
            mean_z / 100,
            3
        ),

        "meters"
    )


    # ========================================================
    # RECTANGLE
    # ========================================================

    if shape == "rectangle":

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


        width = (

            top
            +
            bottom

        ) / 2


        height = (

            left_edge
            +
            right_edge

        ) / 2


        print()
        print("=" * 70)

        print(
            "OBJECT SIZE"
        )

        print("=" * 70)


        print(
            "Top edge =",
            round(
                top,
                3
            ),
            "cm"
        )


        print(
            "Bottom edge =",
            round(
                bottom,
                3
            ),
            "cm"
        )


        print(
            "Left edge =",
            round(
                left_edge,
                3
            ),
            "cm"
        )


        print(
            "Right edge =",
            round(
                right_edge,
                3
            ),
            "cm"
        )


        print()


        print(
            "Estimated WIDTH =",
            round(
                width,
                3
            ),
            "cm"
        )


        print(
            "Estimated HEIGHT =",
            round(
                height,
                3
            ),
            "cm"
        )


        actual_width = input(

            "\nActual width in cm "
            "(Enter to skip): "

        ).strip()


        actual_height = input(

            "Actual height in cm "
            "(Enter to skip): "

        ).strip()


        if actual_width:

            error = percentage_error(

                width,

                float(
                    actual_width
                )
            )


            print(

                "Width error =",

                round(
                    error,
                    2
                ),

                "%"
            )


        if actual_height:

            error = percentage_error(

                height,

                float(
                    actual_height
                )
            )


            print(

                "Height error =",

                round(
                    error,
                    2
                ),

                "%"
            )


    # ========================================================
    # CIRCLE
    # ========================================================

    elif shape == "circle":

        diameter = distance_3d(

            points_3d[0],

            points_3d[1]
        )


        print()
        print("=" * 70)

        print(
            "CIRCLE SIZE"
        )

        print("=" * 70)


        print(

            "Estimated DIAMETER =",

            round(
                diameter,
                3
            ),

            "cm"
        )


        actual = input(

            "Actual diameter in cm "
            "(Enter to skip): "

        ).strip()


        if actual:

            error = percentage_error(

                diameter,

                float(
                    actual
                )
            )


            print(

                "Diameter error =",

                round(
                    error,
                    2
                ),

                "%"
            )


    # ========================================================
    # POLYGON
    # ========================================================

    else:

        print()
        print("=" * 70)

        print(
            "POLYGON EDGE DIMENSIONS"
        )

        print("=" * 70)


        for index in range(
            number_of_points
        ):

            next_index = (

                index
                +
                1

            ) % number_of_points


            edge = distance_3d(

                points_3d[index],

                points_3d[next_index]
            )


            print(

                labels[index],

                "to",

                labels[next_index],

                "=",

                round(
                    edge,
                    3
                ),

                "cm"
            )


# ============================================================
# VIDEO INFORMATION
# ============================================================

def get_video_information(
    video_path
):

    cap = cv2.VideoCapture(
        video_path
    )


    if not cap.isOpened():

        raise RuntimeError(
            "Could not open video."
        )


    frame_count = int(

        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    fps = cap.get(
        cv2.CAP_PROP_FPS
    )


    width = int(

        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )


    height = int(

        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )


    cap.release()


    return (
        frame_count,
        fps,
        width,
        height
    )


# ============================================================
# EXTRACT VIDEO FRAME
# ============================================================

def extract_frame(
    video_path,
    frame_number
):

    cap = cv2.VideoCapture(
        video_path
    )


    cap.set(

        cv2.CAP_PROP_POS_FRAMES,

        frame_number
    )


    success, frame = (
        cap.read()
    )


    cap.release()


    if not success:

        raise RuntimeError(
            "Could not extract frame."
        )


    return frame


# ============================================================
# PROBLEM 3
# ============================================================

def problem_3():

    print()
    print("=" * 70)

    print(
        "PROBLEM 3 - ROOM DIMENSIONS"
    )

    print("=" * 70)


    left_video = input(

        "LEFT room video path: "

    ).strip().strip('"')


    right_video = input(

        "RIGHT room video path: "

    ).strip().strip('"')


    (
        left_frames,
        left_fps,
        left_width,
        left_height

    ) = get_video_information(
        left_video
    )


    (
        right_frames,
        right_fps,
        right_width,
        right_height

    ) = get_video_information(
        right_video
    )


    if (
        left_width
        !=
        right_width

        or

        left_height
        !=
        right_height
    ):

        raise RuntimeError(

            "Left and right videos must "
            "have the same resolution."
        )


    print()
    print(
        "Left video:"
    )

    print(
        left_frames,
        "frames"
    )

    print(
        left_width,
        "x",
        left_height
    )


    print()
    print(
        "Right video:"
    )

    print(
        right_frames,
        "frames"
    )

    print(
        right_width,
        "x",
        right_height
    )


    # --------------------------------------------------------
    # FRAME NUMBERS
    # --------------------------------------------------------

    left_frame_text = input(

        "\nLeft frame number "
        "(Enter = middle): "

    ).strip()


    right_frame_text = input(

        "Right frame number "
        "(Enter = middle): "

    ).strip()


    if left_frame_text:

        left_frame_number = int(
            left_frame_text
        )

    else:

        left_frame_number = (
            left_frames // 2
        )


    if right_frame_text:

        right_frame_number = int(
            right_frame_text
        )

    else:

        right_frame_number = (
            right_frames // 2
        )


    left_image = extract_frame(

        left_video,

        left_frame_number
    )


    right_image = extract_frame(

        right_video,

        right_frame_number
    )


    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    baseline_text = input(

        f"Baseline in cm "
        f"[{DEFAULT_BASELINE_CM}]: "

    ).strip()


    if baseline_text:

        baseline_cm = float(
            baseline_text
        )

    else:

        baseline_cm = (
            DEFAULT_BASELINE_CM
        )


    # --------------------------------------------------------
    # CAMERA CALIBRATION
    # --------------------------------------------------------

    fx, fy, cx, cy = (
        scaled_intrinsics(

            left_width,

            left_height
        )
    )


    # --------------------------------------------------------
    # ROOM CORNERS
    # --------------------------------------------------------

    labels = [

        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H"
    ]


    print()
    print(
        "Select these room corners:"
    )

    print(
        "A = floor front-left"
    )

    print(
        "B = floor front-right"
    )

    print(
        "C = floor back-right"
    )

    print(
        "D = floor back-left"
    )

    print(
        "E = ceiling front-left"
    )

    print(
        "F = ceiling front-right"
    )

    print(
        "G = ceiling back-right"
    )

    print(
        "H = ceiling back-left"
    )


    left_points = select_points(

        left_image,

        8,

        "LEFT ROOM",

        labels,

        False
    )


    right_points = select_points(

        right_image,

        8,

        "RIGHT ROOM",

        labels,

        False
    )


    # --------------------------------------------------------
    # 3D POINTS
    # --------------------------------------------------------

    points_3d = []


    for left_point, right_point in zip(

        left_points,

        right_points
    ):

        result = reconstruct_3d(

            left_point,

            right_point,

            fx,

            fy,

            cx,

            cy,

            baseline_cm
        )


        points_3d.append(
            result
        )


    # --------------------------------------------------------
    # WIDTH
    # --------------------------------------------------------

    width_1 = distance_3d(

        points_3d[0],

        points_3d[1]
    )


    width_2 = distance_3d(

        points_3d[3],

        points_3d[2]
    )


    width_3 = distance_3d(

        points_3d[4],

        points_3d[5]
    )


    width_4 = distance_3d(

        points_3d[7],

        points_3d[6]
    )


    room_width = np.mean(

        [

            width_1,

            width_2,

            width_3,

            width_4

        ]
    )


    # --------------------------------------------------------
    # LENGTH
    # --------------------------------------------------------

    length_1 = distance_3d(

        points_3d[0],

        points_3d[3]
    )


    length_2 = distance_3d(

        points_3d[1],

        points_3d[2]
    )


    length_3 = distance_3d(

        points_3d[4],

        points_3d[7]
    )


    length_4 = distance_3d(

        points_3d[5],

        points_3d[6]
    )


    room_length = np.mean(

        [

            length_1,

            length_2,

            length_3,

            length_4

        ]
    )


    # --------------------------------------------------------
    # HEIGHT
    # --------------------------------------------------------

    height_1 = distance_3d(

        points_3d[0],

        points_3d[4]
    )


    height_2 = distance_3d(

        points_3d[1],

        points_3d[5]
    )


    height_3 = distance_3d(

        points_3d[2],

        points_3d[6]
    )


    height_4 = distance_3d(

        points_3d[3],

        points_3d[7]
    )


    room_height = np.mean(

        [

            height_1,

            height_2,

            height_3,

            height_4

        ]
    )


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 70)

    print(
        "ESTIMATED ROOM DIMENSIONS"
    )

    print("=" * 70)


    print(

        "Width =",

        round(
            room_width,
            2
        ),

        "cm"
    )


    print(

        "Length =",

        round(
            room_length,
            2
        ),

        "cm"
    )


    print(

        "Height =",

        round(
            room_height,
            2
        ),

        "cm"
    )


    print()
    print(
        "In meters:"
    )


    print(

        "Width =",

        round(
            room_width / 100,
            3
        ),

        "m"
    )


    print(

        "Length =",

        round(
            room_length / 100,
            3
        ),

        "m"
    )


    print(

        "Height =",

        round(
            room_height / 100,
            3
        ),

        "m"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "CSc 8830 - ASSIGNMENT 7"
    )

    print("=" * 70)


    print()
    print(
        "1 - Problem 1"
    )

    print(
        "    iPhone calibrated stereo object measurement"
    )


    print()
    print(
        "3 - Problem 3"
    )

    print(
        "    Room measurement from stereo videos"
    )


    print()
    print(
        "Q - Quit"
    )


    choice = input(
        "\nChoose: "
    ).strip().lower()


    try:

        if choice == "1":

            problem_1()


        elif choice == "3":

            problem_3()


        elif choice == "q":

            return


        else:

            print(
                "Invalid selection."
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


    cv2.destroyAllWindows()


if __name__ == "__main__":

    main()