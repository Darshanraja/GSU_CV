import cv2
import numpy as np
import csv
import os


# ============================================================
# CSc 8830 - COMPUTER VISION
# MODULE 2 ASSIGNMENT
#
# STEP 1:
# Camera calibration using smartphone camera
#
# STEP 2:
# Estimate real-world 2D object dimensions using
# perspective projection
#
# STEP 3:
# Validate the Step 2 method using 20 measurements
# and report error statistics
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

# Checkerboard:
# 10 x 7 physical squares
# 9 x 6 inner corners
CHECKERBOARD = (9, 6)

# Each checkerboard square = 2.5 cm
SQUARE_SIZE = 0.025  # meters

# iPhone / smartphone camera index
CAMERA_INDEX = 1

# Calibration settings
MIN_CALIBRATION_CAPTURES = 6
TARGET_CALIBRATION_CAPTURES = 10

CALIBRATION_FILE = "camera_calibration.npz"

# Step 3 validation
VALIDATION_DISTANCE_M = 2.1
NUMBER_OF_VALIDATION_MEASUREMENTS = 20

VALIDATION_CSV = "step3_validation_results.csv"


# ============================================================
# PREDEFINED OBJECTS
# Height and width are in centimeters
# ============================================================

OBJECTS = [
    {"name": "Car",         "height_cm": 3.0,  "width_cm": 5.0},
    {"name": "AirPods",     "height_cm": 2.0,  "width_cm": 5.5},
    {"name": "Bigfoot",     "height_cm": 6.5,  "width_cm": 4.3},
    {"name": "Ruby",        "height_cm": 6.0,  "width_cm": 3.5},
    {"name": "Pepper",      "height_cm": 8.7,  "width_cm": 3.7},
    {"name": "Shoe polish", "height_cm": 4.5,  "width_cm": 10.0},
    {"name": "Container",   "height_cm": 5.5,  "width_cm": 8.0},
    {"name": "Sardines",    "height_cm": 2.4,  "width_cm": 7.0},
    {"name": "Key",         "height_cm": 1.0,  "width_cm": 2.5},
    {"name": "Mouse",       "height_cm": 2.5,  "width_cm": 5.0},
    {"name": "Playdoh",     "height_cm": 3.8,  "width_cm": 4.7},
    {"name": "Bose",        "height_cm": 9.5,  "width_cm": 8.0},
    {"name": "Oil",         "height_cm": 11.5, "width_cm": 6.5},
    {"name": "Juice",       "height_cm": 10.5, "width_cm": 5.5},
    {"name": "Roku",        "height_cm": 1.6,  "width_cm": 4.0},
    {"name": "Lotion",      "height_cm": 14.0, "width_cm": 3.5},
    {"name": "Vitamin",     "height_cm": 10.5, "width_cm": 9.0},
    {"name": "Cup",         "height_cm": 13.0, "width_cm": 7.0},
    {"name": "Brownie",     "height_cm": 1.5,  "width_cm": 9.0},
    {"name": "Egg cup",     "height_cm": 7.5,  "width_cm": 5.0},
]


# ============================================================
# OPEN SMARTPHONE CAMERA
# ============================================================

def open_phone_camera():

    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_AVFOUNDATION
    )

    if not cap.isOpened():

        print("\nERROR: Could not open smartphone camera.")
        print("Check CAMERA_INDEX or Continuity Camera.")

        return None

    return cap


# ============================================================
# STEP 1 - LIVE CAMERA CALIBRATION
# ============================================================

def calibrate_camera():

    print("\n")
    print("=" * 70)
    print("STEP 1 - SMARTPHONE CAMERA CALIBRATION")
    print("=" * 70)


    # --------------------------------------------------------
    # Real-world checkerboard coordinates
    # --------------------------------------------------------

    objp = np.zeros(
        (
            CHECKERBOARD[0]
            * CHECKERBOARD[1],
            3
        ),
        np.float32
    )

    objp[:, :2] = (
        np.mgrid[
            0:CHECKERBOARD[0],
            0:CHECKERBOARD[1]
        ]
        .T
        .reshape(-1, 2)
    )

    objp *= SQUARE_SIZE


    object_points = []
    image_points = []


    # --------------------------------------------------------
    # Open camera
    # --------------------------------------------------------

    print("\nOpening smartphone camera...")

    cap = open_phone_camera()

    if cap is None:
        exit()


    print("\nCamera opened successfully.")
    print("\nInstructions:")
    print("Show the checkerboard to the camera.")
    print("Press SPACE to capture a calibration frame.")
    print("Move and tilt the board between captures.")
    print("Press C when ready to calibrate.")
    print("Press ESC to exit.")


    successful_captures = 0
    image_size = None


    # --------------------------------------------------------
    # Capture calibration frames
    # --------------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:
            print("ERROR: Could not read camera frame.")
            break


        height, width = frame.shape[:2]

        image_size = (
            width,
            height
        )


        display = frame.copy()


        cv2.putText(
            display,
            f"Successful captures: {successful_captures}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        cv2.putText(
            display,
            "SPACE = Capture checkerboard",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        cv2.putText(
            display,
            "C = Calibrate | ESC = Exit",
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        cv2.imshow(
            "STEP 1 - Camera Calibration",
            display
        )


        key = cv2.waitKey(1) & 0xFF


        # ----------------------------------------------------
        # SPACE = capture
        # ----------------------------------------------------

        if key == 32:

            print("\nChecking captured frame...")


            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )


            found, corners = cv2.findChessboardCorners(
                gray,
                CHECKERBOARD,
                flags=(
                    cv2.CALIB_CB_ADAPTIVE_THRESH
                    + cv2.CALIB_CB_NORMALIZE_IMAGE
                )
            )


            # Try stronger detector if normal one fails
            if not found:

                found_sb, corners_sb = (
                    cv2.findChessboardCornersSB(
                        gray,
                        CHECKERBOARD,
                        flags=(
                            cv2.CALIB_CB_NORMALIZE_IMAGE
                            + cv2.CALIB_CB_EXHAUSTIVE
                            + cv2.CALIB_CB_ACCURACY
                        )
                    )
                )

                if found_sb:

                    found = True
                    corners = corners_sb


            if found:

                try:

                    refined = cv2.cornerSubPix(
                        gray,
                        corners.astype(np.float32),
                        (11, 11),
                        (-1, -1),
                        (
                            cv2.TERM_CRITERIA_EPS
                            + cv2.TERM_CRITERIA_MAX_ITER,
                            30,
                            0.001
                        )
                    )

                except cv2.error:

                    refined = corners


                object_points.append(
                    objp.copy()
                )

                image_points.append(
                    refined
                )


                successful_captures += 1


                print(
                    f"[SUCCESS] Capture "
                    f"{successful_captures}"
                )


                preview = frame.copy()


                cv2.drawChessboardCorners(
                    preview,
                    CHECKERBOARD,
                    refined,
                    True
                )


                cv2.imshow(
                    "Detected Checkerboard",
                    preview
                )


                cv2.waitKey(500)

                cv2.destroyWindow(
                    "Detected Checkerboard"
                )


                if (
                    successful_captures
                    == TARGET_CALIBRATION_CAPTURES
                ):

                    print(
                        f"\nYou now have "
                        f"{TARGET_CALIBRATION_CAPTURES} "
                        f"successful captures."
                    )

                    print(
                        "Press C to calibrate."
                    )


            else:

                print(
                    "[FAILED] Checkerboard "
                    "was not detected."
                )


        # ----------------------------------------------------
        # C = calibration
        # ----------------------------------------------------

        elif key == ord("c"):

            if (
                successful_captures
                < MIN_CALIBRATION_CAPTURES
            ):

                print(
                    f"\nNeed at least "
                    f"{MIN_CALIBRATION_CAPTURES} "
                    f"successful captures."
                )

                continue

            break


        # ----------------------------------------------------
        # ESC
        # ----------------------------------------------------

        elif key == 27:

            cap.release()
            cv2.destroyAllWindows()
            exit()


    cap.release()
    cv2.destroyAllWindows()


    # --------------------------------------------------------
    # Calibrate
    # --------------------------------------------------------

    print("\nCalibrating camera...")


    (
        rms_error,
        camera_matrix,
        distortion_coefficients,
        rvecs,
        tvecs
    ) = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None
    )


    # --------------------------------------------------------
    # Reprojection RMSE
    # --------------------------------------------------------

    total_squared_error = 0.0
    total_points = 0


    for i in range(
        len(object_points)
    ):

        projected_points, _ = cv2.projectPoints(
            object_points[i],
            rvecs[i],
            tvecs[i],
            camera_matrix,
            distortion_coefficients
        )


        difference = (
            image_points[i]
            - projected_points
        )


        total_squared_error += np.sum(
            difference ** 2
        )


        total_points += len(
            object_points[i]
        )


    reprojection_rmse = np.sqrt(
        total_squared_error
        / total_points
    )


    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]

    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]


    # --------------------------------------------------------
    # Save calibration
    # --------------------------------------------------------

    np.savez(
        CALIBRATION_FILE,

        camera_matrix=camera_matrix,

        distortion_coefficients=(
            distortion_coefficients
        ),

        rms_error=rms_error,

        reprojection_rmse=(
            reprojection_rmse
        ),

        image_width=image_size[0],

        image_height=image_size[1]
    )


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("STEP 1 RESULTS")
    print("=" * 70)


    print(
        f"\nCalibration images used: "
        f"{successful_captures}"
    )


    print(
        f"Image resolution: "
        f"{image_size[0]} x "
        f"{image_size[1]}"
    )


    print(
        f"\nOpenCV RMS error: "
        f"{rms_error:.6f} pixels"
    )


    print(
        f"Reprojection RMSE: "
        f"{reprojection_rmse:.6f} pixels"
    )


    print("\nCamera Matrix K:")
    print(camera_matrix)


    print("\nDistortion coefficients:")
    print(distortion_coefficients)


    print("\nIntrinsic parameters:")

    print(
        f"fx = {fx:.4f}"
    )

    print(
        f"fy = {fy:.4f}"
    )

    print(
        f"cx = {cx:.4f}"
    )

    print(
        f"cy = {cy:.4f}"
    )


    print(
        f"\nCalibration saved to "
        f"{CALIBRATION_FILE}"
    )


    return (
        camera_matrix,
        distortion_coefficients,
        image_size
    )


# ============================================================
# CAPTURE OBJECT IMAGE
# ============================================================

def capture_object_image(
    distance_m,
    title
):

    cap = open_phone_camera()

    if cap is None:
        return None


    captured_frame = None


    print(
        "\nPress SPACE to capture object."
    )

    print(
        "Press ESC to cancel."
    )


    while True:

        ret, frame = cap.read()

        if not ret:
            print("ERROR: Could not read camera.")
            break


        display = frame.copy()


        cv2.putText(
            display,
            title,
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        cv2.putText(
            display,
            f"Distance: {distance_m:.2f} m",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        cv2.putText(
            display,
            "SPACE = Capture | ESC = Exit",
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        cv2.imshow(
            title,
            display
        )


        key = cv2.waitKey(1) & 0xFF


        if key == 32:

            captured_frame = frame.copy()
            break


        elif key == 27:

            break


    cap.release()
    cv2.destroyAllWindows()


    return captured_frame


# ============================================================
# SELECT OBJECT CORNERS
# ============================================================

def select_object_corners(
    image
):

    points = []

    display = image.copy()


    def mouse_callback(
        event,
        x,
        y,
        flags,
        param
    ):

        if (
            event
            == cv2.EVENT_LBUTTONDOWN
        ):

            if len(points) < 4:

                points.append(
                    (x, y)
                )


                cv2.circle(
                    display,
                    (x, y),
                    7,
                    (0, 255, 0),
                    -1
                )


                cv2.putText(
                    display,
                    str(len(points)),
                    (
                        x + 10,
                        y - 10
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )


                cv2.imshow(
                    "Select Object Corners",
                    display
                )


    print(
        "\nClick corners in this order:"
    )

    print("1. Top-left")
    print("2. Top-right")
    print("3. Bottom-right")
    print("4. Bottom-left")


    cv2.namedWindow(
        "Select Object Corners",
        cv2.WINDOW_NORMAL
    )


    cv2.imshow(
        "Select Object Corners",
        display
    )


    cv2.setMouseCallback(
        "Select Object Corners",
        mouse_callback
    )


    while len(points) < 4:

        key = cv2.waitKey(
            20
        ) & 0xFF

        if key == 27:
            break


    cv2.destroyAllWindows()


    if len(points) != 4:

        return None


    return points


# ============================================================
# STEP 2 MEASUREMENT FUNCTION
# Also reused by Step 3
# ============================================================

def measure_object(
    camera_matrix,
    distortion_coefficients,
    calibration_size,
    distance_m,
    title
):

    frame = capture_object_image(
        distance_m,
        title
    )


    if frame is None:
        return None


    current_height, current_width = (
        frame.shape[:2]
    )


    calibration_width = (
        calibration_size[0]
    )

    calibration_height = (
        calibration_size[1]
    )


    # --------------------------------------------------------
    # Ensure same resolution as calibration
    # --------------------------------------------------------

    if (
        current_width
        != calibration_width
        or current_height
        != calibration_height
    ):

        print(
            "\nERROR: Camera resolution differs "
            "from calibration resolution."
        )

        print(
            f"Calibration: "
            f"{calibration_width} x "
            f"{calibration_height}"
        )

        print(
            f"Current: "
            f"{current_width} x "
            f"{current_height}"
        )

        return None


    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]


    # --------------------------------------------------------
    # Undistort
    # --------------------------------------------------------

    undistorted = cv2.undistort(
        frame,
        camera_matrix,
        distortion_coefficients
    )


    # --------------------------------------------------------
    # Select corners
    # --------------------------------------------------------

    points = select_object_corners(
        undistorted
    )


    if points is None:
        return None


    # --------------------------------------------------------
    # Convert corner coordinates
    # --------------------------------------------------------

    top_left = np.array(
        points[0],
        dtype=float
    )

    top_right = np.array(
        points[1],
        dtype=float
    )

    bottom_right = np.array(
        points[2],
        dtype=float
    )

    bottom_left = np.array(
        points[3],
        dtype=float
    )


    # --------------------------------------------------------
    # Pixel dimensions
    # --------------------------------------------------------

    top_width = np.linalg.norm(
        top_right
        - top_left
    )


    bottom_width = np.linalg.norm(
        bottom_right
        - bottom_left
    )


    left_height = np.linalg.norm(
        bottom_left
        - top_left
    )


    right_height = np.linalg.norm(
        bottom_right
        - top_right
    )


    width_pixels = (
        top_width
        + bottom_width
    ) / 2.0


    height_pixels = (
        left_height
        + right_height
    ) / 2.0


    # --------------------------------------------------------
    # Perspective projection
    #
    # W = (w_pixels * Z) / fx
    #
    # H = (h_pixels * Z) / fy
    # --------------------------------------------------------

    real_width_m = (
        width_pixels
        * distance_m
        / fx
    )


    real_height_m = (
        height_pixels
        * distance_m
        / fy
    )


    real_width_cm = (
        real_width_m
        * 100
    )


    real_height_cm = (
        real_height_m
        * 100
    )


    # --------------------------------------------------------
    # Draw result
    # --------------------------------------------------------

    result_image = (
        undistorted.copy()
    )


    for i in range(4):

        cv2.line(
            result_image,
            points[i],
            points[
                (i + 1) % 4
            ],
            (0, 255, 0),
            3
        )


    cv2.putText(
        result_image,
        f"Width: {real_width_cm:.2f} cm",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )


    cv2.putText(
        result_image,
        f"Height: {real_height_cm:.2f} cm",
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )


    cv2.imshow(
        "Measurement Result",
        result_image
    )


    cv2.waitKey(700)

    cv2.destroyAllWindows()


    return {
        "width_cm":
            real_width_cm,

        "height_cm":
            real_height_cm,

        "width_pixels":
            width_pixels,

        "height_pixels":
            height_pixels
    }


# ============================================================
# STEP 2
# ============================================================

def run_step2(
    camera_matrix,
    distortion_coefficients,
    calibration_size
):

    print("\n")
    print("=" * 70)
    print(
        "STEP 2 - REAL-WORLD 2D OBJECT "
        "DIMENSION MEASUREMENT"
    )
    print("=" * 70)


    distance_m = float(
        input(
            "\nEnter camera-to-object "
            "distance in meters: "
        )
    )


    result = measure_object(
        camera_matrix,
        distortion_coefficients,
        calibration_size,
        distance_m,
        "STEP 2 - Object Measurement"
    )


    if result is None:

        print(
            "\nStep 2 measurement failed."
        )

        exit()


    print("\n")
    print("=" * 70)
    print("STEP 2 RESULT")
    print("=" * 70)


    print(
        f"\nDistance: "
        f"{distance_m:.3f} m"
    )


    print(
        f"Width in image: "
        f"{result['width_pixels']:.2f} px"
    )


    print(
        f"Height in image: "
        f"{result['height_pixels']:.2f} px"
    )


    print(
        f"\nEstimated width: "
        f"{result['width_cm']:.2f} cm"
    )


    print(
        f"Estimated height: "
        f"{result['height_cm']:.2f} cm"
    )


# ============================================================
# STEP 3
# ============================================================

def run_step3(
    camera_matrix,
    distortion_coefficients,
    calibration_size
):

    print("\n")
    print("=" * 70)
    print("STEP 3 - VALIDATION")
    print("=" * 70)


    print(
        f"\nValidation distance: "
        f"{VALIDATION_DISTANCE_M:.2f} m"
    )


    print("\nChoose validation mode:")

    print(
        "1. Predefined objects"
    )

    print(
        "2. Manual entry"
    )


    mode = input(
        "\nEnter 1 or 2: "
    ).strip()


    if mode not in [
        "1",
        "2"
    ]:

        print(
            "Invalid mode."
        )

        return


    results = []


    # ========================================================
    # PREDEFINED MODE
    # ========================================================

    if mode == "1":

        print(
            "\nPREDEFINED MODE SELECTED"
        )


        print(
            f"{len(OBJECTS)} objects "
            "will be measured."
        )


        for measurement_number, obj in enumerate(
            OBJECTS,
            start=1
        ):

            object_name = (
                obj["name"]
            )

            actual_width_cm = (
                obj["width_cm"]
            )

            actual_height_cm = (
                obj["height_cm"]
            )


            print("\n")
            print("-" * 70)

            print(
                f"MEASUREMENT "
                f"{measurement_number}/"
                f"{len(OBJECTS)}"
            )

            print("-" * 70)


            print(
                f"\nObject: "
                f"{object_name}"
            )


            print(
                f"Actual width: "
                f"{actual_width_cm:.2f} cm"
            )


            print(
                f"Actual height: "
                f"{actual_height_cm:.2f} cm"
            )


            input(
                "\nPlace the object at "
                f"{VALIDATION_DISTANCE_M:.2f} m "
                "and press ENTER..."
            )


            estimated = measure_object(
                camera_matrix,
                distortion_coefficients,
                calibration_size,
                VALIDATION_DISTANCE_M,
                (
                    f"STEP 3 - "
                    f"{object_name}"
                )
            )


            if estimated is None:

                print(
                    "Measurement failed."
                )

                break


            append_validation_result(
                results,
                measurement_number,
                object_name,
                actual_width_cm,
                actual_height_cm,
                estimated
            )


    # ========================================================
    # MANUAL MODE
    # ========================================================

    elif mode == "2":

        print(
            "\nMANUAL MODE SELECTED"
        )


        for measurement_number in range(
            1,
            NUMBER_OF_VALIDATION_MEASUREMENTS
            + 1
        ):

            print("\n")
            print("-" * 70)

            print(
                f"MEASUREMENT "
                f"{measurement_number}/"
                f"{NUMBER_OF_VALIDATION_MEASUREMENTS}"
            )

            print("-" * 70)


            object_name = input(
                "\nEnter object name: "
            ).strip()


            actual_width_cm = float(
                input(
                    "Enter actual width in cm: "
                )
            )


            actual_height_cm = float(
                input(
                    "Enter actual height in cm: "
                )
            )


            input(
                "\nPlace the object at "
                f"{VALIDATION_DISTANCE_M:.2f} m "
                "and press ENTER..."
            )


            estimated = measure_object(
                camera_matrix,
                distortion_coefficients,
                calibration_size,
                VALIDATION_DISTANCE_M,
                (
                    f"STEP 3 - "
                    f"{object_name}"
                )
            )


            if estimated is None:

                print(
                    "Measurement failed."
                )

                break


            append_validation_result(
                results,
                measurement_number,
                object_name,
                actual_width_cm,
                actual_height_cm,
                estimated
            )


    # ========================================================
    # CALCULATE STATISTICS
    # ========================================================

    if len(results) == 0:

        print(
            "\nNo validation measurements completed."
        )

        return


    calculate_statistics(
        results
    )


    save_validation_csv(
        results
    )


# ============================================================
# ADD ONE VALIDATION RESULT
# ============================================================

def append_validation_result(
    results,
    measurement_number,
    object_name,
    actual_width_cm,
    actual_height_cm,
    estimated
):

    estimated_width_cm = (
        estimated["width_cm"]
    )

    estimated_height_cm = (
        estimated["height_cm"]
    )


    # --------------------------------------------------------
    # Errors
    # --------------------------------------------------------

    width_error_cm = (
        estimated_width_cm
        - actual_width_cm
    )


    height_error_cm = (
        estimated_height_cm
        - actual_height_cm
    )


    width_absolute_error_cm = abs(
        width_error_cm
    )


    height_absolute_error_cm = abs(
        height_error_cm
    )


    width_percentage_error = (
        width_absolute_error_cm
        / actual_width_cm
        * 100
    )


    height_percentage_error = (
        height_absolute_error_cm
        / actual_height_cm
        * 100
    )


    # --------------------------------------------------------
    # Print measurement result
    # --------------------------------------------------------

    print("\nRESULT")


    print(
        f"\nActual width: "
        f"{actual_width_cm:.2f} cm"
    )


    print(
        f"Estimated width: "
        f"{estimated_width_cm:.2f} cm"
    )


    print(
        f"Width absolute error: "
        f"{width_absolute_error_cm:.2f} cm"
    )


    print(
        f"Width percentage error: "
        f"{width_percentage_error:.2f}%"
    )


    print(
        f"\nActual height: "
        f"{actual_height_cm:.2f} cm"
    )


    print(
        f"Estimated height: "
        f"{estimated_height_cm:.2f} cm"
    )


    print(
        f"Height absolute error: "
        f"{height_absolute_error_cm:.2f} cm"
    )


    print(
        f"Height percentage error: "
        f"{height_percentage_error:.2f}%"
    )


    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    results.append(
        {
            "measurement":
                measurement_number,

            "object_name":
                object_name,

            "distance_m":
                VALIDATION_DISTANCE_M,

            "actual_width_cm":
                actual_width_cm,

            "estimated_width_cm":
                estimated_width_cm,

            "width_error_cm":
                width_error_cm,

            "width_absolute_error_cm":
                width_absolute_error_cm,

            "width_percentage_error":
                width_percentage_error,

            "actual_height_cm":
                actual_height_cm,

            "estimated_height_cm":
                estimated_height_cm,

            "height_error_cm":
                height_error_cm,

            "height_absolute_error_cm":
                height_absolute_error_cm,

            "height_percentage_error":
                height_percentage_error
        }
    )


# ============================================================
# ERROR STATISTICS
# ============================================================

def calculate_statistics(
    results
):

    width_errors = np.array(
        [
            r["width_error_cm"]
            for r in results
        ]
    )


    height_errors = np.array(
        [
            r["height_error_cm"]
            for r in results
        ]
    )


    width_abs_errors = np.abs(
        width_errors
    )


    height_abs_errors = np.abs(
        height_errors
    )


    width_percentage_errors = np.array(
        [
            r["width_percentage_error"]
            for r in results
        ]
    )


    height_percentage_errors = np.array(
        [
            r["height_percentage_error"]
            for r in results
        ]
    )


    # --------------------------------------------------------
    # Width statistics
    # --------------------------------------------------------

    width_mean_error = np.mean(
        width_errors
    )

    width_mae = np.mean(
        width_abs_errors
    )

    width_rmse = np.sqrt(
        np.mean(
            width_errors ** 2
        )
    )

    width_mape = np.mean(
        width_percentage_errors
    )

    width_std = np.std(
        width_errors
    )


    # --------------------------------------------------------
    # Height statistics
    # --------------------------------------------------------

    height_mean_error = np.mean(
        height_errors
    )

    height_mae = np.mean(
        height_abs_errors
    )

    height_rmse = np.sqrt(
        np.mean(
            height_errors ** 2
        )
    )

    height_mape = np.mean(
        height_percentage_errors
    )

    height_std = np.std(
        height_errors
    )


    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL VALIDATION STATISTICS")
    print("=" * 70)


    print(
        f"\nMeasurements completed: "
        f"{len(results)}"
    )


    print("\nWIDTH STATISTICS")


    print(
        f"Mean Error: "
        f"{width_mean_error:.3f} cm"
    )


    print(
        f"Mean Absolute Error (MAE): "
        f"{width_mae:.3f} cm"
    )


    print(
        f"RMSE: "
        f"{width_rmse:.3f} cm"
    )


    print(
        f"MAPE: "
        f"{width_mape:.2f}%"
    )


    print(
        f"Standard Deviation: "
        f"{width_std:.3f} cm"
    )


    print(
        f"Minimum Absolute Error: "
        f"{np.min(width_abs_errors):.3f} cm"
    )


    print(
        f"Maximum Absolute Error: "
        f"{np.max(width_abs_errors):.3f} cm"
    )


    print("\nHEIGHT STATISTICS")


    print(
        f"Mean Error: "
        f"{height_mean_error:.3f} cm"
    )


    print(
        f"Mean Absolute Error (MAE): "
        f"{height_mae:.3f} cm"
    )


    print(
        f"RMSE: "
        f"{height_rmse:.3f} cm"
    )


    print(
        f"MAPE: "
        f"{height_mape:.2f}%"
    )


    print(
        f"Standard Deviation: "
        f"{height_std:.3f} cm"
    )


    print(
        f"Minimum Absolute Error: "
        f"{np.min(height_abs_errors):.3f} cm"
    )


    print(
        f"Maximum Absolute Error: "
        f"{np.max(height_abs_errors):.3f} cm"
    )


# ============================================================
# SAVE STEP 3 CSV
# ============================================================

def save_validation_csv(
    results
):

    fieldnames = [
        "measurement",
        "object_name",
        "distance_m",

        "actual_width_cm",
        "estimated_width_cm",
        "width_error_cm",
        "width_absolute_error_cm",
        "width_percentage_error",

        "actual_height_cm",
        "estimated_height_cm",
        "height_error_cm",
        "height_absolute_error_cm",
        "height_percentage_error"
    ]


    with open(
        VALIDATION_CSV,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            results
        )


    print(
        f"\nValidation results saved to:"
    )

    print(
        VALIDATION_CSV
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("CSc 8830 - COMPUTER VISION")
    print("MODULE 2 ASSIGNMENT")
    print("=" * 70)


    # ========================================================
    # STEP 1
    # ========================================================

    (
        camera_matrix,
        distortion_coefficients,
        calibration_size
    ) = calibrate_camera()


    input(
        "\nSTEP 1 COMPLETE."
        "\nPress ENTER to continue to STEP 2..."
    )


    # ========================================================
    # STEP 2
    # ========================================================

    run_step2(
        camera_matrix,
        distortion_coefficients,
        calibration_size
    )


    input(
        "\nSTEP 2 COMPLETE."
        "\nPress ENTER to continue to STEP 3..."
    )


    # ========================================================
    # STEP 3
    # ========================================================

    run_step3(
        camera_matrix,
        distortion_coefficients,
        calibration_size
    )


    print("\n")
    print("=" * 70)
    print(
        "MODULE 2 PRACTICAL ASSIGNMENT COMPLETE"
    )
    print("=" * 70)


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()