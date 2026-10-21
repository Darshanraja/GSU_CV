import cv2
from pathlib import Path


# ============================================================
# SETTINGS
# ============================================================

# Change this if your iPhone uses a different camera index
IPHONE_CAMERA_INDEX = 1

ASSETS_DIR = Path("assets")
ASSETS_DIR.mkdir(exist_ok=True)


# Files will be captured in this exact order
CAPTURE_LIST = [
    ("left_speaker.jpg", "SPEAKER - LEFT"),
    ("right_speaker.jpg", "SPEAKER - RIGHT"),
    ("left_circle.jpg", "CIRCLE - LEFT"),
    ("right_circle.jpg", "CIRCLE - RIGHT"),
    ("left_polygon.jpg", "POLYGON - LEFT"),
    ("right_polygon.jpg", "POLYGON - RIGHT"),
]


# ============================================================
# OPEN IPHONE CAMERA
# ============================================================

cap = cv2.VideoCapture(
    IPHONE_CAMERA_INDEX,
    cv2.CAP_AVFOUNDATION
)


if not cap.isOpened():
    raise RuntimeError(
        "Could not open iPhone camera.\n"
        "Try changing IPHONE_CAMERA_INDEX to 0, 2, or 3."
    )


# Request 1920 x 1080
cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1920
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    1080
)


window_name = "Assignment 7 - iPhone Capture"

cv2.namedWindow(
    window_name,
    cv2.WINDOW_NORMAL
)


# ============================================================
# CAPTURE LOOP
# ============================================================

current_index = 0

print()
print("=" * 70)
print("ASSIGNMENT 7 IMAGE CAPTURE")
print("=" * 70)

print()
print("SPACE = capture current image")
print("R     = retake previous image")
print("ESC   = quit")
print()


while current_index < len(CAPTURE_LIST):

    success, frame = cap.read()

    if not success:
        continue


    filename, label = (
        CAPTURE_LIST[current_index]
    )


    display = frame.copy()


    # --------------------------------------------------------
    # TOP BANNER
    # --------------------------------------------------------

    cv2.rectangle(
        display,
        (0, 0),
        (
            display.shape[1],
            120
        ),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        display,
        f"Capture {current_index + 1} of {len(CAPTURE_LIST)}",
        (25, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )


    cv2.putText(
        display,
        label,
        (25, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )


    cv2.putText(
        display,
        "SPACE = capture | R = previous | ESC = quit",
        (25, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


    cv2.imshow(
        window_name,
        display
    )


    key = (
        cv2.waitKey(1)
        &
        0xFF
    )


    # ========================================================
    # CAPTURE
    # ========================================================

    if key == 32:  # SPACE

        output_path = (
            ASSETS_DIR
            /
            filename
        )


        cv2.imwrite(
            str(output_path),
            frame
        )


        print(
            f"Saved: {output_path}"
        )


        current_index += 1


        if current_index < len(CAPTURE_LIST):

            next_filename, next_label = (
                CAPTURE_LIST[current_index]
            )

            print()
            print(
                f"Next: {next_label}"
            )


    # ========================================================
    # GO BACK / RETAKE
    # ========================================================

    elif key in [
        ord("r"),
        ord("R")
    ]:

        if current_index > 0:

            current_index -= 1

            filename, label = (
                CAPTURE_LIST[current_index]
            )

            print()
            print(
                f"Retake: {label}"
            )


    # ========================================================
    # ESC
    # ========================================================

    elif key == 27:

        print(
            "Capture cancelled."
        )

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()


if current_index == len(CAPTURE_LIST):

    print()
    print("=" * 70)
    print("ALL 6 IMAGES CAPTURED")
    print("=" * 70)

    print()
    print("Saved in assets/:")
    print()

    for filename, label in CAPTURE_LIST:

        print(
            ASSETS_DIR
            /
            filename
        )