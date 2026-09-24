"""
CSc 8830 - Computer Vision
Module 5 - Problem 2

Planar Structure from Motion / Multi-View Reconstruction

Use four viewpoints of the SAME planar object.

Modes:
1. Camera mode
   - Capture 4 images one after another

2. Upload mode
   - Select 4 images at the same time

For EVERY image click the same four points in this order:

1. Top Left
2. Top Right
3. Bottom Right
4. Bottom Left

Controls during point selection:

LEFT CLICK = select point
R          = reset points
ENTER      = confirm after 4 points
ESC        = cancel

Outputs:
- 4 original images
- 4 images with selected points
- 4 rectified views
- reconstructed object
- reconstructed boundary
- point coordinates CSV
- homography matrices
"""

import cv2
import numpy as np
import csv
from tkinter import Tk, filedialog


# ============================================================
# SETTINGS
# ============================================================

CAMERA_INDEX = 1

NUMBER_VIEWS = 4

REFERENCE_WIDTH = 600
REFERENCE_HEIGHT = 400


# ============================================================
# DISPLAY RESIZE
# ============================================================

def resize_for_display(
    image,
    max_width=1100,
    max_height=750
):

    height, width = image.shape[:2]

    scale = min(
        1.0,
        max_width / width,
        max_height / height
    )

    if scale == 1.0:

        return image.copy(), 1.0

    resized = cv2.resize(
        image,
        (
            int(width * scale),
            int(height * scale)
        ),
        interpolation=cv2.INTER_AREA
    )

    return resized, scale


# ============================================================
# CAMERA MODE
# ============================================================

def capture_four_images():

    images = []

    print()
    print("=======================================")
    print("CAMERA MODE")
    print("=======================================")
    print()
    print("You will capture 4 different viewpoints.")
    print()
    print("SPACE = capture")
    print("ESC   = cancel")
    print()

    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_AVFOUNDATION
    )

    if not cap.isOpened():

        print(
            "Could not open camera."
        )

        print(
            "If needed, try CAMERA_INDEX = 0."
        )

        return None


    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1920
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        1080
    )


    view_number = 1


    while view_number <= NUMBER_VIEWS:

        print()
        print(
            f"Prepare View {view_number}"
        )

        print(
            "Press SPACE when ready."
        )


        captured = False


        while True:

            ret, frame = cap.read()


            if not ret:

                print(
                    "Could not read camera frame."
                )

                cap.release()

                cv2.destroyAllWindows()

                return None


            display = frame.copy()


            cv2.putText(
                display,
                f"VIEW {view_number} of 4",
                (
                    30,
                    50
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (
                    0,
                    255,
                    0
                ),
                2
            )


            cv2.putText(
                display,
                "SPACE = Capture   ESC = Cancel",
                (
                    30,
                    100
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (
                    0,
                    255,
                    0
                ),
                2
            )


            cv2.imshow(
                "Camera - Capture Four Views",
                display
            )


            key = cv2.waitKey(
                1
            ) & 0xFF


            # SPACE
            if key == 32:

                images.append(
                    frame.copy()
                )


                print(
                    f"View {view_number} captured."
                )


                captured = True

                break


            # ESC
            elif key == 27:

                cap.release()

                cv2.destroyAllWindows()

                return None


        if captured:

            view_number += 1


    cap.release()

    cv2.destroyAllWindows()


    print()
    print(
        "All 4 camera views captured successfully."
    )


    return images


# ============================================================
# UPLOAD MODE
# ============================================================

def upload_four_images():

    print()
    print("=======================================")
    print("UPLOAD MODE")
    print("=======================================")
    print()
    print(
        "Select exactly FOUR images at the same time."
    )
    print()

    root = Tk()

    root.withdraw()

    root.attributes(
        "-topmost",
        True
    )


    paths = filedialog.askopenfilenames(

        title="Select Exactly 4 View Images",

        filetypes=[
            (
                "Image Files",
                "*.jpg *.jpeg *.png *.bmp *.JPG *.JPEG *.PNG"
            ),
            (
                "All Files",
                "*.*"
            )
        ]

    )


    root.destroy()


    if not paths:

        print(
            "No images selected."
        )

        return None


    if len(paths) != NUMBER_VIEWS:

        print()
        print(
            "ERROR:"
        )

        print(
            f"You selected {len(paths)} images."
        )

        print(
            "Please select exactly 4 images."
        )

        return None


    images = []


    for index, path in enumerate(
        paths,
        start=1
    ):

        image = cv2.imread(
            path
        )


        if image is None:

            print(
                "Could not read:",
                path
            )

            return None


        images.append(
            image
        )


        print(
            f"View {index}: {path}"
        )


    print()
    print(
        "All 4 images loaded successfully."
    )


    return images


# ============================================================
# CHOOSE INPUT MODE
# ============================================================

def get_four_views():

    print()
    print("=======================================")
    print("STRUCTURE FROM MOTION")
    print("=======================================")

    print()
    print(
        "Choose input method:"
    )

    print()
    print(
        "C = Capture 4 images using camera"
    )

    print(
        "U = Upload 4 images"
    )


    while True:

        choice = input(
            "\nChoose C or U: "
        ).strip().lower()


        if choice == "c":

            return capture_four_images()


        elif choice == "u":

            return upload_four_images()


        else:

            print(
                "Please enter C or U."
            )


# ============================================================
# SELECT FOUR CORNERS
# ============================================================

def select_four_points(
    image,
    view_number
):

    display, scale = resize_for_display(
        image
    )


    points = []


    corner_names = [

        "Top Left",

        "Top Right",

        "Bottom Right",

        "Bottom Left"

    ]


    window_name = (
        f"View {view_number} - Select Four Corners"
    )


    # --------------------------------------------------------
    # Mouse Callback
    # --------------------------------------------------------

    def mouse_callback(
        event,
        x,
        y,
        flags,
        param
    ):

        if event != cv2.EVENT_LBUTTONDOWN:

            return


        if len(points) >= 4:

            return


        original_x = (
            x / scale
        )

        original_y = (
            y / scale
        )


        points.append(
            (
                original_x,
                original_y
            )
        )


        index = (
            len(points) - 1
        )


        print(
            f"View {view_number} - "
            f"{corner_names[index]}:"
        )

        print(
            f"x = {original_x:.2f}, "
            f"y = {original_y:.2f}"
        )


    # --------------------------------------------------------
    # Window
    # --------------------------------------------------------

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_AUTOSIZE
    )


    cv2.setMouseCallback(
        window_name,
        mouse_callback
    )


    print()
    print(
        "======================================="
    )

    print(
        f"VIEW {view_number}"
    )

    print(
        "======================================="
    )


    print()
    print(
        "Click corners in this EXACT order:"
    )

    print(
        "1. Top Left"
    )

    print(
        "2. Top Right"
    )

    print(
        "3. Bottom Right"
    )

    print(
        "4. Bottom Left"
    )


    print()
    print(
        "After selecting all four:"
    )

    print(
        "ENTER = confirm"
    )

    print(
        "R = reset"
    )

    print(
        "ESC = cancel"
    )


    # --------------------------------------------------------
    # Display Loop
    # --------------------------------------------------------

    while True:

        frame = display.copy()


        # Draw selected points
        for index, point in enumerate(
            points
        ):

            px = int(
                point[0] * scale
            )

            py = int(
                point[1] * scale
            )


            cv2.circle(
                frame,
                (
                    px,
                    py
                ),
                8,
                (
                    0,
                    0,
                    255
                ),
                -1
            )


            cv2.putText(
                frame,
                str(
                    index + 1
                ),
                (
                    px + 10,
                    py - 10
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (
                    0,
                    255,
                    0
                ),
                2
            )


        # Draw lines
        if len(points) >= 2:

            for i in range(
                len(points) - 1
            ):

                p1 = (
                    int(
                        points[i][0]
                        *
                        scale
                    ),
                    int(
                        points[i][1]
                        *
                        scale
                    )
                )


                p2 = (
                    int(
                        points[i + 1][0]
                        *
                        scale
                    ),
                    int(
                        points[i + 1][1]
                        *
                        scale
                    )
                )


                cv2.line(
                    frame,
                    p1,
                    p2,
                    (
                        0,
                        255,
                        0
                    ),
                    2
                )


        # Close polygon when all 4 are selected
        if len(points) == 4:

            p1 = (
                int(
                    points[3][0]
                    *
                    scale
                ),
                int(
                    points[3][1]
                    *
                    scale
                )
            )


            p2 = (
                int(
                    points[0][0]
                    *
                    scale
                ),
                int(
                    points[0][1]
                    *
                    scale
                )
            )


            cv2.line(
                frame,
                p1,
                p2,
                (
                    0,
                    255,
                    0
                ),
                2
            )


        # Instructions
        cv2.putText(
            frame,
            f"View {view_number}/4",
            (
                20,
                35
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (
                255,
                255,
                0
            ),
            2
        )


        cv2.putText(
            frame,
            f"Selected: {len(points)}/4",
            (
                20,
                70
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (
                255,
                255,
                0
            ),
            2
        )


        if len(points) < 4:

            next_corner = corner_names[
                len(points)
            ]


            cv2.putText(
                frame,
                f"Next: {next_corner}",
                (
                    20,
                    105
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


        else:

            cv2.putText(
                frame,
                "Press ENTER to confirm",
                (
                    20,
                    105
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (
                    0,
                    255,
                    0
                ),
                2
            )


        cv2.imshow(
            window_name,
            frame
        )


        key = cv2.waitKey(
            20
        ) & 0xFF


        # ENTER
        if key in (
            13,
            10
        ):

            if len(points) == 4:

                break


        # R
        elif key in (
            ord("r"),
            ord("R")
        ):

            points.clear()

            print()
            print(
                f"View {view_number} points reset."
            )


        # ESC
        elif key == 27:

            points.clear()

            cv2.destroyWindow(
                window_name
            )

            return None


    cv2.destroyWindow(
        window_name
    )


    # Important on macOS:
    # process OpenCV window events before
    # opening the next image.
    for _ in range(5):

        cv2.waitKey(
            1
        )


    return np.array(
        points,
        dtype=np.float32
    )


# ============================================================
# DRAW POINTS ON ORIGINAL IMAGE
# ============================================================

def draw_selected_boundary(
    image,
    points,
    view_number
):

    output = image.copy()


    pts = points.astype(
        np.int32
    )


    cv2.polylines(
        output,
        [
            pts
        ],
        True,
        (
            0,
            255,
            0
        ),
        4
    )


    for i, point in enumerate(
        pts
    ):

        x = int(
            point[0]
        )

        y = int(
            point[1]
        )


        cv2.circle(
            output,
            (
                x,
                y
            ),
            9,
            (
                0,
                0,
                255
            ),
            -1
        )


        cv2.putText(
            output,
            str(
                i + 1
            ),
            (
                x + 12,
                y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (
                255,
                0,
                0
            ),
            2
        )


    cv2.imwrite(
        f"view_{view_number}_points.png",
        output
    )


# ============================================================
# SAVE POINT COORDINATES
# ============================================================

def save_coordinates(
    all_points
):

    with open(
        "view_point_coordinates.csv",
        "w",
        newline=""
    ) as file:

        writer = csv.writer(
            file
        )


        writer.writerow(
            [
                "View",
                "Point",
                "Corner",
                "x",
                "y"
            ]
        )


        names = [

            "Top Left",

            "Top Right",

            "Bottom Right",

            "Bottom Left"

        ]


        for view_index, points in enumerate(
            all_points,
            start=1
        ):

            for point_index, point in enumerate(
                points,
                start=1
            ):

                writer.writerow(
                    [
                        view_index,

                        point_index,

                        names[
                            point_index - 1
                        ],

                        float(
                            point[0]
                        ),

                        float(
                            point[1]
                        )
                    ]
                )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Step 1:
    # Get all four images FIRST
    # --------------------------------------------------------

    images = get_four_views()


    if images is None:

        print(
            "No images obtained."
        )

        return


    if len(images) != 4:

        print(
            "Exactly four images are required."
        )

        return


    # --------------------------------------------------------
    # Save originals
    # --------------------------------------------------------

    for i, image in enumerate(
        images,
        start=1
    ):

        cv2.imwrite(
            f"view_{i}_original.png",
            image
        )


    print()
    print(
        "======================================="
    )

    print(
        "ALL 4 IMAGES READY"
    )

    print(
        "======================================="
    )

    print()
    print(
        "Now select the same four book corners"
    )

    print(
        "in all four images."
    )


    # --------------------------------------------------------
    # Step 2:
    # Select points
    # --------------------------------------------------------

    all_points = []


    for view_number in range(
        1,
        NUMBER_VIEWS + 1
    ):

        points = select_four_points(
            images[
                view_number - 1
            ],
            view_number
        )


        if points is None:

            print(
                "Point selection cancelled."
            )

            cv2.destroyAllWindows()

            return


        all_points.append(
            points
        )


        draw_selected_boundary(
            images[
                view_number - 1
            ],
            points,
            view_number
        )


        print()
        print(
            f"View {view_number} completed."
        )


    # --------------------------------------------------------
    # Save coordinates
    # --------------------------------------------------------

    save_coordinates(
        all_points
    )


    # --------------------------------------------------------
    # Reference plane
    # --------------------------------------------------------

    reference_points = np.array(
        [

            [
                0,
                0
            ],

            [
                REFERENCE_WIDTH - 1,
                0
            ],

            [
                REFERENCE_WIDTH - 1,
                REFERENCE_HEIGHT - 1
            ],

            [
                0,
                REFERENCE_HEIGHT - 1
            ]

        ],
        dtype=np.float32
    )


    homographies = []

    warped_images = []


    # --------------------------------------------------------
    # Homography for each view
    # --------------------------------------------------------

    for i in range(
        NUMBER_VIEWS
    ):

        H = cv2.getPerspectiveTransform(
            all_points[i],
            reference_points
        )


        homographies.append(
            H
        )


        warped = cv2.warpPerspective(
            images[i],
            H,
            (
                REFERENCE_WIDTH,
                REFERENCE_HEIGHT
            )
        )


        warped_images.append(
            warped
        )


        cv2.imwrite(
            f"view_{i + 1}_rectified.png",
            warped
        )


        print()
        print(
            f"======================================="
        )

        print(
            f"HOMOGRAPHY MATRIX - VIEW {i + 1}"
        )

        print(
            f"======================================="
        )

        print(
            H
        )


    # --------------------------------------------------------
    # Save homographies
    # --------------------------------------------------------

    with open(
        "homography_matrices.txt",
        "w"
    ) as file:

        for i, H in enumerate(
            homographies,
            start=1
        ):

            file.write(
                f"View {i}\n"
            )

            file.write(
                str(
                    H
                )
            )

            file.write(
                "\n\n"
            )


    # --------------------------------------------------------
    # Average the four rectified views
    # --------------------------------------------------------

    stack = np.stack(
        warped_images,
        axis=0
    ).astype(
        np.float32
    )


    reconstructed = np.mean(
        stack,
        axis=0
    ).astype(
        np.uint8
    )


    cv2.imwrite(
        "reconstructed_object.png",
        reconstructed
    )


    # --------------------------------------------------------
    # Draw reconstructed boundary
    # --------------------------------------------------------

    reconstructed_boundary = (
        reconstructed.copy()
    )


    boundary = np.array(
        [
            [
                0,
                0
            ],

            [
                REFERENCE_WIDTH - 1,
                0
            ],

            [
                REFERENCE_WIDTH - 1,
                REFERENCE_HEIGHT - 1
            ],

            [
                0,
                REFERENCE_HEIGHT - 1
            ]
        ],
        dtype=np.int32
    )


    cv2.polylines(
        reconstructed_boundary,
        [
            boundary
        ],
        True,
        (
            0,
            255,
            0
        ),
        5
    )


    cv2.imwrite(
        "reconstructed_boundary.png",
        reconstructed_boundary
    )


    # --------------------------------------------------------
    # Final results
    # --------------------------------------------------------

    print()
    print(
        "======================================="
    )

    print(
        "STRUCTURE FROM MOTION COMPLETE"
    )

    print(
        "======================================="
    )


    print()
    print(
        "Saved:"
    )


    print(
        "view_1_original.png"
    )

    print(
        "view_2_original.png"
    )

    print(
        "view_3_original.png"
    )

    print(
        "view_4_original.png"
    )


    print()
    print(
        "view_1_points.png"
    )

    print(
        "view_2_points.png"
    )

    print(
        "view_3_points.png"
    )

    print(
        "view_4_points.png"
    )


    print()
    print(
        "view_1_rectified.png"
    )

    print(
        "view_2_rectified.png"
    )

    print(
        "view_3_rectified.png"
    )

    print(
        "view_4_rectified.png"
    )


    print()
    print(
        "view_point_coordinates.csv"
    )

    print(
        "homography_matrices.txt"
    )

    print(
        "reconstructed_object.png"
    )

    print(
        "reconstructed_boundary.png"
    )


    # --------------------------------------------------------
    # Display reconstruction
    # --------------------------------------------------------

    display, scale = resize_for_display(
        reconstructed_boundary
    )


    cv2.imshow(
        "Final Reconstructed Object Boundary",
        display
    )


    print()
    print(
        "Press any key to close."
    )


    cv2.waitKey(
        0
    )

    cv2.destroyAllWindows()


if __name__ == "__main__":

    main()