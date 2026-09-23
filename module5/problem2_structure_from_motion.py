"""
CSc 8830 - Computer Vision
Module 5 - Problem 2

Planar Structure from Motion / Multi-View Reconstruction

Four viewpoints of the same planar object.

For each view:
C = Capture from camera
U = Upload image

Then click the same four corners:

1. Top Left
2. Top Right
3. Bottom Right
4. Bottom Left

The script:
- collects four views
- records pixel coordinates
- estimates homographies
- maps all views to a common reference
- reconstructs the planar object boundary
- saves camera/view information
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
# CAMERA CAPTURE
# ============================================================

def capture_camera(
    view_number
):

    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_AVFOUNDATION
    )


    if not cap.isOpened():

        print(
            "Could not open camera."
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


    print()
    print(
        f"View {view_number}"
    )

    print(
        "SPACE = capture"
    )

    print(
        "ESC = cancel"
    )


    captured = None


    while True:

        ret, frame = cap.read()


        if not ret:

            break


        display = frame.copy()


        cv2.putText(
            display,
            f"View {view_number} - SPACE to Capture",
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


        cv2.imshow(
            "Camera",
            display
        )


        key = (
            cv2.waitKey(1)
            &
            0xFF
        )


        if key == 32:

            captured = frame.copy()

            break


        elif key == 27:

            break


    cap.release()

    cv2.destroyAllWindows()


    return captured


# ============================================================
# UPLOAD IMAGE
# ============================================================

def upload_image():

    root = Tk()

    root.withdraw()


    path = filedialog.askopenfilename(
        title="Select View Image",
        filetypes=[
            (
                "Images",
                "*.jpg *.jpeg *.png *.bmp"
            )
        ]
    )


    root.destroy()


    if not path:

        return None


    image = cv2.imread(
        path
    )


    return image


# ============================================================
# GET IMAGE FOR EACH VIEW
# ============================================================

def get_view(
    view_number
):

    while True:

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

        print(
            "C = Camera"
        )

        print(
            "U = Upload Photo"
        )


        option = input(
            "Choose C or U: "
        ).strip().lower()


        if option == "c":

            return capture_camera(
                view_number
            )


        elif option == "u":

            return upload_image()


        else:

            print(
                "Invalid option."
            )


# ============================================================
# RESIZE DISPLAY
# ============================================================

def resize_for_display(
    image,
    max_width=1200,
    max_height=800
):

    height, width = image.shape[:2]


    scale = min(

        1.0,

        max_width / width,

        max_height / height

    )


    resized = cv2.resize(
        image,
        (
            int(
                width * scale
            ),
            int(
                height * scale
            )
        ),
        interpolation=cv2.INTER_AREA
    )


    return (
        resized,
        scale
    )


# ============================================================
# SELECT FOUR CORNERS
# ============================================================

def select_points(
    image,
    view_number
):

    display, scale = resize_for_display(
        image
    )


    points = []


    names = [

        "Top Left",

        "Top Right",

        "Bottom Right",

        "Bottom Left"

    ]


    window = (
        f"View {view_number} - Select Corners"
    )


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

            if len(
                points
            ) < 4:

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


                print(
                    names[
                        len(points)
                        -
                        1
                    ],
                    "=",
                    (
                        original_x,
                        original_y
                    )
                )


    cv2.namedWindow(
        window,
        cv2.WINDOW_AUTOSIZE
    )


    cv2.setMouseCallback(
        window,
        mouse_callback
    )


    print()
    print(
        f"VIEW {view_number}"
    )

    print(
        "Click corners in this order:"
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


    while True:

        frame = display.copy()


        for i, point in enumerate(
            points
        ):

            px = int(
                point[0]
                *
                scale
            )

            py = int(
                point[1]
                *
                scale
            )


            cv2.circle(
                frame,
                (
                    px,
                    py
                ),
                7,
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
                    i + 1
                ),
                (
                    px + 10,
                    py
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


        if len(
            points
        ) >= 2:

            for i in range(
                len(points)
                -
                1
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


        cv2.imshow(
            window,
            frame
        )


        key = (
            cv2.waitKey(20)
            &
            0xFF
        )


        if (
            len(points)
            ==
            4
            and
            key in (
                13,
                10
            )
        ):

            break


        if key in (
            ord("r"),
            ord("R")
        ):

            points.clear()


        elif key == 27:

            points.clear()

            break


    cv2.destroyWindow(
        window
    )


    if len(
        points
    ) != 4:

        return None


    return np.array(
        points,
        dtype=np.float32
    )


# ============================================================
# DRAW SELECTED BOUNDARY
# ============================================================

def draw_boundary(
    image,
    points,
    view_number
):

    output = image.copy()


    integer_points = points.astype(
        np.int32
    )


    cv2.polylines(
        output,
        [
            integer_points
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
        integer_points
    ):

        cv2.circle(
            output,
            tuple(point),
            8,
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
                int(
                    point[0]
                )
                +
                10,

                int(
                    point[1]
                )
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
# MAIN
# ============================================================

def main():

    images = []

    all_points = []


    # ========================================================
    # ACQUIRE FOUR VIEWS
    # ========================================================

    for view in range(
        1,
        NUMBER_VIEWS + 1
    ):

        image = get_view(
            view
        )


        if image is None:

            print(
                "Could not obtain view."
            )

            return


        cv2.imwrite(
            f"view_{view}_original.png",
            image
        )


        points = select_points(
            image,
            view
        )


        if points is None:

            print(
                "Point selection cancelled."
            )

            return


        draw_boundary(
            image,
            points,
            view
        )


        images.append(
            image
        )


        all_points.append(
            points
        )


    # ========================================================
    # REFERENCE PLANE
    # ========================================================

    reference_points = np.array(
        [

            [
                0,
                0
            ],

            [
                REFERENCE_WIDTH,
                0
            ],

            [
                REFERENCE_WIDTH,
                REFERENCE_HEIGHT
            ],

            [
                0,
                REFERENCE_HEIGHT
            ]

        ],
        dtype=np.float32
    )


    homographies = []

    warped_images = []


    # ========================================================
    # HOMOGRAPHY FROM EACH VIEW
    # ========================================================

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
            f"Homography View {i + 1}:"
        )

        print(
            H
        )


    # ========================================================
    # COMBINE FOUR RECTIFIED VIEWS
    # ========================================================

    stack = np.stack(
        warped_images,
        axis=0
    ).astype(
        np.float32
    )


    average_view = np.mean(
        stack,
        axis=0
    ).astype(
        np.uint8
    )


    cv2.imwrite(
        "reconstructed_object.png",
        average_view
    )


    # ========================================================
    # DRAW RECONSTRUCTED BOUNDARY
    # ========================================================

    reconstruction = average_view.copy()


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
        reconstruction,
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
        reconstruction
    )


    # ========================================================
    # SAVE PIXEL COORDINATES
    # ========================================================

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
                "x",
                "y"
            ]
        )


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

                        float(
                            point[0]
                        ),

                        float(
                            point[1]
                        )
                    ]
                )


    # ========================================================
    # SAVE HOMOGRAPHIES
    # ========================================================

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


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "Reconstructed Object",
        reconstruction
    )


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


    print(
        "Saved four original images."
    )

    print(
        "Saved four images with selected points."
    )

    print(
        "Saved four rectified views."
    )

    print(
        "Saved reconstructed_object.png"
    )

    print(
        "Saved reconstructed_boundary.png"
    )

    print(
        "Saved view_point_coordinates.csv"
    )

    print(
        "Saved homography_matrices.txt"
    )


    cv2.waitKey(
        0
    )

    cv2.destroyAllWindows()


if __name__ == "__main__":

    main()