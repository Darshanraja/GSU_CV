"""
CSc 8830 - Computer Vision
Module 5 - Problem 1

Optical Flow Using Uploaded Video

Method:
- Upload video
- Process first 30 seconds
- Shi-Tomasi feature detection
- Lucas-Kanade optical flow
- Draw motion trails
- Save optical-flow video
- Save two consecutive validation frames
- Save actual tracked pixel locations

Controls while processing:
ESC = stop
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

VALIDATION_FRAME_1 = "validation_frame_1.png"
VALIDATION_FRAME_2 = "validation_frame_2.png"

VALIDATION_CSV = "tracking_validation.csv"


# ============================================================
# SELECT VIDEO
# ============================================================

def select_video():

    root = Tk()
    root.withdraw()

    path = filedialog.askopenfilename(
        title="Select Video",
        filetypes=[
            ("Video Files", "*.mp4 *.mov *.avi *.mkv"),
            ("All Files", "*.*")
        ]
    )

    root.destroy()

    return path


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

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    duration = (
        total_frames / fps
        if fps > 0
        else 0
    )


    print()
    print(
        "======================================="
    )

    print(
        "VIDEO INFORMATION"
    )

    print(
        "======================================="
    )

    print(
        "FPS:",
        fps
    )

    print(
        "Resolution:",
        width,
        "x",
        height
    )

    print(
        "Duration:",
        duration,
        "seconds"
    )


    if duration < 30:

        print()
        print(
            "WARNING: Assignment asks for at least 30 seconds."
        )


    max_frames = int(
        fps * PROCESS_SECONDS
    )


    # ========================================================
    # OUTPUT VIDEO
    # ========================================================

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )


    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        fourcc,
        fps,
        (
            width,
            height
        )
    )


    # ========================================================
    # READ FIRST FRAME
    # ========================================================

    ret, old_frame = cap.read()


    if not ret:

        print(
            "Could not read first frame."
        )

        return


    old_gray = cv2.cvtColor(
        old_frame,
        cv2.COLOR_BGR2GRAY
    )


    # ========================================================
    # SHI-TOMASI CORNERS
    # ========================================================

    feature_params = dict(

        maxCorners=200,

        qualityLevel=0.3,

        minDistance=7,

        blockSize=7

    )


    old_points = cv2.goodFeaturesToTrack(
        old_gray,
        mask=None,
        **feature_params
    )


    # ========================================================
    # LUCAS-KANADE PARAMETERS
    # ========================================================

    lk_params = dict(

        winSize=(
            21,
            21
        ),

        maxLevel=3,

        criteria=(
            cv2.TERM_CRITERIA_EPS
            |
            cv2.TERM_CRITERIA_COUNT,

            30,

            0.01
        )

    )


    trail_mask = np.zeros_like(
        old_frame
    )


    frame_number = 1


    validation_done = False


    # Pick validation roughly 5 seconds into video
    validation_target = int(
        fps * 5
    )


    validation_rows = []


    # ========================================================
    # PROCESS VIDEO
    # ========================================================

    while True:

        ret, frame = cap.read()


        if not ret:

            break


        if frame_number >= max_frames:

            break


        frame_gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        if (
            old_points is None
            or len(old_points) < 10
        ):

            old_points = cv2.goodFeaturesToTrack(
                old_gray,
                mask=None,
                **feature_params
            )

            trail_mask = np.zeros_like(
                frame
            )


        new_points, status, error = (
            cv2.calcOpticalFlowPyrLK(
                old_gray,
                frame_gray,
                old_points,
                None,
                **lk_params
            )
        )


        if new_points is None:

            old_gray = frame_gray.copy()

            old_points = cv2.goodFeaturesToTrack(
                old_gray,
                mask=None,
                **feature_params
            )

            continue


        good_new = new_points[
            status.flatten() == 1
        ]


        good_old = old_points[
            status.flatten() == 1
        ]


        # ====================================================
        # DRAW MOTION
        # ====================================================

        for new, old in zip(
            good_new,
            good_old
        ):

            x_new, y_new = new.ravel()

            x_old, y_old = old.ravel()


            cv2.line(
                trail_mask,
                (
                    int(x_new),
                    int(y_new)
                ),
                (
                    int(x_old),
                    int(y_old)
                ),
                (
                    0,
                    255,
                    0
                ),
                2
            )


            cv2.circle(
                frame,
                (
                    int(x_new),
                    int(y_new)
                ),
                4,
                (
                    0,
                    0,
                    255
                ),
                -1
            )


        output = cv2.add(
            frame,
            trail_mask
        )


        # ====================================================
        # SAVE VALIDATION FRAME PAIR
        # ====================================================

        if (
            not validation_done
            and frame_number >= validation_target
        ):

            frame1 = old_frame.copy()

            frame2 = frame.copy()


            number_points = min(
                10,
                len(good_new)
            )


            for i in range(
                number_points
            ):

                old_x, old_y = (
                    good_old[i].ravel()
                )

                new_x, new_y = (
                    good_new[i].ravel()
                )


                dx = (
                    new_x
                    -
                    old_x
                )

                dy = (
                    new_y
                    -
                    old_y
                )


                # Mark old point
                cv2.circle(
                    frame1,
                    (
                        int(old_x),
                        int(old_y)
                    ),
                    6,
                    (
                        0,
                        255,
                        0
                    ),
                    -1
                )


                cv2.putText(
                    frame1,
                    str(i + 1),
                    (
                        int(old_x) + 8,
                        int(old_y)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (
                        0,
                        255,
                        0
                    ),
                    2
                )


                # Mark new point
                cv2.circle(
                    frame2,
                    (
                        int(new_x),
                        int(new_y)
                    ),
                    6,
                    (
                        0,
                        0,
                        255
                    ),
                    -1
                )


                cv2.putText(
                    frame2,
                    str(i + 1),
                    (
                        int(new_x) + 8,
                        int(new_y)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (
                        0,
                        0,
                        255
                    ),
                    2
                )


                validation_rows.append(
                    [
                        i + 1,

                        frame_number - 1,

                        frame_number,

                        float(old_x),

                        float(old_y),

                        float(new_x),

                        float(new_y),

                        float(dx),

                        float(dy)
                    ]
                )


            cv2.imwrite(
                VALIDATION_FRAME_1,
                frame1
            )


            cv2.imwrite(
                VALIDATION_FRAME_2,
                frame2
            )


            validation_done = True


        # ====================================================
        # DISPLAY INFORMATION
        # ====================================================

        cv2.putText(
            output,
            f"Frame: {frame_number}",
            (
                20,
                40
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (
                255,
                255,
                255
            ),
            2
        )


        cv2.putText(
            output,
            f"Tracked Points: {len(good_new)}",
            (
                20,
                80
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (
                255,
                255,
                255
            ),
            2
        )


        writer.write(
            output
        )


        cv2.imshow(
            "Optical Flow",
            output
        )


        if (
            cv2.waitKey(1)
            &
            0xFF
        ) == 27:

            break


        # ====================================================
        # NEXT FRAME
        # ====================================================

        old_gray = frame_gray.copy()

        old_frame = frame.copy()

        old_points = good_new.reshape(
            -1,
            1,
            2
        )


        frame_number += 1


    # ========================================================
    # SAVE TRACKING COORDINATES
    # ========================================================

    with open(
        VALIDATION_CSV,
        "w",
        newline=""
    ) as file:

        writer_csv = csv.writer(
            file
        )


        writer_csv.writerow(
            [
                "Point",
                "Frame 1",
                "Frame 2",
                "x1",
                "y1",
                "x2",
                "y2",
                "dx",
                "dy"
            ]
        )


        writer_csv.writerows(
            validation_rows
        )


    cap.release()

    writer.release()

    cv2.destroyAllWindows()


    print()
    print(
        "======================================="
    )

    print(
        "OPTICAL FLOW COMPLETE"
    )

    print(
        "======================================="
    )


    print(
        "Saved:",
        OUTPUT_VIDEO
    )

    print(
        "Saved:",
        VALIDATION_FRAME_1
    )

    print(
        "Saved:",
        VALIDATION_FRAME_2
    )

    print(
        "Saved:",
        VALIDATION_CSV
    )


    print()
    print(
        "TRACKING VALIDATION"
    )


    for row in validation_rows:

        print(
            f"Point {row[0]}:"
        )

        print(
            f"({row[3]:.2f}, {row[4]:.2f})"
            f" -> "
            f"({row[5]:.2f}, {row[6]:.2f})"
        )

        print(
            f"dx = {row[7]:.2f}, "
            f"dy = {row[8]:.2f}"
        )

        print()


if __name__ == "__main__":

    main()