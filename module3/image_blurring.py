"""
CSc 8830 - Computer Vision
Module 3 Assignment

Image blurring using:
1. Spatial-domain filtering
2. Fourier-domain filtering

Goal:
Show that convolution in the spatial domain is equivalent
to multiplication in the Fourier domain.

Controls:
SPACE = Capture image
ESC   = Exit
"""

import cv2
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

CAMERA_INDEX = 1
KERNEL_SIZE = 5


# ============================================================
# CREATE AVERAGING FILTER
# ============================================================

def create_average_kernel(size):

    kernel = np.ones(
        (size, size),
        dtype=np.float64
    )

    kernel = kernel / (size * size)

    return kernel


# ============================================================
# SPATIAL DOMAIN FILTERING
# ============================================================

def spatial_blur(image, kernel):

    image_float = image.astype(
        np.float64
    )

    blurred = cv2.filter2D(
        image_float,
        -1,
        kernel,
        borderType=cv2.BORDER_CONSTANT
    )

    return blurred


# ============================================================
# FOURIER DOMAIN FILTERING
# ============================================================

def fourier_blur_channel(
    channel,
    kernel
):

    image_h, image_w = channel.shape

    kernel_h, kernel_w = kernel.shape


    # Full convolution size
    full_h = image_h + kernel_h - 1
    full_w = image_w + kernel_w - 1


    # Fourier transform of image
    F = np.fft.fft2(
        channel,
        s=(full_h, full_w)
    )


    # Fourier transform of kernel
    H = np.fft.fft2(
        kernel,
        s=(full_h, full_w)
    )


    # Multiplication in frequency domain
    G = F * H


    # Inverse Fourier transform
    result_full = np.fft.ifft2(
        G
    ).real


    # Crop back to original image size
    offset_y = kernel_h // 2
    offset_x = kernel_w // 2


    result = result_full[
        offset_y:offset_y + image_h,
        offset_x:offset_x + image_w
    ]


    return result


def fourier_blur(
    image,
    kernel
):

    image_float = image.astype(
        np.float64
    )


    channels = cv2.split(
        image_float
    )


    output_channels = []


    for channel in channels:

        blurred_channel = fourier_blur_channel(
            channel,
            kernel
        )

        output_channels.append(
            blurred_channel
        )


    result = cv2.merge(
        output_channels
    )


    return result


# ============================================================
# VALIDATION
# ============================================================

def calculate_error(
    spatial,
    fourier
):

    difference = (
        spatial
        -
        fourier
    )


    absolute_difference = np.abs(
        difference
    )


    mae = np.mean(
        absolute_difference
    )


    rmse = np.sqrt(
        np.mean(
            difference ** 2
        )
    )


    max_difference = np.max(
        absolute_difference
    )


    return (
        mae,
        rmse,
        max_difference,
        absolute_difference
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

def show_results(
    original,
    spatial,
    fourier
):

    (
        mae,
        rmse,
        max_difference,
        difference
    ) = calculate_error(
        spatial,
        fourier
    )


    spatial_display = np.clip(
        spatial,
        0,
        255
    ).astype(
        np.uint8
    )


    fourier_display = np.clip(
        fourier,
        0,
        255
    ).astype(
        np.uint8
    )


    # Make tiny numerical difference visible
    if np.max(difference) > 0:

        difference_display = (
            difference
            /
            np.max(difference)
            *
            255
        ).astype(
            np.uint8
        )

    else:

        difference_display = np.zeros_like(
            original
        )


    print()
    print(
        "======================================="
    )

    print(
        "MODULE 3 VALIDATION"
    )

    print(
        "======================================="
    )


    print(
        "Kernel Size:",
        KERNEL_SIZE,
        "x",
        KERNEL_SIZE
    )


    print()

    print(
        "MAE:",
        mae
    )


    print(
        "RMSE:",
        rmse
    )


    print(
        "Maximum Difference:",
        max_difference
    )


    print()
    print(
        "If these values are extremely close"
    )

    print(
        "to zero, spatial convolution and"
    )

    print(
        "Fourier multiplication produced"
    )

    print(
        "the same result."
    )

    print(
        "======================================="
    )


    # Show results
    cv2.imshow(
        "Original Image",
        original
    )


    cv2.imshow(
        "Spatial Blur",
        spatial_display
    )


    cv2.imshow(
        "Fourier Blur",
        fourier_display
    )


    cv2.imshow(
        "Difference",
        difference_display
    )


    # Save results
    cv2.imwrite(
        "original.png",
        original
    )


    cv2.imwrite(
        "spatial_blur.png",
        spatial_display
    )


    cv2.imwrite(
        "fourier_blur.png",
        fourier_display
    )


    cv2.imwrite(
        "difference.png",
        difference_display
    )


    print()
    print(
        "Saved:"
    )

    print(
        "original.png"
    )

    print(
        "spatial_blur.png"
    )

    print(
        "fourier_blur.png"
    )

    print(
        "difference.png"
    )


    print()
    print(
        "Press any key to close the result windows."
    )


    cv2.waitKey(
        0
    )

    cv2.destroyAllWindows()


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print()
    print(
        "CSc 8830 - Module 3"
    )

    print(
        "Image Blurring"
    )

    print()
    print(
        "Opening phone camera..."
    )


    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_AVFOUNDATION
    )


    if not cap.isOpened():

        print()
        print(
            "Could not open camera index",
            CAMERA_INDEX
        )

        print(
            "Try CAMERA_INDEX = 0 if needed."
        )

        return


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
        "SPACE = Capture"
    )

    print(
        "ESC = Exit"
    )


    captured_image = None


    while True:

        ret, frame = cap.read()


        if not ret:

            print(
                "Could not read camera frame."
            )

            break


        display_frame = frame.copy()


        cv2.putText(
            display_frame,
            "SPACE: Capture   ESC: Exit",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )


        cv2.imshow(
            "Module 3 - Phone Camera",
            display_frame
        )


        key = cv2.waitKey(
            1
        ) & 0xFF


        # SPACE
        if key == 32:

            captured_image = frame.copy()

            print()
            print(
                "Image captured successfully."
            )

            break


        # ESC
        elif key == 27:

            print()
            print(
                "Exited."
            )

            break


    cap.release()

    cv2.destroyAllWindows()


    if captured_image is None:

        return


    # ========================================================
    # CREATE FILTER
    # ========================================================

    kernel = create_average_kernel(
        KERNEL_SIZE
    )


    print()
    print(
        "Averaging filter:"
    )

    print(
        kernel
    )


    # ========================================================
    # SPATIAL DOMAIN
    # ========================================================

    print()
    print(
        "Running spatial-domain filtering..."
    )


    spatial = spatial_blur(
        captured_image,
        kernel
    )


    # ========================================================
    # FOURIER DOMAIN
    # ========================================================

    print(
        "Running Fourier-domain filtering..."
    )


    fourier = fourier_blur(
        captured_image,
        kernel
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    show_results(
        captured_image,
        spatial,
        fourier
    )


if __name__ == "__main__":

    main()