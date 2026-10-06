# Module 3 - Spatial and Fourier Domain Image Blurring

## Objective

This module compares image blurring in the spatial domain with image blurring in the Fourier domain and validates the convolution theorem experimentally.

## Method

The same averaging kernel is applied in two ways. The spatial-domain result is produced by direct 2D convolution. The Fourier-domain result is produced by transforming the image and kernel with the FFT, multiplying them in the frequency domain, and applying the inverse FFT. The two results are compared using MAE, RMSE, and maximum absolute difference.

## Website Demo

Open `module3.html`. The page supports image upload or camera capture, kernel-size selection, spatial convolution, Fourier convolution, a difference image, and numerical validation.

## Experiment Result

The saved experiment used a 5 x 5 averaging kernel on a 213 x 320 image. The obtained values were:

- MAE: 4.409147e-13
- RMSE: 5.625711e-13
- Maximum difference: 1.449507e-12
- Result: Equivalent

The very small numerical differences show that the spatial-domain and Fourier-domain implementations produce the same blur within floating-point numerical precision.

## Saved Output Images

The permanent website result gallery is stored in `module3_results/` and contains the original input image, spatial blur, Fourier blur, and numerical validation output.

## Files

- `module3.html` - web interface
- `module3.js` - spatial convolution, FFT/Fourier convolution, and validation
- `module3_extra.css` - Module 3 layout and saved-result gallery styles
- `module3_results/` - saved experiment outputs
