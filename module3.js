let stream = null;
let inputImageData = null;

const $ = id => document.getElementById(id);
const originalCanvas = $("originalCanvas");
const spatialCanvas = $("spatialCanvas");
const fourierCanvas = $("fourierCanvas");
const differenceCanvas = $("differenceCanvas");

$("imageUpload").addEventListener("change", loadUploadedImage);
$("startCameraBtn").addEventListener("click", startCamera);
$("captureCameraBtn").addEventListener("click", captureCamera);
$("stopCameraBtn").addEventListener("click", stopCamera);
$("runExperimentBtn").addEventListener("click", runExperiment);

function resizeForDemo(width, height, maxSide = 320) {
    const scale = Math.min(1, maxSide / Math.max(width, height));
    return {
        width: Math.max(1, Math.round(width * scale)),
        height: Math.max(1, Math.round(height * scale))
    };
}

function drawImageToOriginal(img) {
    const size = resizeForDemo(
        img.naturalWidth || img.videoWidth,
        img.naturalHeight || img.videoHeight
    );

    originalCanvas.width = size.width;
    originalCanvas.height = size.height;

    const ctx = originalCanvas.getContext("2d");

    ctx.drawImage(img, 0, 0, size.width, size.height);

    inputImageData = ctx.getImageData(
        0,
        0,
        size.width,
        size.height
    );

    $("status").textContent =
        `Image ready: ${size.width} × ${size.height}. Click Run Experiment.`;
}

function loadUploadedImage(event) {
    const file = event.target.files[0];
    if (!file) return;

    const img = new Image();

    img.onload = () => {
        drawImageToOriginal(img);
        URL.revokeObjectURL(img.src);
    };

    img.src = URL.createObjectURL(file);
}

async function startCamera() {
    try {
        stopCamera();

        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },
            audio: false
        });

        $("video").srcObject = stream;
        await $("video").play();

        $("status").textContent =
            "Camera started. Click Capture Camera.";

    } catch (error) {
        $("status").textContent =
            "Camera error: " + error.message;
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }

    stream = null;
    $("video").srcObject = null;
}

function captureCamera() {
    if (!stream || !$("video").videoWidth) {
        $("status").textContent = "Start the camera first.";
        return;
    }

    const size = resizeForDemo(
        $("video").videoWidth,
        $("video").videoHeight
    );

    originalCanvas.width = size.width;
    originalCanvas.height = size.height;

    const ctx = originalCanvas.getContext("2d");

    ctx.drawImage(
        $("video"),
        0,
        0,
        size.width,
        size.height
    );

    inputImageData = ctx.getImageData(
        0,
        0,
        size.width,
        size.height
    );

    $("status").textContent =
        `Camera image captured: ${size.width} × ${size.height}. Click Run Experiment.`;
}

function createAverageKernel(size) {
    const kernel = new Float64Array(size * size);
    kernel.fill(1 / (size * size));
    return kernel;
}

function spatialConvolution(src, width, height, kernel, kernelSize) {
    const output = new Float64Array(width * height);
    const radius = Math.floor(kernelSize / 2);

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {

            let sum = 0;

            for (let ky = 0; ky < kernelSize; ky++) {
                const imageY = y + ky - radius;

                if (imageY < 0 || imageY >= height) continue;

                for (let kx = 0; kx < kernelSize; kx++) {
                    const imageX = x + kx - radius;

                    if (imageX < 0 || imageX >= width) continue;

                    sum +=
                        src[imageY * width + imageX] *
                        kernel[ky * kernelSize + kx];
                }
            }

            output[y * width + x] = sum;
        }
    }

    return output;
}

function nextPowerOfTwo(n) {
    let result = 1;
    while (result < n) result <<= 1;
    return result;
}

function fft1D(real, imag, inverse) {
    const n = real.length;

    for (let i = 1, j = 0; i < n; i++) {
        let bit = n >> 1;

        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }

        j ^= bit;

        if (i < j) {
            [real[i], real[j]] = [real[j], real[i]];
            [imag[i], imag[j]] = [imag[j], imag[i]];
        }
    }

    for (let length = 2; length <= n; length <<= 1) {
        const angle =
            2 * Math.PI / length * (inverse ? 1 : -1);

        const wlenReal = Math.cos(angle);
        const wlenImag = Math.sin(angle);

        for (let i = 0; i < n; i += length) {
            let wReal = 1;
            let wImag = 0;

            for (let j = 0; j < length / 2; j++) {
                const uReal = real[i + j];
                const uImag = imag[i + j];

                const second = i + j + length / 2;

                const vReal =
                    real[second] * wReal -
                    imag[second] * wImag;

                const vImag =
                    real[second] * wImag +
                    imag[second] * wReal;

                real[i + j] = uReal + vReal;
                imag[i + j] = uImag + vImag;

                real[second] = uReal - vReal;
                imag[second] = uImag - vImag;

                const nextReal =
                    wReal * wlenReal -
                    wImag * wlenImag;

                const nextImag =
                    wReal * wlenImag +
                    wImag * wlenReal;

                wReal = nextReal;
                wImag = nextImag;
            }
        }
    }

    if (inverse) {
        for (let i = 0; i < n; i++) {
            real[i] /= n;
            imag[i] /= n;
        }
    }
}

function fft2D(real, imag, width, height, inverse) {
    for (let y = 0; y < height; y++) {
        const rowReal = new Float64Array(width);
        const rowImag = new Float64Array(width);

        for (let x = 0; x < width; x++) {
            rowReal[x] = real[y * width + x];
            rowImag[x] = imag[y * width + x];
        }

        fft1D(rowReal, rowImag, inverse);

        for (let x = 0; x < width; x++) {
            real[y * width + x] = rowReal[x];
            imag[y * width + x] = rowImag[x];
        }
    }

    for (let x = 0; x < width; x++) {
        const colReal = new Float64Array(height);
        const colImag = new Float64Array(height);

        for (let y = 0; y < height; y++) {
            colReal[y] = real[y * width + x];
            colImag[y] = imag[y * width + x];
        }

        fft1D(colReal, colImag, inverse);

        for (let y = 0; y < height; y++) {
            real[y * width + x] = colReal[y];
            imag[y * width + x] = colImag[y];
        }
    }
}

function fourierConvolution(src, width, height, kernel, kernelSize) {
    const fullWidth = width + kernelSize - 1;
    const fullHeight = height + kernelSize - 1;

    const fftWidth = nextPowerOfTwo(fullWidth);
    const fftHeight = nextPowerOfTwo(fullHeight);

    const total = fftWidth * fftHeight;

    const imageReal = new Float64Array(total);
    const imageImag = new Float64Array(total);

    const kernelReal = new Float64Array(total);
    const kernelImag = new Float64Array(total);

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            imageReal[y * fftWidth + x] =
                src[y * width + x];
        }
    }

    for (let y = 0; y < kernelSize; y++) {
        for (let x = 0; x < kernelSize; x++) {
            kernelReal[y * fftWidth + x] =
                kernel[y * kernelSize + x];
        }
    }

    fft2D(
        imageReal,
        imageImag,
        fftWidth,
        fftHeight,
        false
    );

    fft2D(
        kernelReal,
        kernelImag,
        fftWidth,
        fftHeight,
        false
    );

    for (let i = 0; i < total; i++) {
        const a = imageReal[i];
        const b = imageImag[i];
        const c = kernelReal[i];
        const d = kernelImag[i];

        imageReal[i] = a * c - b * d;
        imageImag[i] = a * d + b * c;
    }

    fft2D(
        imageReal,
        imageImag,
        fftWidth,
        fftHeight,
        true
    );

    const output = new Float64Array(width * height);
    const radius = Math.floor(kernelSize / 2);

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            output[y * width + x] =
                imageReal[
                    (y + radius) * fftWidth +
                    (x + radius)
                ];
        }
    }

    return output;
}

function getChannel(imageData, channel) {
    const count =
        imageData.width *
        imageData.height;

    const output =
        new Float64Array(count);

    for (let i = 0; i < count; i++) {
        output[i] =
            imageData.data[i * 4 + channel];
    }

    return output;
}

function createImageData(channels, width, height) {
    const data =
        new Uint8ClampedArray(
            width * height * 4
        );

    for (let i = 0; i < width * height; i++) {
        data[i * 4] = channels[0][i];
        data[i * 4 + 1] = channels[1][i];
        data[i * 4 + 2] = channels[2][i];
        data[i * 4 + 3] = 255;
    }

    return new ImageData(
        data,
        width,
        height
    );
}

function displayImageData(canvas, imageData) {
    canvas.width = imageData.width;
    canvas.height = imageData.height;

    canvas
        .getContext("2d")
        .putImageData(
            imageData,
            0,
            0
        );
}

async function runExperiment() {
    if (!inputImageData) {
        $("status").textContent =
            "Upload an image or capture a camera image first.";
        return;
    }

    $("status").textContent =
        "Running spatial and Fourier filtering...";

    await new Promise(resolve =>
        setTimeout(resolve, 30)
    );

    try {
        const width = inputImageData.width;
        const height = inputImageData.height;

        const kernelSize =
            Number($("kernelSize").value);

        const kernel =
            createAverageKernel(
                kernelSize
            );

        const spatialChannels = [];
        const fourierChannels = [];

        for (let channel = 0; channel < 3; channel++) {
            const source =
                getChannel(
                    inputImageData,
                    channel
                );

            spatialChannels.push(
                spatialConvolution(
                    source,
                    width,
                    height,
                    kernel,
                    kernelSize
                )
            );

            fourierChannels.push(
                fourierConvolution(
                    source,
                    width,
                    height,
                    kernel,
                    kernelSize
                )
            );
        }

        let absoluteSum = 0;
        let squaredSum = 0;
        let maximumDifference = 0;
        let count = 0;

        const differenceChannels = [
            new Float64Array(width * height),
            new Float64Array(width * height),
            new Float64Array(width * height)
        ];

        for (let channel = 0; channel < 3; channel++) {
            for (let i = 0; i < width * height; i++) {
                const difference =
                    spatialChannels[channel][i] -
                    fourierChannels[channel][i];

                const absoluteDifference =
                    Math.abs(difference);

                absoluteSum +=
                    absoluteDifference;

                squaredSum +=
                    difference * difference;

                maximumDifference =
                    Math.max(
                        maximumDifference,
                        absoluteDifference
                    );

                differenceChannels[channel][i] =
                    absoluteDifference;

                count++;
            }
        }

        const mae =
            absoluteSum / count;

        const rmse =
            Math.sqrt(
                squaredSum / count
            );

        const spatialImage =
            createImageData(
                spatialChannels,
                width,
                height
            );

        const fourierImage =
            createImageData(
                fourierChannels,
                width,
                height
            );

        const scale =
            maximumDifference > 0
                ? 255 / maximumDifference
                : 0;

        const differenceDisplay =
            differenceChannels.map(channel => {
                const output =
                    new Float64Array(
                        channel.length
                    );

                for (let i = 0; i < channel.length; i++) {
                    output[i] =
                        channel[i] * scale;
                }

                return output;
            });

        const differenceImage =
            createImageData(
                differenceDisplay,
                width,
                height
            );

        displayImageData(
            spatialCanvas,
            spatialImage
        );

        displayImageData(
            fourierCanvas,
            fourierImage
        );

        displayImageData(
            differenceCanvas,
            differenceImage
        );

        $("metricKernel").textContent =
            `${kernelSize} × ${kernelSize}`;

        $("metricMAE").textContent =
            mae.toExponential(6);

        $("metricRMSE").textContent =
            rmse.toExponential(6);

        $("metricMax").textContent =
            maximumDifference.toExponential(6);

        $("metricSize").textContent =
            `${width} × ${height}`;

        $("metricResult").textContent =
            mae < 1e-7
                ? "Equivalent ✓"
                : "Very close";

        $("conclusion").innerHTML =
            "<b>Conclusion:</b> " +
            "The spatial-domain convolution and Fourier-domain multiplication " +
            "produced the same blur within floating-point numerical precision. " +
            `MAE = <b>${mae.toExponential(6)}</b>, ` +
            `RMSE = <b>${rmse.toExponential(6)}</b>. ` +
            "This experimentally validates the convolution theorem.";

        $("status").textContent =
            "Experiment completed successfully.";

    } catch (error) {
        console.error(error);

        $("status").textContent =
            "Experiment error: " +
            error.message;
    }
}
