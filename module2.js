/* =========================================================
   CSc 8830 - Computer Vision
   Module 2 Web Application
   ========================================================= */

let cvReady = false;
let stream = null;
let selectedDeviceId = "";

const CHECKERBOARD_COLS = 9;
const CHECKERBOARD_ROWS = 6;

let calibrationCaptures = [];
let calibration = null;

let measurementFrame = null;
let measurementPoints = [];

let validationFrame = null;
let validationPoints = [];
let validationResults = [];
let validationIndex = 0;
let validationStarted = false;

const OBJECTS = [
  { name: "Car",         width: 5.0,  height: 3.0 },
  { name: "AirPods",     width: 5.5,  height: 2.0 },
  { name: "Bigfoot",     width: 4.3,  height: 6.5 },
  { name: "Ruby",        width: 3.5,  height: 6.0 },
  { name: "Pepper",      width: 3.7,  height: 8.7 },
  { name: "Shoe polish", width: 10.0, height: 4.5 },
  { name: "Container",   width: 8.0,  height: 5.5 },
  { name: "Sardines",    width: 7.0,  height: 2.4 },
  { name: "Key",         width: 2.5,  height: 1.0 },
  { name: "Mouse",       width: 5.0,  height: 2.5 },
  { name: "Playdoh",     width: 4.7,  height: 3.8 },
  { name: "Bose",        width: 8.0,  height: 9.5 },
  { name: "Oil",         width: 6.5,  height: 11.5 },
  { name: "Juice",       width: 5.5,  height: 10.5 },
  { name: "Roku",        width: 4.0,  height: 1.6 },
  { name: "Lotion",      width: 3.5,  height: 14.0 },
  { name: "Vitamin",     width: 9.0,  height: 10.5 },
  { name: "Cup",         width: 7.0,  height: 13.0 },
  { name: "Brownie",     width: 9.0,  height: 1.5 },
  { name: "Egg cup",     width: 5.0,  height: 7.5 }
];

const $ = (id) => document.getElementById(id);

function onOpenCvReady() {
  cvReady = true;
  $("calibrationStatus").textContent = "OpenCV.js loaded. Start the camera.";
}

/* -----------------------------
   STEP NAVIGATION
----------------------------- */

document.querySelectorAll(".step-tab").forEach(button => {
  button.addEventListener("click", () => showStep(button.dataset.step));
});

function showStep(stepId) {
  document.querySelectorAll(".step-tab").forEach(b => {
    b.classList.toggle("active", b.dataset.step === stepId);
  });

  document.querySelectorAll(".step-panel").forEach(panel => {
    panel.classList.toggle("active-panel", panel.id === stepId);
  });
}

/* -----------------------------
   CAMERA
----------------------------- */

$("startCameraBtn").addEventListener("click", startCamera);
$("refreshCamerasBtn").addEventListener("click", loadCameraDevices);
$("stopCameraBtn").addEventListener("click", stopCamera);
$("cameraSelect").addEventListener("change", async () => {
  selectedDeviceId = $("cameraSelect").value;
  if (stream) await startCamera();
});

async function loadCameraDevices() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cameras = devices.filter(d => d.kind === "videoinput");

    const select = $("cameraSelect");
    const previous = select.value;
    select.innerHTML = "";

    cameras.forEach((camera, index) => {
      const option = document.createElement("option");
      option.value = camera.deviceId;
      option.textContent = camera.label || `Camera ${index + 1}`;
      select.appendChild(option);
    });

    if (previous && [...select.options].some(o => o.value === previous)) {
      select.value = previous;
    }

    selectedDeviceId = select.value;
  } catch (err) {
    $("cameraStatus").textContent = `Could not list cameras: ${err.message}`;
  }
}

async function startCamera() {
  try {
    stopCamera();

    let constraints = {
      audio: false,
      video: {
        width: { ideal: 1920 },
        height: { ideal: 1080 }
      }
    };

    if (selectedDeviceId || $("cameraSelect").value) {
      const id = selectedDeviceId || $("cameraSelect").value;
      constraints.video.deviceId = { exact: id };
    }

    stream = await navigator.mediaDevices.getUserMedia(constraints);
    $("video").srcObject = stream;

    await new Promise(resolve => {
      $("video").onloadedmetadata = resolve;
    });

    await $("video").play();

    await loadCameraDevices();

    $("cameraStatus").textContent =
      `Camera started: ${$("video").videoWidth} × ${$("video").videoHeight}`;

    if (cvReady) {
      $("calibrationStatus").textContent =
        "Camera ready. Capture checkerboard views.";
    }
  } catch (err) {
    $("cameraStatus").textContent =
      `Camera error: ${err.message}. Allow camera permission and try again.`;
  }
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }
  $("video").srcObject = null;
}

function captureVideoToCanvas(canvas) {
  if (!stream || !$("video").videoWidth) {
    throw new Error("Start the camera first.");
  }

  canvas.width = $("video").videoWidth;
  canvas.height = $("video").videoHeight;

  const ctx = canvas.getContext("2d");
  ctx.drawImage($("video"), 0, 0, canvas.width, canvas.height);
}

/* -----------------------------
   STEP 1 - CALIBRATION
----------------------------- */

$("captureCheckerboardBtn").addEventListener("click", captureCheckerboard);
$("calibrateBtn").addEventListener("click", calibrateCamera);
$("clearCalibrationBtn").addEventListener("click", clearCalibration);

function captureCheckerboard() {
  if (!cvReady) {
    $("calibrationStatus").textContent = "OpenCV.js is still loading.";
    return;
  }

  try {
    const canvas = $("calibrationCanvas");
    captureVideoToCanvas(canvas);

    let src = cv.imread(canvas);
    let gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

    const patternSize = new cv.Size(CHECKERBOARD_COLS, CHECKERBOARD_ROWS);
    let corners = new cv.Mat();

    const flags = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE;
    const found = cv.findChessboardCorners(gray, patternSize, corners, flags);

    if (!found) {
      $("calibrationStatus").textContent =
        "Checkerboard not detected. Change angle/distance and try again.";
      src.delete();
      gray.delete();
      corners.delete();
      return;
    }

    const criteria = new cv.TermCriteria(
      cv.TermCriteria_EPS + cv.TermCriteria_MAX_ITER,
      30,
      0.001
    );

    cv.cornerSubPix(
      gray,
      corners,
      new cv.Size(11, 11),
      new cv.Size(-1, -1),
      criteria
    );

    cv.drawChessboardCorners(src, patternSize, corners, true);
    cv.imshow(canvas, src);

    const points = [];
    for (let i = 0; i < corners.rows; i++) {
      points.push([
        corners.data32F[i * 2],
        corners.data32F[i * 2 + 1]
      ]);
    }

    calibrationCaptures.push({
      corners: points,
      width: canvas.width,
      height: canvas.height
    });

    $("captureCount").textContent = calibrationCaptures.length;
    $("calibrationStatus").textContent =
      `Successful capture ${calibrationCaptures.length}. Move/tilt the checkerboard and capture again.`;

    src.delete();
    gray.delete();
    corners.delete();
  } catch (err) {
    $("calibrationStatus").textContent = `Calibration capture error: ${err.message}`;
  }
}

function clearCalibration() {
  calibrationCaptures = [];
  calibration = null;
  $("captureCount").textContent = "0";
  $("calibrationResults").classList.add("hidden");
  $("calibrationStatus").textContent = "Calibration captures cleared.";
  localStorage.removeItem("module2Calibration");
}

function calibrateCamera() {
  if (!cvReady) {
    $("calibrationStatus").textContent = "OpenCV.js is not ready.";
    return;
  }

  if (calibrationCaptures.length < 6) {
    $("calibrationStatus").textContent =
      "Capture at least 6 successful checkerboard views. 10 or more is recommended.";
    return;
  }

  if (typeof cv.calibrateCamera !== "function") {
    $("calibrationStatus").textContent =
      "This OpenCV.js build does not expose calibrateCamera. Use the saved Python calibration or a full OpenCV.js build.";
    return;
  }

  const squareMeters = parseFloat($("squareSize").value) / 100.0;

  let objectPoints = new cv.MatVector();
  let imagePoints = new cv.MatVector();

  let cameraMatrix = cv.Mat.eye(3, 3, cv.CV_64F);
  let distCoeffs = new cv.Mat();
  let rvecs = new cv.MatVector();
  let tvecs = new cv.MatVector();

  try {
    for (const capture of calibrationCaptures) {
      let obj = new cv.Mat(
        CHECKERBOARD_ROWS * CHECKERBOARD_COLS,
        1,
        cv.CV_32FC3
      );

      let img = new cv.Mat(
        CHECKERBOARD_ROWS * CHECKERBOARD_COLS,
        1,
        cv.CV_32FC2
      );

      let k = 0;

      for (let r = 0; r < CHECKERBOARD_ROWS; r++) {
        for (let c = 0; c < CHECKERBOARD_COLS; c++) {
          obj.data32F[k * 3] = c * squareMeters;
          obj.data32F[k * 3 + 1] = r * squareMeters;
          obj.data32F[k * 3 + 2] = 0;

          img.data32F[k * 2] = capture.corners[k][0];
          img.data32F[k * 2 + 1] = capture.corners[k][1];

          k++;
        }
      }

      objectPoints.push_back(obj);
      imagePoints.push_back(img);

      obj.delete();
      img.delete();
    }

    const width = calibrationCaptures[0].width;
    const height = calibrationCaptures[0].height;

    const criteria = new cv.TermCriteria(
      cv.TermCriteria_EPS + cv.TermCriteria_MAX_ITER,
      100,
      1e-6
    );

    const rms = cv.calibrateCamera(
      objectPoints,
      imagePoints,
      new cv.Size(width, height),
      cameraMatrix,
      distCoeffs,
      rvecs,
      tvecs,
      0,
      criteria
    );

    const fx = cameraMatrix.doubleAt(0, 0);
    const fy = cameraMatrix.doubleAt(1, 1);
    const cx = cameraMatrix.doubleAt(0, 2);
    const cy = cameraMatrix.doubleAt(1, 2);

    const dist = [];
    const totalDist = distCoeffs.rows * distCoeffs.cols;
    for (let i = 0; i < totalDist; i++) {
      dist.push(distCoeffs.data64F[i]);
    }

    calibration = {
      rms,
      fx,
      fy,
      cx,
      cy,
      width,
      height,
      dist
    };

    localStorage.setItem("module2Calibration", JSON.stringify(calibration));

    displayCalibration();

    $("calibrationStatus").textContent =
      `Calibration complete using ${calibrationCaptures.length} images.`;
  } catch (err) {
    $("calibrationStatus").textContent =
      `Calibration failed: ${err.message}`;
  } finally {
    objectPoints.delete();
    imagePoints.delete();
    cameraMatrix.delete();
    distCoeffs.delete();
    rvecs.delete();
    tvecs.delete();
  }
}

function displayCalibration() {
  if (!calibration) return;

  $("rmsValue").textContent = Number(calibration.rms).toFixed(4) + " px";
  $("fxValue").textContent = calibration.fx.toFixed(2);
  $("fyValue").textContent = calibration.fy.toFixed(2);
  $("cxValue").textContent = calibration.cx.toFixed(2);
  $("cyValue").textContent = calibration.cy.toFixed(2);
  $("resolutionValue").textContent = `${calibration.width} × ${calibration.height}`;
  $("calibrationResults").classList.remove("hidden");
}

function loadSavedCalibration() {
  try {
    const saved = localStorage.getItem("module2Calibration");
    if (saved) {
      calibration = JSON.parse(saved);
      displayCalibration();
      $("calibrationStatus").textContent =
        "Saved browser calibration loaded. You may recalibrate if needed.";
    }
  } catch (_) {}
}

/* -----------------------------
   SHARED MEASUREMENT HELPERS
----------------------------- */

function getCanvasPoint(event, canvas) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * (canvas.width / rect.width),
    y: (event.clientY - rect.top) * (canvas.height / rect.height)
  };
}

function distance(a, b) {
  return Math.hypot(b.x - a.x, b.y - a.y);
}

function drawPoints(canvas, points) {
  const ctx = canvas.getContext("2d");

  points.forEach((p, i) => {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 8, 0, Math.PI * 2);
    ctx.fillStyle = "#00FF66";
    ctx.fill();

    ctx.font = "24px Arial";
    ctx.fillStyle = "#00FF66";
    ctx.fillText(String(i + 1), p.x + 10, p.y - 10);
  });

  if (points.length > 1) {
    ctx.strokeStyle = "#00FF66";
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);

    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i].x, points[i].y);
    }

    if (points.length === 4) ctx.closePath();
    ctx.stroke();
  }
}

function calculateDimensions(points, distanceMeters) {
  if (!calibration) {
    throw new Error("Complete Step 1 calibration first.");
  }

  if (points.length !== 4) {
    throw new Error("Select exactly 4 corners.");
  }

  const [tl, tr, br, bl] = points;

  const topWidth = distance(tl, tr);
  const bottomWidth = distance(bl, br);
  const leftHeight = distance(tl, bl);
  const rightHeight = distance(tr, br);

  const widthPixels = (topWidth + bottomWidth) / 2;
  const heightPixels = (leftHeight + rightHeight) / 2;

  const widthCm = (widthPixels * distanceMeters / calibration.fx) * 100;
  const heightCm = (heightPixels * distanceMeters / calibration.fy) * 100;

  return {
    widthPixels,
    heightPixels,
    widthCm,
    heightCm
  };
}

/* -----------------------------
   STEP 2
----------------------------- */

$("captureObjectBtn").addEventListener("click", () => {
  try {
    const canvas = $("measurementCanvas");
    captureVideoToCanvas(canvas);
    measurementFrame = canvas.toDataURL();
    measurementPoints = [];
    $("measurementResults").classList.add("hidden");
    $("measurementStatus").textContent =
      "Image captured. Click Top-left, Top-right, Bottom-right, Bottom-left.";
  } catch (err) {
    $("measurementStatus").textContent = err.message;
  }
});

$("measurementCanvas").addEventListener("click", event => {
  if (!measurementFrame) return;
  if (measurementPoints.length >= 4) return;

  measurementPoints.push(getCanvasPoint(event, $("measurementCanvas")));
  restoreMeasurementCanvas();
  drawPoints($("measurementCanvas"), measurementPoints);

  $("measurementStatus").textContent =
    `${measurementPoints.length}/4 corners selected.`;
});

$("resetCornersBtn").addEventListener("click", () => {
  measurementPoints = [];
  restoreMeasurementCanvas();
  $("measurementStatus").textContent = "Corners reset.";
});

$("calculateObjectBtn").addEventListener("click", () => {
  try {
    const z = parseFloat($("step2Distance").value);
    const result = calculateDimensions(measurementPoints, z);

    $("pixelWidthResult").textContent = result.widthPixels.toFixed(2) + " px";
    $("pixelHeightResult").textContent = result.heightPixels.toFixed(2) + " px";
    $("widthResult").textContent = result.widthCm.toFixed(2) + " cm";
    $("heightResult").textContent = result.heightCm.toFixed(2) + " cm";
    $("measurementResults").classList.remove("hidden");

    $("measurementStatus").textContent = "Measurement calculated.";
  } catch (err) {
    $("measurementStatus").textContent = err.message;
  }
});

function restoreMeasurementCanvas() {
  if (!measurementFrame) return;
  const img = new Image();
  img.onload = () => {
    const canvas = $("measurementCanvas");
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    drawPoints(canvas, measurementPoints);
  };
  img.src = measurementFrame;
}

/* -----------------------------
   STEP 3
----------------------------- */

$("validationMode").addEventListener("change", () => {
  $("manualFields").classList.toggle(
    "hidden",
    $("validationMode").value !== "manual"
  );
});

$("beginValidationBtn").addEventListener("click", beginValidation);
$("captureValidationBtn").addEventListener("click", captureValidation);
$("resetValidationCornersBtn").addEventListener("click", () => {
  validationPoints = [];
  restoreValidationCanvas();
});
$("saveValidationBtn").addEventListener("click", saveValidationMeasurement);
$("finishValidationBtn").addEventListener("click", generateReport);
$("clearValidationBtn").addEventListener("click", clearValidation);

$("validationCanvas").addEventListener("click", event => {
  if (!validationFrame || validationPoints.length >= 4) return;

  validationPoints.push(getCanvasPoint(event, $("validationCanvas")));
  restoreValidationCanvas();
  drawPoints($("validationCanvas"), validationPoints);

  $("validationStatus").textContent =
    `${validationPoints.length}/4 corners selected.`;
});

function beginValidation() {
  if (!calibration) {
    $("validationStatus").textContent =
      "Complete Step 1 calibration before validation.";
    return;
  }

  validationResults = [];
  validationIndex = 0;
  validationStarted = true;
  validationFrame = null;
  validationPoints = [];

  $("validationTable").querySelector("tbody").innerHTML = "";
  updateCurrentObject();

  $("validationStatus").textContent =
    "Validation started. Capture the current object.";
}

function getCurrentGroundTruth() {
  if ($("validationMode").value === "predefined") {
    return OBJECTS[validationIndex] || null;
  }

  const name = $("manualName").value.trim();
  const width = parseFloat($("manualWidth").value);
  const height = parseFloat($("manualHeight").value);

  if (!name || !Number.isFinite(width) || !Number.isFinite(height)) {
    return null;
  }

  return { name, width, height };
}

function updateCurrentObject() {
  if ($("validationMode").value === "predefined") {
    if (validationIndex >= OBJECTS.length) {
      $("currentObject").textContent = "All 20 predefined objects completed.";
      return;
    }

    const obj = OBJECTS[validationIndex];
    $("currentObject").innerHTML =
      `<b>${validationIndex + 1}/20 - ${obj.name}</b><br>` +
      `Actual width: ${obj.width.toFixed(2)} cm | ` +
      `Actual height: ${obj.height.toFixed(2)} cm`;
  } else {
    $("currentObject").textContent =
      "Enter the object name, actual width and actual height above.";
  }
}

function captureValidation() {
  if (!validationStarted) {
    $("validationStatus").textContent = "Press Begin Validation first.";
    return;
  }

  const truth = getCurrentGroundTruth();
  if (!truth) {
    $("validationStatus").textContent =
      "Enter valid object name, actual width and actual height.";
    return;
  }

  try {
    const canvas = $("validationCanvas");
    captureVideoToCanvas(canvas);
    validationFrame = canvas.toDataURL();
    validationPoints = [];

    $("validationStatus").textContent =
      "Image captured. Select the 4 corners.";
  } catch (err) {
    $("validationStatus").textContent = err.message;
  }
}

function restoreValidationCanvas() {
  if (!validationFrame) return;

  const img = new Image();
  img.onload = () => {
    const canvas = $("validationCanvas");
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    drawPoints(canvas, validationPoints);
  };
  img.src = validationFrame;
}

function saveValidationMeasurement() {
  if (!validationStarted) return;

  const truth = getCurrentGroundTruth();
  if (!truth) {
    $("validationStatus").textContent =
      "Ground-truth dimensions are missing.";
    return;
  }

  try {
    const z = parseFloat($("validationDistance").value);
    const estimate = calculateDimensions(validationPoints, z);

    const widthError = estimate.widthCm - truth.width;
    const heightError = estimate.heightCm - truth.height;

    const widthAbs = Math.abs(widthError);
    const heightAbs = Math.abs(heightError);

    const widthPct = widthAbs / truth.width * 100;
    const heightPct = heightAbs / truth.height * 100;

    const row = {
      measurement: validationResults.length + 1,
      object: truth.name,
      distance: z,
      actualWidth: truth.width,
      cameraWidth: estimate.widthCm,
      widthError,
      widthAbs,
      widthPct,
      actualHeight: truth.height,
      cameraHeight: estimate.heightCm,
      heightError,
      heightAbs,
      heightPct
    };

    validationResults.push(row);
    appendValidationRow(row);

    validationFrame = null;
    validationPoints = [];

    if ($("validationMode").value === "predefined") {
      validationIndex++;
      updateCurrentObject();

      if (validationIndex >= OBJECTS.length) {
        $("validationStatus").textContent =
          "All 20 predefined measurements completed. Generate the report.";
      } else {
        $("validationStatus").textContent =
          "Measurement saved. Place the next object and capture it.";
      }
    } else {
      validationIndex++;
      $("manualName").value = "";
      $("manualWidth").value = "";
      $("manualHeight").value = "";
      updateCurrentObject();
      $("validationStatus").textContent =
        "Measurement saved. Enter the next object's dimensions.";
    }
  } catch (err) {
    $("validationStatus").textContent = err.message;
  }
}

function appendValidationRow(r) {
  const tr = document.createElement("tr");
  tr.innerHTML = `
    <td>${r.measurement}</td>
    <td>${r.object}</td>
    <td>${r.actualWidth.toFixed(2)}</td>
    <td>${r.cameraWidth.toFixed(2)}</td>
    <td>${r.actualHeight.toFixed(2)}</td>
    <td>${r.cameraHeight.toFixed(2)}</td>
    <td>${r.widthPct.toFixed(2)}%</td>
    <td>${r.heightPct.toFixed(2)}%</td>
  `;

  $("validationTable").querySelector("tbody").appendChild(tr);
}

function clearValidation() {
  validationResults = [];
  validationIndex = 0;
  validationStarted = false;
  validationFrame = null;
  validationPoints = [];
  $("validationTable").querySelector("tbody").innerHTML = "";
  $("currentObject").textContent = 'Press "Begin Validation" to start.';
  $("validationStatus").textContent = "Validation cleared.";
}

/* -----------------------------
   FINAL REPORT
----------------------------- */

function mean(values) {
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function rmse(values) {
  return Math.sqrt(mean(values.map(v => v * v)));
}

function std(values) {
  const m = mean(values);
  return Math.sqrt(mean(values.map(v => (v - m) ** 2)));
}

function generateReport() {
  if (!validationResults.length) {
    $("validationStatus").textContent =
      "Complete at least one validation measurement first.";
    return;
  }

  const wErrors = validationResults.map(r => r.widthError);
  const hErrors = validationResults.map(r => r.heightError);
  const wAbs = validationResults.map(r => r.widthAbs);
  const hAbs = validationResults.map(r => r.heightAbs);
  const wPct = validationResults.map(r => r.widthPct);
  const hPct = validationResults.map(r => r.heightPct);

  const wMean = mean(wErrors);
  const hMean = mean(hErrors);
  const wMAE = mean(wAbs);
  const hMAE = mean(hAbs);
  const wRMSE = rmse(wErrors);
  const hRMSE = rmse(hErrors);
  const wMAPE = mean(wPct);
  const hMAPE = mean(hPct);
  const wStd = std(wErrors);
  const hStd = std(hErrors);
  const overall = (wMAPE + hMAPE) / 2;

  $("reportCount").textContent = validationResults.length;
  $("reportWMean").textContent = wMean.toFixed(3) + " cm";
  $("reportWMAE").textContent = wMAE.toFixed(3) + " cm";
  $("reportWRMSE").textContent = wRMSE.toFixed(3) + " cm";
  $("reportWMAPE").textContent = wMAPE.toFixed(2) + "%";
  $("reportWStd").textContent = wStd.toFixed(3) + " cm";

  $("reportHMean").textContent = hMean.toFixed(3) + " cm";
  $("reportHMAE").textContent = hMAE.toFixed(3) + " cm";
  $("reportHRMSE").textContent = hRMSE.toFixed(3) + " cm";
  $("reportHMAPE").textContent = hMAPE.toFixed(2) + "%";
  $("reportHStd").textContent = hStd.toFixed(3) + " cm";

  $("reportOverall").textContent = overall.toFixed(2) + "%";

  $("reportSummary").innerHTML =
    `The system was validated using <b>${validationResults.length}</b> measurement(s). ` +
    `Width MAPE was <b>${wMAPE.toFixed(2)}%</b>, height MAPE was ` +
    `<b>${hMAPE.toFixed(2)}%</b>, and the combined overall average ` +
    `percentage error was <b>${overall.toFixed(2)}%</b>.`;

  const tbody = $("reportTable").querySelector("tbody");
  tbody.innerHTML = "";

  validationResults.forEach(r => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${r.measurement}</td>
      <td>${r.object}</td>
      <td>${r.actualWidth.toFixed(2)} cm</td>
      <td>${r.cameraWidth.toFixed(2)} cm</td>
      <td>${r.actualHeight.toFixed(2)} cm</td>
      <td>${r.cameraHeight.toFixed(2)} cm</td>
    `;
    tbody.appendChild(tr);
  });

  $("reportStats").classList.remove("hidden");
  showStep("report");
}

$("downloadCsvBtn").addEventListener("click", downloadCsv);
$("printReportBtn").addEventListener("click", () => window.print());

function downloadCsv() {
  if (!validationResults.length) return;

  const rows = [
    [
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
  ];

  validationResults.forEach(r => {
    rows.push([
      r.measurement,
      r.object,
      r.distance,
      r.actualWidth,
      r.cameraWidth,
      r.widthError,
      r.widthAbs,
      r.widthPct,
      r.actualHeight,
      r.cameraHeight,
      r.heightError,
      r.heightAbs,
      r.heightPct
    ]);
  });

  const csv = rows
    .map(row =>
      row.map(value => `"${String(value).replaceAll('"', '""')}"`).join(",")
    )
    .join("\n");

  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = "step3_validation_results_web.csv";
  a.click();

  URL.revokeObjectURL(url);
}

/* -----------------------------
   INITIALIZATION
----------------------------- */

window.addEventListener("load", async () => {
  loadSavedCalibration();

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    $("cameraStatus").textContent =
      "This browser does not support camera access.";
  }

  $("validationMode").dispatchEvent(new Event("change"));
});
