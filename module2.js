let stream = null;
let selectedDeviceId = "";
const CALIBRATION_MAX_WIDTH = 1280;
const JPEG_QUALITY = 0.72;

let calibrationImages = [];
let calibration = null;

let measurementFrame = null;
let measurementPoints = [];

let validationFrame = null;
let validationPoints = [];
let validationResults = [];
let validationIndex = 0;
let validationStarted = false;

const OBJECTS = [
  { name: "Car", width: 5.0, height: 3.0 },
  { name: "AirPods", width: 5.5, height: 2.0 },
  { name: "Bigfoot", width: 4.3, height: 6.5 },
  { name: "Ruby", width: 3.5, height: 6.0 },
  { name: "Pepper", width: 3.7, height: 8.7 },
  { name: "Shoe polish", width: 10.0, height: 4.5 },
  { name: "Container", width: 8.0, height: 5.5 },
  { name: "Sardines", width: 7.0, height: 2.4 },
  { name: "Key", width: 2.5, height: 1.0 },
  { name: "Mouse", width: 5.0, height: 2.5 },
  { name: "Playdoh", width: 4.7, height: 3.8 },
  { name: "Bose", width: 8.0, height: 9.5 },
  { name: "Oil", width: 6.5, height: 11.5 },
  { name: "Juice", width: 5.5, height: 10.5 },
  { name: "Roku", width: 4.0, height: 1.6 },
  { name: "Lotion", width: 3.5, height: 14.0 },
  { name: "Vitamin", width: 9.0, height: 10.5 },
  { name: "Cup", width: 7.0, height: 13.0 },
  { name: "Brownie", width: 9.0, height: 1.5 },
  { name: "Egg cup", width: 5.0, height: 7.5 },
];

const $ = (id) => document.getElementById(id);

document.querySelectorAll(".step-tab").forEach((btn) => {
  btn.addEventListener("click", () => showStep(btn.dataset.step));
});

function showStep(id) {
  document.querySelectorAll(".step-tab").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.step === id);
  });
  document.querySelectorAll(".step-panel").forEach((panel) => {
    panel.classList.toggle("active-panel", panel.id === id);
  });
}

// CAMERA
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
    const cameras = devices.filter((d) => d.kind === "videoinput");
    const select = $("cameraSelect");
    const previous = select.value;
    select.innerHTML = "";

    cameras.forEach((camera, i) => {
      const option = document.createElement("option");
      option.value = camera.deviceId;
      option.textContent = camera.label || `Camera ${i + 1}`;
      select.appendChild(option);
    });

    if (previous && [...select.options].some((o) => o.value === previous)) {
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

    const videoConstraints = {
      width: { ideal: 1920 },
      height: { ideal: 1080 },
    };

    const id = selectedDeviceId || $("cameraSelect").value;
    if (id) videoConstraints.deviceId = { exact: id };

    stream = await navigator.mediaDevices.getUserMedia({
      video: videoConstraints,
      audio: false,
    });

    $("video").srcObject = stream;
    await $("video").play();
    await loadCameraDevices();

    $("cameraStatus").textContent =
      `Camera started: ${$("video").videoWidth} × ${$("video").videoHeight}`;
  } catch (err) {
    $("cameraStatus").textContent = `Camera error: ${err.message}`;
  }
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  $("video").srcObject = null;
}

function captureVideoToCanvas(canvas, maxWidth = null) {
  if (!stream || !$("video").videoWidth) {
    throw new Error("Start the camera first.");
  }

  let width = $("video").videoWidth;
  let height = $("video").videoHeight;

  if (maxWidth && width > maxWidth) {
    const scale = maxWidth / width;
    width = Math.round(width * scale);
    height = Math.round(height * scale);
  }

  canvas.width = width;
  canvas.height = height;
  canvas.getContext("2d").drawImage($("video"), 0, 0, width, height);
}

// STEP 1
$("captureCheckerboardBtn").addEventListener("click", () => {
  try {
    const canvas = $("calibrationCanvas");
    captureVideoToCanvas(canvas, CALIBRATION_MAX_WIDTH);

    const dataUrl = canvas.toDataURL("image/jpeg", JPEG_QUALITY);
    calibrationImages.push(dataUrl);

    $("captureCount").textContent = calibrationImages.length;
    $("calibrationStatus").textContent =
      `Captured image ${calibrationImages.length}. Move/tilt the checkerboard and capture another view.`;
  } catch (err) {
    $("calibrationStatus").textContent = err.message;
  }
});

$("clearCalibrationBtn").addEventListener("click", () => {
  calibrationImages = [];
  calibration = null;
  $("captureCount").textContent = "0";
  $("calibrationResults").classList.add("hidden");
  $("calibrationStatus").textContent = "Calibration captures cleared.";
  localStorage.removeItem("module2Calibration");
});

$("calibrateBtn").addEventListener("click", calibrateOnServer);

async function calibrateOnServer() {
  if (calibrationImages.length < 6) {
    $("calibrationStatus").textContent =
      "Capture at least 6 checkerboard images. 10 is recommended.";
    return;
  }

  $("calibrationStatus").textContent =
    "Sending checkerboard images to Python OpenCV for calibration...";
  $("calibrateBtn").disabled = true;

  try {
    const response = await fetch("/api/calibrate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        images: calibrationImages,
        checkerboard_cols: 9,
        checkerboard_rows: 6,
        square_size_cm: parseFloat($("squareSize").value),
      }),
    });

    const responseText = await response.text();

    let data;

    try {
      data = JSON.parse(responseText);
    } catch {
      throw new Error("Server response: " + responseText);
    }

    if (!response.ok) {
      throw new Error(data.detail || "Calibration failed.");
    }

    calibration = {
      rms: data.rms,
      fx: data.fx,
      fy: data.fy,
      cx: data.cx,
      cy: data.cy,
      width: data.image_width,
      height: data.image_height,
      distortion: data.distortion,
    };

    localStorage.setItem("module2Calibration", JSON.stringify(calibration));
    displayCalibration(data.valid_images);

    $("calibrationStatus").textContent =
      `Calibration complete. ${data.valid_images}/${data.total_images} checkerboard images were valid.`;
  } catch (err) {
    $("calibrationStatus").textContent = `Calibration error: ${err.message}`;
  } finally {
    $("calibrateBtn").disabled = false;
  }
}

function displayCalibration(validImages = "-") {
  if (!calibration) return;

  $("validImagesValue").textContent = validImages;
  $("rmsValue").textContent = Number(calibration.rms).toFixed(4) + " px";
  $("fxValue").textContent = Number(calibration.fx).toFixed(2);
  $("fyValue").textContent = Number(calibration.fy).toFixed(2);
  $("cxValue").textContent = Number(calibration.cx).toFixed(2);
  $("cyValue").textContent = Number(calibration.cy).toFixed(2);
  $("resolutionValue").textContent =
    `${calibration.width} × ${calibration.height}`;
  $("calibrationResults").classList.remove("hidden");
}

function loadSavedCalibration() {
  try {
    const saved = localStorage.getItem("module2Calibration");
    if (saved) {
      calibration = JSON.parse(saved);
      displayCalibration("saved");
      $("calibrationStatus").textContent =
        "Saved calibration loaded. You may recalibrate if needed.";
    }
  } catch (_) {}
}

// SHARED MEASUREMENT
function canvasPoint(event, canvas) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) * canvas.width) / rect.width,
    y: ((event.clientY - rect.top) * canvas.height) / rect.height,
  };
}

function dist(a, b) {
  return Math.hypot(b.x - a.x, b.y - a.y);
}

function drawPoints(canvas, points) {
  const ctx = canvas.getContext("2d");
  ctx.lineWidth = 4;
  ctx.strokeStyle = "#00ff66";
  ctx.fillStyle = "#00ff66";
  ctx.font = "24px Arial";

  points.forEach((p, i) => {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 8, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillText(String(i + 1), p.x + 10, p.y - 10);
  });

  if (points.length > 1) {
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++)
      ctx.lineTo(points[i].x, points[i].y);
    if (points.length === 4) ctx.closePath();
    ctx.stroke();
  }
}

function scaledIntrinsics(canvas) {
  if (!calibration) throw new Error("Complete Step 1 calibration first.");

  const sx = canvas.width / calibration.width;
  const sy = canvas.height / calibration.height;

  return {
    fx: calibration.fx * sx,
    fy: calibration.fy * sy,
  };
}

function calculateDimensions(points, z, canvas) {
  if (points.length !== 4) throw new Error("Select exactly 4 corners.");

  const [tl, tr, br, bl] = points;
  const widthPixels = (dist(tl, tr) + dist(bl, br)) / 2;
  const heightPixels = (dist(tl, bl) + dist(tr, br)) / 2;
  const intr = scaledIntrinsics(canvas);

  return {
    widthPixels,
    heightPixels,
    widthCm: ((widthPixels * z) / intr.fx) * 100,
    heightCm: ((heightPixels * z) / intr.fy) * 100,
  };
}

function restoreCanvas(canvas, dataUrl, points) {
  if (!dataUrl) return;

  const img = new Image();
  img.onload = () => {
    canvas.getContext("2d").drawImage(img, 0, 0, canvas.width, canvas.height);
    drawPoints(canvas, points);
  };
  img.src = dataUrl;
}

// STEP 2
$("captureObjectBtn").addEventListener("click", () => {
  try {
    const canvas = $("measurementCanvas");
    captureVideoToCanvas(canvas);
    measurementFrame = canvas.toDataURL("image/jpeg", 0.9);
    measurementPoints = [];
    $("measurementResults").classList.add("hidden");
    $("measurementStatus").textContent =
      "Captured. Click Top-left, Top-right, Bottom-right, Bottom-left.";
  } catch (err) {
    $("measurementStatus").textContent = err.message;
  }
});

$("measurementCanvas").addEventListener("click", (e) => {
  if (!measurementFrame || measurementPoints.length >= 4) return;
  measurementPoints.push(canvasPoint(e, $("measurementCanvas")));
  restoreCanvas($("measurementCanvas"), measurementFrame, measurementPoints);
  $("measurementStatus").textContent =
    `${measurementPoints.length}/4 corners selected.`;
});

$("resetCornersBtn").addEventListener("click", () => {
  measurementPoints = [];
  restoreCanvas($("measurementCanvas"), measurementFrame, measurementPoints);
});

$("calculateObjectBtn").addEventListener("click", () => {
  try {
    const z = parseFloat($("step2Distance").value);
    const r = calculateDimensions(measurementPoints, z, $("measurementCanvas"));

    $("pixelWidthResult").textContent = r.widthPixels.toFixed(2) + " px";
    $("pixelHeightResult").textContent = r.heightPixels.toFixed(2) + " px";
    $("widthResult").textContent = r.widthCm.toFixed(2) + " cm";
    $("heightResult").textContent = r.heightCm.toFixed(2) + " cm";
    $("measurementResults").classList.remove("hidden");
    $("measurementStatus").textContent = "Measurement calculated.";
  } catch (err) {
    $("measurementStatus").textContent = err.message;
  }
});

// STEP 3
$("validationMode").addEventListener("change", () => {
  $("manualFields").classList.toggle(
    "hidden",
    $("validationMode").value !== "manual",
  );
});

$("beginValidationBtn").addEventListener("click", () => {
  if (!calibration) {
    $("validationStatus").textContent = "Complete Step 1 calibration first.";
    return;
  }

  validationResults = [];
  validationIndex = 0;
  validationStarted = true;
  validationFrame = null;
  validationPoints = [];
  $("validationTable").querySelector("tbody").innerHTML = "";
  updateCurrentObject();
  $("validationStatus").textContent = "Validation started.";
});

function currentTruth() {
  if ($("validationMode").value === "predefined") {
    return OBJECTS[validationIndex] || null;
  }

  const name = $("manualName").value.trim();
  const width = parseFloat($("manualWidth").value);
  const height = parseFloat($("manualHeight").value);

  if (!name || !Number.isFinite(width) || !Number.isFinite(height)) return null;
  return { name, width, height };
}

function updateCurrentObject() {
  if ($("validationMode").value === "predefined") {
    const obj = OBJECTS[validationIndex];

    if (!obj) {
      $("currentObject").textContent = "All 20 predefined objects completed.";
      return;
    }

    $("currentObject").innerHTML =
      `<b>${validationIndex + 1}/20 - ${obj.name}</b><br>` +
      `Actual width: ${obj.width.toFixed(2)} cm | Actual height: ${obj.height.toFixed(2)} cm`;
  } else {
    $("currentObject").textContent =
      "Enter object name and actual dimensions above.";
  }
}

$("captureValidationBtn").addEventListener("click", () => {
  if (!validationStarted) {
    $("validationStatus").textContent = "Press Begin Validation first.";
    return;
  }

  if (!currentTruth()) {
    $("validationStatus").textContent = "Enter valid object information.";
    return;
  }

  try {
    const canvas = $("validationCanvas");
    captureVideoToCanvas(canvas);
    validationFrame = canvas.toDataURL("image/jpeg", 0.9);
    validationPoints = [];
    $("validationStatus").textContent =
      "Captured. Select Top-left, Top-right, Bottom-right, Bottom-left.";
  } catch (err) {
    $("validationStatus").textContent = err.message;
  }
});

$("validationCanvas").addEventListener("click", (e) => {
  if (!validationFrame || validationPoints.length >= 4) return;
  validationPoints.push(canvasPoint(e, $("validationCanvas")));
  restoreCanvas($("validationCanvas"), validationFrame, validationPoints);
  $("validationStatus").textContent =
    `${validationPoints.length}/4 corners selected.`;
});

$("resetValidationCornersBtn").addEventListener("click", () => {
  validationPoints = [];
  restoreCanvas($("validationCanvas"), validationFrame, validationPoints);
});

$("saveValidationBtn").addEventListener("click", () => {
  const truth = currentTruth();

  if (!truth) {
    $("validationStatus").textContent = "Object information is missing.";
    return;
  }

  try {
    const z = parseFloat($("validationDistance").value);
    const est = calculateDimensions(validationPoints, z, $("validationCanvas"));

    const widthError = est.widthCm - truth.width;
    const heightError = est.heightCm - truth.height;
    const widthAbs = Math.abs(widthError);
    const heightAbs = Math.abs(heightError);
    const widthPct = (widthAbs / truth.width) * 100;
    const heightPct = (heightAbs / truth.height) * 100;

    const row = {
      measurement: validationResults.length + 1,
      object: truth.name,
      distance: z,
      actualWidth: truth.width,
      cameraWidth: est.widthCm,
      widthError,
      widthAbs,
      widthPct,
      actualHeight: truth.height,
      cameraHeight: est.heightCm,
      heightError,
      heightAbs,
      heightPct,
    };

    validationResults.push(row);
    appendValidationRow(row);

    validationFrame = null;
    validationPoints = [];
    validationIndex++;

    if ($("validationMode").value === "manual") {
      $("manualName").value = "";
      $("manualWidth").value = "";
      $("manualHeight").value = "";
    }

    updateCurrentObject();

    $("validationStatus").textContent =
      validationIndex >= 20 && $("validationMode").value === "predefined"
        ? "All 20 measurements complete. Generate the report."
        : "Measurement saved. Continue with the next object.";
  } catch (err) {
    $("validationStatus").textContent = err.message;
  }
});

function appendValidationRow(r) {
  const tr = document.createElement("tr");
  tr.innerHTML = `
    <td>${r.measurement}</td><td>${r.object}</td>
    <td>${r.actualWidth.toFixed(2)}</td><td>${r.cameraWidth.toFixed(2)}</td>
    <td>${r.actualHeight.toFixed(2)}</td><td>${r.cameraHeight.toFixed(2)}</td>
    <td>${r.widthPct.toFixed(2)}%</td><td>${r.heightPct.toFixed(2)}%</td>
  `;
  $("validationTable").querySelector("tbody").appendChild(tr);
}

$("clearValidationBtn").addEventListener("click", () => {
  validationResults = [];
  validationIndex = 0;
  validationStarted = false;
  validationFrame = null;
  validationPoints = [];
  $("validationTable").querySelector("tbody").innerHTML = "";
  $("currentObject").textContent = "Press Begin Validation to start.";
  $("validationStatus").textContent = "Validation cleared.";
});

// REPORT
function mean(a) {
  return a.reduce((x, y) => x + y, 0) / a.length;
}
function rmse(a) {
  return Math.sqrt(mean(a.map((v) => v * v)));
}
function std(a) {
  const m = mean(a);
  return Math.sqrt(mean(a.map((v) => (v - m) ** 2)));
}

$("finishValidationBtn").addEventListener("click", generateReport);

function generateReport() {
  if (!validationResults.length) {
    $("validationStatus").textContent =
      "Complete validation measurements first.";
    return;
  }

  const we = validationResults.map((r) => r.widthError);
  const he = validationResults.map((r) => r.heightError);
  const wa = validationResults.map((r) => r.widthAbs);
  const ha = validationResults.map((r) => r.heightAbs);
  const wp = validationResults.map((r) => r.widthPct);
  const hp = validationResults.map((r) => r.heightPct);

  const wMean = mean(we),
    hMean = mean(he);
  const wMAE = mean(wa),
    hMAE = mean(ha);
  const wRMSE = rmse(we),
    hRMSE = rmse(he);
  const wMAPE = mean(wp),
    hMAPE = mean(hp);
  const wStd = std(we),
    hStd = std(he);
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
    `Width MAPE = <b>${wMAPE.toFixed(2)}%</b>, height MAPE = ` +
    `<b>${hMAPE.toFixed(2)}%</b>, overall average percentage error = ` +
    `<b>${overall.toFixed(2)}%</b>.`;

  const tbody = $("reportTable").querySelector("tbody");
  tbody.innerHTML = "";

  validationResults.forEach((r) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${r.measurement}</td><td>${r.object}</td>
      <td>${r.actualWidth.toFixed(2)} cm</td><td>${r.cameraWidth.toFixed(2)} cm</td>
      <td>${r.actualHeight.toFixed(2)} cm</td><td>${r.cameraHeight.toFixed(2)} cm</td>
    `;
    tbody.appendChild(tr);
  });

  $("reportStats").classList.remove("hidden");
  showStep("report");
}

$("downloadCsvBtn").addEventListener("click", () => {
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
      "height_percentage_error",
    ],
  ];

  validationResults.forEach((r) =>
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
      r.heightPct,
    ]),
  );

  const csv = rows
    .map((row) =>
      row.map((v) => `"${String(v).replaceAll('"', '""')}"`).join(","),
    )
    .join("\n");

  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "step3_validation_results_web.csv";
  a.click();
  URL.revokeObjectURL(url);
});

$("printReportBtn").addEventListener("click", () => window.print());

window.addEventListener("load", () => {
  loadSavedCalibration();
  $("validationMode").dispatchEvent(new Event("change"));
});
