
// ============================================================
// Module 4 - Classical Human Boundary Detection
// Browser/OpenCV.js implementation.
// The supplied Python scripts are preserved unchanged in the package.
// ============================================================

const $ = id => document.getElementById(id);

let rgbStream = null;
let rgbSource = null;           // cv.Mat
let rgbMask = null;             // GrabCut label mask
let rgbSeedMask = null;
let rgbBgModel = null;
let rgbFgModel = null;
let rgbFinalBinary = null;
let rgbMode = "rect";
let rgbRect = null;
let rgbRectStart = null;
let rgbPainting = false;

let thermalSource = null;
let thermalSmooth = null;
let thermalLabels = null;
let thermalSelectedLabels = new Set();
let thermalFinalMask = null;
let thermalObjectURLs = [];

let samUploadedMask = null;

const THERMAL_FOLDER_CANDIDATES = ["IMage1", "Image1", "image1"];

function cvOk() {
  if (!window.cvReady || typeof cv === "undefined") {
    alert("OpenCV.js is still loading. Please wait a few seconds and try again.");
    return false;
  }
  return true;
}

function deleteMat(m) {
  if (m && typeof m.delete === "function") m.delete();
}

function setCanvasSize(canvas, w, h) {
  canvas.width = w;
  canvas.height = h;
}

function matToCanvas(mat, canvas) {
  setCanvasSize(canvas, mat.cols, mat.rows);
  cv.imshow(canvas, mat);
}

function cloneToRGBA(src) {
  const out = new cv.Mat();
  if (src.channels() === 4) src.copyTo(out);
  else if (src.channels() === 3) cv.cvtColor(src, out, cv.COLOR_BGR2RGBA);
  else cv.cvtColor(src, out, cv.COLOR_GRAY2RGBA);
  return out;
}

function binaryFromGrabcut(mask) {
  const out = new cv.Mat.zeros(mask.rows, mask.cols, cv.CV_8UC1);
  for (let y = 0; y < mask.rows; y++) {
    for (let x = 0; x < mask.cols; x++) {
      const v = mask.ucharPtr(y, x)[0];
      if (v === cv.GC_FGD || v === cv.GC_PR_FGD) {
        out.ucharPtr(y, x)[0] = 255;
      }
    }
  }
  return out;
}

function largestComponent(binary) {
  const labels = new cv.Mat();
  const stats = new cv.Mat();
  const centroids = new cv.Mat();
  const n = cv.connectedComponentsWithStats(binary, labels, stats, centroids, 8, cv.CV_32S);

  if (n <= 1) {
    labels.delete(); stats.delete(); centroids.delete();
    return binary.clone();
  }

  let best = 1;
  let bestArea = stats.intAt(1, cv.CC_STAT_AREA);
  for (let i = 2; i < n; i++) {
    const area = stats.intAt(i, cv.CC_STAT_AREA);
    if (area > bestArea) {
      bestArea = area;
      best = i;
    }
  }

  const out = cv.Mat.zeros(binary.rows, binary.cols, cv.CV_8UC1);
  for (let y = 0; y < labels.rows; y++) {
    for (let x = 0; x < labels.cols; x++) {
      if (labels.intAt(y, x) === best) out.ucharPtr(y, x)[0] = 255;
    }
  }

  labels.delete(); stats.delete(); centroids.delete();
  return out;
}

function contourMetrics(binary) {
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  cv.findContours(binary, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE);

  let area = 0, perimeter = 0, points = 0;
  for (let i = 0; i < contours.size(); i++) {
    const c = contours.get(i);
    area += cv.contourArea(c);
    perimeter += cv.arcLength(c, true);
    points += c.rows;
    c.delete();
  }

  hierarchy.delete();
  return {contours, area, perimeter, points};
}

function drawBoundary(sourceRGBA, binary) {
  const out = sourceRGBA.clone();
  const info = contourMetrics(binary);
  cv.drawContours(out, info.contours, -1, new cv.Scalar(0,255,0,255), 2, cv.LINE_8);
  info.contours.delete();
  return {out, area:info.area, perimeter:info.perimeter, points:info.points};
}

function segmentedRGBA(sourceRGBA, binary) {
  const out = cv.Mat.zeros(sourceRGBA.rows, sourceRGBA.cols, sourceRGBA.type());
  sourceRGBA.copyTo(out, binary);
  return out;
}

// ============================================================
// Navigation
// ============================================================

document.querySelectorAll(".step-tab").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".step-tab").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".step-panel").forEach(p => p.classList.remove("active-panel"));
    btn.classList.add("active");
    $(btn.dataset.target).classList.add("active-panel");
  });
});

// ============================================================
// RGB camera
// ============================================================

async function enumerateCameras() {
  const sel = $("cameraSelect");
  sel.innerHTML = "";

  try {
    // Permission is often needed before labels become visible.
    const tmp = await navigator.mediaDevices.getUserMedia({video:true, audio:false});
    tmp.getTracks().forEach(t => t.stop());

    const devices = await navigator.mediaDevices.enumerateDevices();
    const videos = devices.filter(d => d.kind === "videoinput");

    videos.forEach((d, i) => {
      const opt = document.createElement("option");
      opt.value = d.deviceId;
      opt.textContent = d.label || `Camera ${i+1}`;
      sel.appendChild(opt);
    });

    const iphoneIndex = videos.findIndex(d =>
      /iphone|continuity/i.test(d.label || "")
    );
    if (iphoneIndex >= 0) sel.selectedIndex = iphoneIndex;

  } catch (e) {
    $("rgbStatus").textContent = "Camera permission error: " + e.message;
  }
}

async function startRgbCamera() {
  try {
    stopRgbCamera();
    const deviceId = $("cameraSelect").value;

    rgbStream = await navigator.mediaDevices.getUserMedia({
      video: deviceId ? {deviceId:{exact:deviceId}, width:{ideal:1920}, height:{ideal:1080}}
                      : {width:{ideal:1920}, height:{ideal:1080}},
      audio:false
    });

    $("rgbVideo").srcObject = rgbStream;
    await $("rgbVideo").play();
    $("rgbStatus").textContent = "Camera running. Click Capture.";
  } catch(e) {
    $("rgbStatus").textContent = "Could not start camera: " + e.message;
  }
}

function stopRgbCamera() {
  if (rgbStream) rgbStream.getTracks().forEach(t => t.stop());
  rgbStream = null;
  $("rgbVideo").srcObject = null;
}

function captureRgb() {
  if (!cvOk()) return;
  const video = $("rgbVideo");
  if (!video.videoWidth) {
    $("rgbStatus").textContent = "Start the camera first.";
    return;
  }

  const c = $("rgbCanvas");
  c.width = video.videoWidth;
  c.height = video.videoHeight;
  c.getContext("2d").drawImage(video, 0, 0);

  deleteMat(rgbSource);
  rgbSource = cv.imread(c);

  resetRgbSegmentation();
  $("rgbStatus").textContent = "Captured. Click Draw Rectangle, then drag tightly around the person.";
}

function resetRgbSegmentation() {
  deleteMat(rgbMask); rgbMask = null;
  deleteMat(rgbSeedMask); rgbSeedMask = null;
  deleteMat(rgbBgModel); rgbBgModel = null;
  deleteMat(rgbFgModel); rgbFgModel = null;
  deleteMat(rgbFinalBinary); rgbFinalBinary = null;
  rgbRect = null;
  rgbRectStart = null;
  rgbMode = "rect";
  if (rgbSource) {
    cv.imshow("rgbCanvas", rgbSource);
    cv.imshow("rgbPreview", rgbSource);
  }
  $("rgbArea").textContent = "-";
  $("rgbPerimeter").textContent = "-";
  $("rgbPoints").textContent = "-";
}

function canvasPoint(canvas, ev) {
  const r = canvas.getBoundingClientRect();
  return {
    x: Math.round((ev.clientX-r.left) * canvas.width / r.width),
    y: Math.round((ev.clientY-r.top) * canvas.height / r.height)
  };
}

function initializeRgbGrabcut(rect) {
  if (!rgbSource) return;

  deleteMat(rgbMask);
  deleteMat(rgbBgModel);
  deleteMat(rgbFgModel);

  rgbMask = new cv.Mat(rgbSource.rows, rgbSource.cols, cv.CV_8UC1, new cv.Scalar(cv.GC_BGD));

  // Probable foreground inside the rectangle.
  for (let y = rect.y; y < rect.y + rect.h; y++) {
    const row = rgbMask.ucharPtr(y);
    for (let x = rect.x; x < rect.x + rect.w; x++) row[x] = cv.GC_PR_FGD;
  }

  // Small sure-foreground ellipse in lower-middle, matching the intent of the Python auto-seed.
  const cx = Math.round(rect.x + rect.w * 0.5);
  const cy = Math.round(rect.y + rect.h * 0.6);
  cv.ellipse(
    rgbMask,
    new cv.Point(cx, cy),
    new cv.Size(Math.max(3, Math.round(rect.w*0.10)), Math.max(3, Math.round(rect.h*0.18))),
    0, 0, 360, new cv.Scalar(cv.GC_FGD), -1
  );

  rgbBgModel = new cv.Mat();
  rgbFgModel = new cv.Mat();

  cv.grabCut(rgbSource, rgbMask, new cv.Rect(0,0,1,1), rgbBgModel, rgbFgModel, 5, cv.GC_INIT_WITH_MASK);

  deleteMat(rgbSeedMask);
  rgbSeedMask = rgbMask.clone();
  updateRgbPreview();
}

function updateRgbPreview() {
  if (!rgbSource || !rgbMask) return;
  const b = binaryFromGrabcut(rgbMask);
  const seg = segmentedRGBA(rgbSource, b);
  const boundary = drawBoundary(rgbSource, b).out;
  cv.imshow("rgbPreview", seg);
  cv.imshow("rgbCanvas", boundary);
  b.delete(); seg.delete(); boundary.delete();
}

$("rgbCanvas").addEventListener("mousedown", ev => {
  if (!rgbSource) return;
  const p = canvasPoint($("rgbCanvas"), ev);

  if (rgbMode === "rect") {
    rgbRectStart = p;
    rgbPainting = true;
  } else if (rgbMask) {
    rgbPainting = true;
    paintRgbLabel(p.x, p.y);
  }
});

$("rgbCanvas").addEventListener("mousemove", ev => {
  if (!rgbPainting || !rgbSource) return;
  const p = canvasPoint($("rgbCanvas"), ev);

  if (rgbMode === "rect" && rgbRectStart) {
    cv.imshow("rgbCanvas", rgbSource);
    const ctx = $("rgbCanvas").getContext("2d");
    ctx.strokeStyle = "#00ff00";
    ctx.lineWidth = 4;
    ctx.strokeRect(rgbRectStart.x, rgbRectStart.y, p.x-rgbRectStart.x, p.y-rgbRectStart.y);
  } else if (rgbMask) {
    paintRgbLabel(p.x, p.y);
  }
});

window.addEventListener("mouseup", ev => {
  if (!rgbPainting) return;
  rgbPainting = false;

  if (rgbMode === "rect" && rgbRectStart && rgbSource) {
    const p = canvasPoint($("rgbCanvas"), ev);
    const x1 = Math.max(0, Math.min(rgbRectStart.x, p.x));
    const y1 = Math.max(0, Math.min(rgbRectStart.y, p.y));
    const x2 = Math.min(rgbSource.cols-1, Math.max(rgbRectStart.x, p.x));
    const y2 = Math.min(rgbSource.rows-1, Math.max(rgbRectStart.y, p.y));

    if (x2-x1 > 10 && y2-y1 > 10) {
      rgbRect = {x:x1, y:y1, w:x2-x1, h:y2-y1};
      initializeRgbGrabcut(rgbRect);
      rgbMode = "human";
      $("rgbStatus").textContent =
        "Initial GrabCut complete. Use Mark Human / Mark Background, then click Refine.";
    }
    rgbRectStart = null;
  }
});

function paintRgbLabel(x, y) {
  if (!rgbMask) return;
  const label = rgbMode === "background" ? cv.GC_BGD : cv.GC_FGD;
  cv.circle(rgbMask, new cv.Point(x,y), 12, new cv.Scalar(label), -1);
  updateRgbPreview();
}

function refineRgb() {
  if (!rgbMask) {
    $("rgbStatus").textContent = "Draw the person rectangle first.";
    return;
  }
  cv.grabCut(rgbSource, rgbMask, new cv.Rect(0,0,1,1), rgbBgModel, rgbFgModel, 5, cv.GC_INIT_WITH_MASK);
  updateRgbPreview();
  $("rgbStatus").textContent = "GrabCut refined. Continue marking or finalize.";
}

function finalizeRgb() {
  if (!rgbMask) return;

  const raw = binaryFromGrabcut(rgbMask);
  const largest = largestComponent(raw);

  // Mild 3x3 closing, matching the Python post-processing intent.
  const kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(3,3));
  const cleaned = new cv.Mat();
  cv.morphologyEx(largest, cleaned, cv.MORPH_CLOSE, kernel);

  deleteMat(rgbFinalBinary);
  rgbFinalBinary = cleaned.clone();

  const seg = segmentedRGBA(rgbSource, cleaned);
  const boundary = drawBoundary(rgbSource, cleaned);

  cv.imshow("rgbMaskCanvas", cleaned);
  cv.imshow("rgbSegmentedCanvas", seg);
  cv.imshow("rgbBoundaryCanvas", boundary.out);

  $("rgbArea").textContent = boundary.area.toFixed(1) + " px";
  $("rgbPerimeter").textContent = boundary.perimeter.toFixed(1) + " px";
  $("rgbPoints").textContent = boundary.points;

  $("rgbStatus").textContent = "RGB result finalized.";

  raw.delete(); largest.delete(); kernel.delete(); cleaned.delete();
  seg.delete(); boundary.out.delete();
}

$("rgbStartCamera").addEventListener("click", startRgbCamera);
$("rgbStopCamera").addEventListener("click", stopRgbCamera);
$("rgbCapture").addEventListener("click", captureRgb);
$("rgbReset").addEventListener("click", resetRgbSegmentation);
$("rgbRectMode").addEventListener("click", () => { rgbMode="rect"; $("rgbStatus").textContent="Drag a rectangle around the person."; });
$("rgbHumanMode").addEventListener("click", () => { rgbMode="human"; $("rgbStatus").textContent="Paint definite human regions."; });
$("rgbBackgroundMode").addEventListener("click", () => { rgbMode="background"; $("rgbStatus").textContent="Paint definite background regions."; });
$("rgbRefine").addEventListener("click", refineRgb);
$("rgbFinish").addEventListener("click", finalizeRgb);

// ============================================================
// Thermal image discovery / loading
// ============================================================

function candidateThermalNames() {
  const names = [];
  const stems = ["", "image", "Image", "thermal", "Thermal", "img", "IMG"];
  const exts = ["jpg","jpeg","png","JPG","JPEG","PNG"];
  for (let i=1; i<=5; i++) {
    for (const stem of stems) {
      for (const ext of exts) {
        names.push(`${stem}${i}.${ext}`);
      }
    }
  }
  return names;
}

function probeImage(url) {
  return new Promise(resolve => {
    const img = new Image();
    img.onload = () => resolve(url);
    img.onerror = () => resolve(null);
    img.src = url + "?cvprobe=1";
  });
}

async function discoverThermalImages() {
  const found = [];
  const names = candidateThermalNames();

  for (const folder of THERMAL_FOLDER_CANDIDATES) {
    for (const name of names) {
      if (found.length >= 5) break;
      const url = `${folder}/${name}`;
      const ok = await probeImage(url);
      if (ok && !found.includes(ok)) found.push(ok);
    }
    if (found.length >= 5) break;
  }

  populateThermalSelect(found);
}

function populateThermalSelect(items) {
  const sel = $("thermalImageSelect");
  sel.innerHTML = "";

  if (!items.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "No hosted images auto-detected - choose folder below";
    sel.appendChild(opt);
    return;
  }

  items.forEach((src, i) => {
    const opt = document.createElement("option");
    opt.value = src;
    opt.textContent = `Thermal ${i+1} - ${src.split("/").pop()}`;
    sel.appendChild(opt);
  });

  loadThermalImage(items[0]);
}

function loadThermalImage(src) {
  if (!src || !cvOk()) return;

  const img = new Image();
  img.onload = () => {
    const c = $("thermalCanvas");
    c.width = img.naturalWidth;
    c.height = img.naturalHeight;
    c.getContext("2d").drawImage(img,0,0);

    deleteMat(thermalSource);
    deleteMat(thermalSmooth);
    deleteMat(thermalLabels);
    deleteMat(thermalFinalMask);
    thermalSelectedLabels.clear();

    thermalSource = cv.imread(c);
    thermalSmooth = thermalPreprocess(thermalSource);

    const otsu = getOtsuThreshold(thermalSmooth);
    $("thermalThreshold").value = otsu;
    $("thermalThresholdValue").textContent = otsu;

    processThermal();
  };
  img.src = src;
}

$("thermalImageSelect").addEventListener("change", e => loadThermalImage(e.target.value));

$("thermalFolderInput").addEventListener("change", e => {
  thermalObjectURLs.forEach(URL.revokeObjectURL);
  thermalObjectURLs = [];

  const files = [...e.target.files].filter(f => f.type.startsWith("image/")).slice(0,5);
  const items = files.map(f => {
    const u = URL.createObjectURL(f);
    thermalObjectURLs.push(u);
    return {url:u, name:f.name};
  });

  const sel = $("thermalImageSelect");
  sel.innerHTML = "";
  items.forEach((it,i) => {
    const opt = document.createElement("option");
    opt.value = it.url;
    opt.textContent = `Thermal ${i+1} - ${it.name}`;
    sel.appendChild(opt);
  });

  if (items.length) loadThermalImage(items[0].url);
});

function thermalPreprocess(src) {
  const gray = new cv.Mat();
  const smooth = new cv.Mat();
  const norm = new cv.Mat();

  cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
  cv.GaussianBlur(gray, smooth, new cv.Size(5,5), 0, 0, cv.BORDER_DEFAULT);
  cv.normalize(smooth, norm, 0, 255, cv.NORM_MINMAX);

  gray.delete();
  smooth.delete();
  return norm;
}

function getOtsuThreshold(gray) {
  const tmp = new cv.Mat();
  const t = cv.threshold(gray, tmp, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
  tmp.delete();
  return Math.round(t);
}

function thresholdThermal(gray, threshold, invert) {
  const out = new cv.Mat();
  cv.threshold(gray, out, threshold, 255, invert ? cv.THRESH_BINARY_INV : cv.THRESH_BINARY);
  return out;
}

function cleanupThermal(mask, closeSize) {
  const k3 = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(3,3));
  const opened = new cv.Mat();
  cv.morphologyEx(mask, opened, cv.MORPH_OPEN, k3);
  k3.delete();

  if (closeSize <= 0) return opened;

  const size = 2*closeSize+1;
  const k = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(size,size));
  const closed = new cv.Mat();
  cv.morphologyEx(opened, closed, cv.MORPH_CLOSE, k);
  opened.delete(); k.delete();
  return closed;
}

function fillExternalContours(mask) {
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  cv.findContours(mask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE);
  const filled = cv.Mat.zeros(mask.rows, mask.cols, cv.CV_8UC1);
  cv.drawContours(filled, contours, -1, new cv.Scalar(255), -1);
  contours.delete(); hierarchy.delete();
  return filled;
}

function selectedThermalComponents(mask) {
  deleteMat(thermalLabels);
  thermalLabels = new cv.Mat();
  const stats = new cv.Mat();
  const centroids = new cv.Mat();
  const n = cv.connectedComponentsWithStats(mask, thermalLabels, stats, centroids, 8, cv.CV_32S);

  let keep = new Set([...thermalSelectedLabels].filter(v => v > 0 && v < n));

  if (!keep.size && n > 1) {
    let best = 1;
    let bestArea = stats.intAt(1, cv.CC_STAT_AREA);
    for (let i=2; i<n; i++) {
      const a = stats.intAt(i, cv.CC_STAT_AREA);
      if (a > bestArea) {bestArea=a; best=i;}
    }
    keep.add(best);
  }

  const out = cv.Mat.zeros(mask.rows, mask.cols, cv.CV_8UC1);
  for (let y=0; y<thermalLabels.rows; y++) {
    for (let x=0; x<thermalLabels.cols; x++) {
      if (keep.has(thermalLabels.intAt(y,x))) out.ucharPtr(y,x)[0]=255;
    }
  }

  stats.delete(); centroids.delete();
  return out;
}

function thermalGrabcutRefine(source, binary) {
  const maxDim = Math.max(binary.rows, binary.cols);
  const band = Math.max(3, Math.floor(0.01*maxDim));
  const size = 2*band+1;
  const k = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(size,size));

  const sureFg = new cv.Mat();
  const maybe = new cv.Mat();
  cv.erode(binary, sureFg, k);
  cv.dilate(binary, maybe, k);

  const gc = new cv.Mat(binary.rows, binary.cols, cv.CV_8UC1, new cv.Scalar(cv.GC_BGD));
  for (let y=0; y<binary.rows; y++) {
    for (let x=0; x<binary.cols; x++) {
      let v = cv.GC_BGD;
      if (maybe.ucharPtr(y,x)[0] > 0) v = cv.GC_PR_BGD;
      if (binary.ucharPtr(y,x)[0] > 0) v = cv.GC_PR_FGD;
      if (sureFg.ucharPtr(y,x)[0] > 0) v = cv.GC_FGD;
      gc.ucharPtr(y,x)[0] = v;
    }
  }

  const bg = new cv.Mat(), fg = new cv.Mat();
  try {
    cv.grabCut(source, gc, new cv.Rect(0,0,1,1), bg, fg, 5, cv.GC_INIT_WITH_MASK);
    const result = binaryFromGrabcut(gc);
    k.delete(); sureFg.delete(); maybe.delete(); gc.delete(); bg.delete(); fg.delete();
    return result;
  } catch(e) {
    k.delete(); sureFg.delete(); maybe.delete(); gc.delete(); bg.delete(); fg.delete();
    return binary.clone();
  }
}

function processThermal() {
  if (!thermalSource || !thermalSmooth || !cvOk()) return;

  const thr = parseInt($("thermalThreshold").value,10);
  const close = parseInt($("thermalClose").value,10);
  const invert = $("thermalInvert").checked;
  const refine = $("thermalGrabcut").checked;

  $("thermalThresholdValue").textContent = thr;
  $("thermalCloseValue").textContent = close;

  const t = thresholdThermal(thermalSmooth, thr, invert);
  const cleaned = cleanupThermal(t, close);
  const filled = fillExternalContours(cleaned);
  let selected = selectedThermalComponents(filled);

  if (refine) {
    const r = thermalGrabcutRefine(thermalSource, selected);
    selected.delete();
    selected = fillExternalContours(r);
    r.delete();
  }

  deleteMat(thermalFinalMask);
  thermalFinalMask = selected.clone();

  const seg = segmentedRGBA(thermalSource, selected);
  const boundary = drawBoundary(thermalSource, selected);

  // Left panel = source with candidate contours yellow + selected green
  const left = thermalSource.clone();

  const c1 = new cv.MatVector(), h1 = new cv.Mat();
  cv.findContours(filled, c1, h1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE);
  cv.drawContours(left, c1, -1, new cv.Scalar(255,255,0,255), 1);

  const c2 = new cv.MatVector(), h2 = new cv.Mat();
  cv.findContours(selected, c2, h2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE);
  cv.drawContours(left, c2, -1, new cv.Scalar(0,255,0,255), 2);

  cv.imshow("thermalCanvas", left);
  cv.imshow("thermalPreview", seg);
  cv.imshow("thermalMaskCanvas", selected);
  cv.imshow("thermalSegmentedCanvas", seg);
  cv.imshow("thermalBoundaryCanvas", boundary.out);

  $("thermalArea").textContent = boundary.area.toFixed(1) + " px";
  $("thermalPerimeter").textContent = boundary.perimeter.toFixed(1) + " px";
  $("thermalPoints").textContent = boundary.points;
  $("thermalStatus").textContent =
    `Processed. threshold=${thr}, close=${close}, invert=${invert}, GrabCut=${refine}.`;

  t.delete(); cleaned.delete(); filled.delete(); selected.delete();
  seg.delete(); boundary.out.delete();
  left.delete(); c1.delete(); h1.delete(); c2.delete(); h2.delete();
}

$("thermalThreshold").addEventListener("input", () => {
  $("thermalThresholdValue").textContent = $("thermalThreshold").value;
});
$("thermalClose").addEventListener("input", () => {
  $("thermalCloseValue").textContent = $("thermalClose").value;
});

$("thermalRun").addEventListener("click", processThermal);
$("thermalAuto").addEventListener("click", () => {
  if (!thermalSmooth) return;
  const otsu = getOtsuThreshold(thermalSmooth);
  $("thermalThreshold").value = otsu;
  $("thermalThresholdValue").textContent = otsu;
  thermalSelectedLabels.clear();
  $("thermalGrabcut").checked = true;
  processThermal();
});
$("thermalClearSelection").addEventListener("click", () => {
  thermalSelectedLabels.clear();
  processThermal();
});

$("thermalCanvas").addEventListener("click", ev => {
  if (!thermalLabels) return;
  const p = canvasPoint($("thermalCanvas"), ev);
  if (p.x < 0 || p.y < 0 || p.x >= thermalLabels.cols || p.y >= thermalLabels.rows) return;
  const lab = thermalLabels.intAt(p.y,p.x);
  if (lab <= 0) return;
  if (thermalSelectedLabels.has(lab)) thermalSelectedLabels.delete(lab);
  else thermalSelectedLabels.add(lab);
  processThermal();
});

// ============================================================
// SAM2 comparison
// ============================================================

$("samMaskInput").addEventListener("change", e => {
  const f = e.target.files[0];
  if (!f || !cvOk()) return;

  const img = new Image();
  img.onload = () => {
    const c = $("samUploadedCanvas");
    c.width = img.naturalWidth;
    c.height = img.naturalHeight;
    c.getContext("2d").drawImage(img,0,0);

    deleteMat(samUploadedMask);
    const src = cv.imread(c);
    const gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
    samUploadedMask = new cv.Mat();
    cv.threshold(gray, samUploadedMask, 127, 255, cv.THRESH_BINARY);
    src.delete(); gray.delete();
  };
  img.src = URL.createObjectURL(f);
});

function resizeMaskNearest(mask, w, h) {
  const out = new cv.Mat();
  cv.resize(mask, out, new cv.Size(w,h), 0, 0, cv.INTER_NEAREST);
  return out;
}

function compareMasks() {
  if (!samUploadedMask) {
    alert("Upload a SAM2 mask first.");
    return;
  }

  const classical = $("samTarget").value === "rgb" ? rgbFinalBinary : thermalFinalMask;
  if (!classical) {
    alert("Finalize the selected classical segmentation first.");
    return;
  }

  const sam = resizeMaskNearest(samUploadedMask, classical.cols, classical.rows);

  let tp=0, fp=0, fn=0, union=0, inter=0;
  const diff = cv.Mat.zeros(classical.rows, classical.cols, cv.CV_8UC4);

  for (let y=0; y<classical.rows; y++) {
    for (let x=0; x<classical.cols; x++) {
      const a = classical.ucharPtr(y,x)[0] > 0;
      const b = sam.ucharPtr(y,x)[0] > 0;

      if (a && b) {tp++; inter++;}
      if (a && !b) fp++;
      if (!a && b) fn++;
      if (a || b) union++;

      const px = diff.ucharPtr(y,x);
      if (a && b) {px[0]=0; px[1]=255; px[2]=0; px[3]=255;}
      else if (a) {px[0]=255; px[1]=0; px[2]=0; px[3]=255;}
      else if (b) {px[0]=0; px[1]=0; px[2]=255; px[3]=255;}
      else {px[0]=0; px[1]=0; px[2]=0; px[3]=255;}
    }
  }

  const iou = union ? inter/union : 1;
  const dice = (2*tp+fp+fn) ? (2*tp)/(2*tp+fp+fn) : 1;
  const precision = (tp+fp) ? tp/(tp+fp) : 1;
  const recall = (tp+fn) ? tp/(tp+fn) : 1;

  cv.imshow("samClassicalCanvas", classical);
  cv.imshow("samUploadedCanvas", sam);
  cv.imshow("samDiffCanvas", diff);

  $("samIoU").textContent = iou.toFixed(4);
  $("samDice").textContent = dice.toFixed(4);
  $("samPrecision").textContent = precision.toFixed(4);
  $("samRecall").textContent = recall.toFixed(4);

  sam.delete(); diff.delete();
}

$("compareSam").addEventListener("click", compareMasks);

// ============================================================
// Startup
// ============================================================

window.addEventListener("load", async () => {
  if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
    await enumerateCameras();
  }
  // Wait a little for OpenCV.js before probing hosted thermal images.
  const wait = setInterval(() => {
    if (window.cvReady) {
      clearInterval(wait);
      discoverThermalImages();
    }
  }, 250);
});
