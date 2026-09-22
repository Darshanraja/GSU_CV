// ============================================================
// Module 4 - Classical Human Boundary Detection
// Browser/OpenCV.js implementation.
// The supplied Python scripts are preserved unchanged in the package.
// ============================================================

const $ = (id) => document.getElementById(id);

let rgbStream = null;
let rgbSource = null; // cv.Mat
let rgbMask = null; // GrabCut label mask
let rgbSeedMask = null;
let rgbBgModel = null;
let rgbFgModel = null;
let rgbFinalMasks = { body: null, face: null }; // finished classical masks, one per target
let rgbOriginalSaved = false;
let rgbMode = "rect";
let rgbRect = null;
let rgbRectStart = null;
let rgbPainting = false;
let rgbSource3 = null; // cv.Mat, 3-channel RGB copy (cv.grabCut needs 8UC3, NOT RGBA)
let rgbBusy = false;
let rgbStrokes = []; // brush strokes, only used to draw the green/red overlay

// Working resolution for the RGB pipeline. GrabCut in WebAssembly is much slower than the
// native Python version, so big camera frames are scaled down (long side) before segmenting.
const RGB_MAX_SIDE = 1000;
const RGB_BRUSH = 12;

let thermalStream = null;
let thermalSource = null; // RGBA cv.Mat (working size)
let thermalSmooth = null; // grayscale, blurred, contrast-stretched
let thermalCandidates = null; // binary mask of all warm regions
let thermalLabels = null; // connected-component labels of the warm regions
let thermalLabelCount = 0;
let thermalSelectedLabels = new Set();
let thermalMask = null; // current human mask
let thermalFinalMask = null; // copy used by the SAM2 comparison
let thermalBusy = false;

let samUploadedMask = null;

function cvOk() {
  if (!window.cvReady || typeof cv === "undefined") {
    alert(
      "OpenCV.js is still loading. Please wait a few seconds and try again.",
    );
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

// >>> RGB CORE BEGIN
// cv.imread() gives RGBA, but cv.grabCut() only accepts 3-channel 8-bit images.
function rgbaToRgb(src) {
  const out = new cv.Mat();
  cv.cvtColor(src, out, cv.COLOR_RGBA2RGB);
  return out;
}

function binaryFromGrabcut(mask) {
  const out = cv.Mat.zeros(mask.rows, mask.cols, cv.CV_8UC1);
  const m = mask.data,
    o = out.data;
  for (let i = 0; i < m.length; i++) {
    const v = m[i];
    if (v === cv.GC_FGD || v === cv.GC_PR_FGD) o[i] = 255;
  }
  return out;
}

function largestComponent(binary) {
  const labels = new cv.Mat();
  const stats = new cv.Mat();
  const centroids = new cv.Mat();
  const n = cv.connectedComponentsWithStats(
    binary,
    labels,
    stats,
    centroids,
    8,
    cv.CV_32S,
  );

  if (n <= 1) {
    labels.delete();
    stats.delete();
    centroids.delete();
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
  const lab = labels.data32S,
    o = out.data;
  for (let i = 0; i < lab.length; i++) if (lab[i] === best) o[i] = 255;

  labels.delete();
  stats.delete();
  centroids.delete();
  return out;
}

// Port of build_initial_mask() from rgb_human_boundary.py, with two modes.
//
// mode "body" (rectangle around the WHOLE person)
//   1. outside the box                        -> sure background
//   2. learn the background colour (Lab mean + covariance) from the two TOP CORNERS and the
//      upper side edges of the box (top-centre is skipped: that is where the head is)
//   3. pixels in the box close to that colour (Mahalanobis distance) -> probable background,
//      everything else -> probable foreground
//   4. background-coloured pixels touching top corners / upper sides  -> sure background
//   5. torso prior: lower-middle of the box is never "background-coloured"
//   6. small ellipse in the middle of the box -> sure foreground
//
// mode "face" (rectangle around the HEAD only)
//   A tight head box is mostly hair/face, so the background is ONLY in the corners of the box:
//   sample the two top corners only, no side strips, no torso prior, core ellipse on the face.
function buildRgbSeedMask(
  rgb3,
  rect,
  mode = "body",
  thresh = 9.0,
  edgeFrac = 0.1,
) {
  const face = mode === "face";
  const H = rgb3.rows,
    W = rgb3.cols;
  const x0 = Math.max(0, rect.x),
    y0 = Math.max(0, rect.y);
  const rw = Math.min(rect.w, W - x0),
    rh = Math.min(rect.h, H - y0);

  const mask = new cv.Mat(H, W, cv.CV_8UC1, new cv.Scalar(cv.GC_BGD));

  const blur = new cv.Mat();
  const lab = new cv.Mat();
  cv.GaussianBlur(rgb3, blur, new cv.Size(5, 5), 0, 0, cv.BORDER_DEFAULT);
  cv.cvtColor(blur, lab, cv.COLOR_RGB2Lab);
  const L = lab.data; // interleaved L,a,b bytes
  const M = mask.data;

  const bw = Math.max(5, Math.floor(rw * edgeFrac));
  const bh = Math.max(5, Math.floor(rh * edgeFrac));
  const cw = face
    ? Math.max(5, Math.floor(rw * 0.1))
    : Math.max(1, Math.floor(rw * 0.25));
  const ch = face ? Math.max(5, Math.floor(rh * 0.08)) : bh * 2; // corner-sample height
  const sideH = Math.floor(rh * 0.5);

  // background sample regions, in box coordinates: [row0,row1,col0,col1]
  const regions = face
    ? [
        [0, Math.min(rh, ch), 0, Math.min(rw, cw)],
        [0, Math.min(rh, ch), Math.max(0, rw - cw), rw],
      ]
    : [
        [0, Math.min(rh, ch), 0, Math.min(rw, cw)],
        [0, Math.min(rh, ch), Math.max(0, rw - cw), rw],
        [0, Math.min(rh, sideH), 0, Math.min(rw, bw)],
        [0, Math.min(rh, sideH), Math.max(0, rw - bw), rw],
      ];

  // mean + (unbiased) covariance of the sampled Lab values
  let n = 0;
  const s = [0, 0, 0];
  const ss = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (const [r0, r1, c0, c1] of regions) {
    for (let r = r0; r < r1; r++) {
      for (let c = c0; c < c1; c++) {
        const i = ((y0 + r) * W + (x0 + c)) * 3;
        const v = [L[i], L[i + 1], L[i + 2]];
        n++;
        for (let a = 0; a < 3; a++) {
          s[a] += v[a];
          for (let b = 0; b < 3; b++) ss[a][b] += v[a] * v[b];
        }
      }
    }
  }
  const mean = s.map((v) => v / n);
  const cov = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let a = 0; a < 3; a++) {
    for (let b = 0; b < 3; b++) {
      cov[a][b] = (ss[a][b] - n * mean[a] * mean[b]) / Math.max(1, n - 1);
    }
    cov[a][a] += 4.0; // regularisation, same as the Python version
  }

  // inverse of a 3x3 matrix by cofactors
  const [[a, b, c], [d, e, f], [g, h, k]] = cov;
  const det = a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g);
  const inv = [
    [(e * k - f * h) / det, (c * h - b * k) / det, (b * f - c * e) / det],
    [(f * g - d * k) / det, (a * k - c * g) / det, (c * d - a * f) / det],
    [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det],
  ];

  const torsoR0 = Math.floor(rh * 0.65);
  const torsoC0 = Math.floor(rw * 0.15);
  const torsoC1 = Math.floor(rw * 0.85);

  for (let r = 0; r < rh; r++) {
    for (let c = 0; c < rw; c++) {
      const i = ((y0 + r) * W + (x0 + c)) * 3;
      const dx = L[i] - mean[0],
        dy = L[i + 1] - mean[1],
        dz = L[i + 2] - mean[2];
      const maha2 =
        inv[0][0] * dx * dx +
        inv[1][1] * dy * dy +
        inv[2][2] * dz * dz +
        2 * (inv[0][1] * dx * dy + inv[0][2] * dx * dz + inv[1][2] * dy * dz);

      let likelyBg = maha2 < thresh;
      if (!face && r >= torsoR0 && c >= torsoC0 && c < torsoC1)
        likelyBg = false;

      let v = likelyBg ? cv.GC_PR_BGD : cv.GC_PR_FGD;
      const inSureBg = face
        ? (r < ch && c < cw) || (r < ch && c >= rw - cw)
        : (r < bh && c < cw) ||
          (r < bh && c >= rw - cw) ||
          (r < sideH && c < bw) ||
          (r < sideH && c >= rw - bw);
      if (likelyBg && inSureBg) v = cv.GC_BGD;
      M[(y0 + r) * W + (x0 + c)] = v;
    }
  }

  // sure-foreground core in the middle of the box
  const coreY = face ? 0.55 : 0.6;
  const coreW = face ? 0.12 : 0.1;
  cv.ellipse(
    mask,
    new cv.Point(x0 + Math.floor(rw / 2), y0 + Math.floor(rh * coreY)),
    new cv.Size(
      Math.max(3, Math.floor(rw * coreW)),
      Math.max(3, Math.floor(rh * 0.18)),
    ),
    0,
    0,
    360,
    new cv.Scalar(cv.GC_FGD),
    -1,
  );

  blur.delete();
  lab.delete();
  return mask;
}
// <<< RGB CORE END

function contourMetrics(binary) {
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  cv.findContours(
    binary,
    contours,
    hierarchy,
    cv.RETR_EXTERNAL,
    cv.CHAIN_APPROX_NONE,
  );

  let area = 0,
    perimeter = 0,
    points = 0;
  for (let i = 0; i < contours.size(); i++) {
    const c = contours.get(i);
    area += cv.contourArea(c);
    perimeter += cv.arcLength(c, true);
    points += c.rows;
    c.delete();
  }

  hierarchy.delete();
  return { contours, area, perimeter, points };
}

function drawBoundary(sourceRGBA, binary) {
  const out = sourceRGBA.clone();
  const info = contourMetrics(binary);
  cv.drawContours(
    out,
    info.contours,
    -1,
    new cv.Scalar(0, 255, 0, 255),
    2,
    cv.LINE_8,
  );
  info.contours.delete();
  return {
    out,
    area: info.area,
    perimeter: info.perimeter,
    points: info.points,
  };
}

function segmentedRGBA(sourceRGBA, binary) {
  const out = cv.Mat.zeros(sourceRGBA.rows, sourceRGBA.cols, sourceRGBA.type());
  sourceRGBA.copyTo(out, binary);
  return out;
}

// ============================================================
// Navigation
// ============================================================

document.querySelectorAll(".step-tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document
      .querySelectorAll(".step-tab")
      .forEach((b) => b.classList.remove("active"));
    document
      .querySelectorAll(".step-panel")
      .forEach((p) => p.classList.remove("active-panel"));
    btn.classList.add("active");
    $(btn.dataset.target).classList.add("active-panel");
  });
});

// ============================================================
// RGB camera
// ============================================================

async function enumerateCameras() {
  const rgbSel = $("cameraSelect");
  const thSel = $("thermalCameraSelect");
  rgbSel.innerHTML = "";
  thSel.innerHTML = "";

  try {
    // Permission is often needed before labels become visible.
    const tmp = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false,
    });
    tmp.getTracks().forEach((t) => t.stop());

    const devices = await navigator.mediaDevices.enumerateDevices();
    const videos = devices.filter((d) => d.kind === "videoinput");

    videos.forEach((d, i) => {
      for (const sel of [rgbSel, thSel]) {
        const opt = document.createElement("option");
        opt.value = d.deviceId;
        opt.textContent = d.label || `Camera ${i + 1}`;
        sel.appendChild(opt);
      }
    });

    const iphoneIndex = videos.findIndex((d) =>
      /iphone|continuity/i.test(d.label || ""),
    );
    if (iphoneIndex >= 0) rgbSel.selectedIndex = iphoneIndex;
  } catch (e) {
    $("rgbStatus").textContent = "Camera permission error: " + e.message;
    $("thermalStatus").textContent = "Camera permission error: " + e.message;
  }
}

async function startRgbCamera() {
  try {
    stopThermalCamera();
    stopRgbCamera();
    const deviceId = $("cameraSelect").value;

    rgbStream = await navigator.mediaDevices.getUserMedia({
      video: deviceId
        ? {
            deviceId: { exact: deviceId },
            width: { ideal: 1920 },
            height: { ideal: 1080 },
          }
        : { width: { ideal: 1920 }, height: { ideal: 1080 } },
      audio: false,
    });

    $("rgbVideo").srcObject = rgbStream;
    await $("rgbVideo").play();
    $("rgbStatus").textContent = "Camera running. Click Capture.";
  } catch (e) {
    $("rgbStatus").textContent = "Could not start camera: " + e.message;
  }
}

function stopRgbCamera() {
  if (rgbStream) rgbStream.getTracks().forEach((t) => t.stop());
  rgbStream = null;
  $("rgbVideo").srcObject = null;
}

// Read the (full-resolution) canvas, scale it down if needed, and make it the RGB source.
function setRgbSourceFromCanvas(canvas) {
  let full = cv.imread(canvas);
  let work = full;
  const maxSide = Math.max(full.cols, full.rows);
  if (maxSide > RGB_MAX_SIDE) {
    const s = RGB_MAX_SIDE / maxSide;
    work = new cv.Mat();
    cv.resize(
      full,
      work,
      new cv.Size(Math.round(full.cols * s), Math.round(full.rows * s)),
      0,
      0,
      cv.INTER_AREA,
    );
    full.delete();
  }

  deleteMat(rgbSource);
  deleteMat(rgbSource3);
  rgbSource = work;
  rgbSource3 = rgbaToRgb(work);
  clearRgbResults();
  resetRgbSegmentation();
  return `${work.cols}x${work.rows}`;
}

function loadRgbUploadedImage(file) {
  if (!file || !cvOk()) return;

  stopRgbCamera();

  const img = new Image();
  const objectUrl = URL.createObjectURL(file);

  img.onload = () => {
    const c = $("rgbCanvas");
    c.width = img.naturalWidth;
    c.height = img.naturalHeight;
    c.getContext("2d").drawImage(img, 0, 0);

    const size = setRgbSourceFromCanvas(c);
    $("rgbStatus").textContent =
      `RGB image loaded (working size ${size}). Click Draw Rectangle, then drag tightly around the person.`;

    URL.revokeObjectURL(objectUrl);
  };

  img.onerror = () => {
    $("rgbStatus").textContent = "Could not load the selected RGB image.";
    URL.revokeObjectURL(objectUrl);
  };

  img.src = objectUrl;
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

  const size = setRgbSourceFromCanvas(c);
  $("rgbStatus").textContent =
    `Captured (working size ${size}). Click Draw Rectangle, then drag tightly around the person.`;
}

// Which thing is being segmented right now: "body" (whole person) or "face" (head).
function rgbTarget() {
  return $("rgbTarget").value === "face" ? "face" : "body";
}

// A new image was loaded: forget the finished results of BOTH targets.
function clearRgbResults() {
  deleteMat(rgbFinalMasks.body);
  rgbFinalMasks.body = null;
  deleteMat(rgbFinalMasks.face);
  rgbFinalMasks.face = null;
  rgbOriginalSaved = false;
  [
    "rgbMaskCanvas",
    "rgbSegmentedCanvas",
    "rgbBoundaryCanvas",
    "rgbFaceMaskCanvas",
    "rgbFaceSegmentedCanvas",
    "rgbFaceBoundaryCanvas",
  ].forEach((id) => {
    const c = $(id);
    c.getContext("2d").clearRect(0, 0, c.width, c.height);
  });
  [
    "rgbArea",
    "rgbPerimeter",
    "rgbPoints",
    "rgbFaceArea",
    "rgbFacePerimeter",
    "rgbFacePoints",
  ].forEach((id) => {
    $(id).textContent = "-";
  });
}

function resetRgbSegmentation() {
  deleteMat(rgbMask);
  rgbMask = null;
  deleteMat(rgbSeedMask);
  rgbSeedMask = null;
  deleteMat(rgbBgModel);
  rgbBgModel = null;
  deleteMat(rgbFgModel);
  rgbFgModel = null;
  rgbRect = null;
  rgbRectStart = null;
  rgbStrokes = [];
  rgbMode = "rect";
  if (rgbSource) {
    cv.imshow("rgbCanvas", rgbSource);
    cv.imshow("rgbPreview", rgbSource);
  }
}

function canvasPoint(canvas, ev) {
  const r = canvas.getBoundingClientRect();
  return {
    x: Math.round(((ev.clientX - r.left) * canvas.width) / r.width),
    y: Math.round(((ev.clientY - r.top) * canvas.height) / r.height),
  };
}

const nextTick = () => new Promise((r) => setTimeout(r, 30));

async function initializeRgbGrabcut(rect) {
  if (!rgbSource3 || rgbBusy) return;
  rgbBusy = true;
  $("rgbStatus").textContent =
    `Auto-seeding (${rgbTarget() === "face" ? "face" : "whole body"}) and running GrabCut... (a few seconds)`;
  await nextTick(); // let the browser repaint the status text

  try {
    deleteMat(rgbMask);
    deleteMat(rgbBgModel);
    deleteMat(rgbFgModel);
    deleteMat(rgbSeedMask);
    rgbStrokes = [];

    // NOT "whole rectangle = foreground": seed from the box's own colour statistics.
    rgbMask = buildRgbSeedMask(rgbSource3, rect, rgbTarget());
    rgbSeedMask = rgbMask.clone();

    rgbBgModel = new cv.Mat();
    rgbFgModel = new cv.Mat();
    cv.grabCut(
      rgbSource3,
      rgbMask,
      new cv.Rect(0, 0, 1, 1),
      rgbBgModel,
      rgbFgModel,
      5,
      cv.GC_INIT_WITH_MASK,
    );

    updateRgbPreview();
    $("rgbStatus").textContent =
      "Initial GrabCut complete. Use Mark Human / Mark Background on any mistakes, then click Refine.";
  } catch (e) {
    console.error("GrabCut failed:", e);
    $("rgbStatus").textContent =
      "GrabCut failed - see the browser console. Try a different rectangle.";
  } finally {
    rgbBusy = false;
  }
}

function updateRgbPreview() {
  if (!rgbSource || !rgbMask) return;
  const b = binaryFromGrabcut(rgbMask);
  const seg = segmentedRGBA(rgbSource, b);
  const boundary = drawBoundary(rgbSource, b).out;
  cv.imshow("rgbPreview", seg);
  cv.imshow("rgbCanvas", boundary);

  // green = marked human, red = marked background (like the dots in the Python window)
  const ctx = $("rgbCanvas").getContext("2d");
  for (const s of rgbStrokes) {
    ctx.fillStyle =
      s.label === cv.GC_FGD ? "rgba(0,255,0,0.55)" : "rgba(255,0,0,0.55)";
    ctx.beginPath();
    ctx.arc(s.x, s.y, RGB_BRUSH, 0, 2 * Math.PI);
    ctx.fill();
  }

  b.delete();
  seg.delete();
  boundary.delete();
}

$("rgbCanvas").addEventListener("mousedown", (ev) => {
  if (!rgbSource || rgbBusy) return;
  ev.preventDefault();
  const p = canvasPoint($("rgbCanvas"), ev);

  if (rgbMode === "rect") {
    rgbRectStart = p;
    rgbPainting = true;
  } else if (rgbMask) {
    rgbPainting = true;
    paintRgbLabel(p.x, p.y);
  }
});

$("rgbCanvas").addEventListener("mousemove", (ev) => {
  if (!rgbPainting || !rgbSource) return;
  const p = canvasPoint($("rgbCanvas"), ev);

  if (rgbMode === "rect" && rgbRectStart) {
    cv.imshow("rgbCanvas", rgbSource);
    const ctx = $("rgbCanvas").getContext("2d");
    ctx.strokeStyle = "#00ff00";
    ctx.lineWidth = 4;
    ctx.strokeRect(
      rgbRectStart.x,
      rgbRectStart.y,
      p.x - rgbRectStart.x,
      p.y - rgbRectStart.y,
    );
  } else if (rgbMask) {
    paintRgbLabel(p.x, p.y);
  }
});

window.addEventListener("mouseup", (ev) => {
  if (!rgbPainting) return;
  rgbPainting = false;

  if (rgbMode === "rect" && rgbRectStart && rgbSource) {
    const p = canvasPoint($("rgbCanvas"), ev);
    const x1 = Math.max(0, Math.min(rgbRectStart.x, p.x));
    const y1 = Math.max(0, Math.min(rgbRectStart.y, p.y));
    const x2 = Math.min(rgbSource.cols - 1, Math.max(rgbRectStart.x, p.x));
    const y2 = Math.min(rgbSource.rows - 1, Math.max(rgbRectStart.y, p.y));

    if (x2 - x1 > 10 && y2 - y1 > 10) {
      rgbRect = { x: x1, y: y1, w: x2 - x1, h: y2 - y1 };
      rgbMode = "human";
      initializeRgbGrabcut(rgbRect);
    }
    rgbRectStart = null;
  }
});

function paintRgbLabel(x, y) {
  if (!rgbMask) return;
  const label = rgbMode === "background" ? cv.GC_BGD : cv.GC_FGD;
  cv.circle(rgbMask, new cv.Point(x, y), RGB_BRUSH, new cv.Scalar(label), -1);

  const last = rgbStrokes[rgbStrokes.length - 1];
  if (!last || last.label !== label || Math.hypot(last.x - x, last.y - y) > 4) {
    rgbStrokes.push({ x, y, label });
  }
  updateRgbPreview();
}

async function refineRgb() {
  if (!rgbMask) {
    $("rgbStatus").textContent = "Draw the person rectangle first.";
    return;
  }
  if (rgbBusy) return;
  rgbBusy = true;
  $("rgbStatus").textContent = "Refining with GrabCut...";
  await nextTick();

  try {
    cv.grabCut(
      rgbSource3,
      rgbMask,
      new cv.Rect(0, 0, 1, 1),
      rgbBgModel,
      rgbFgModel,
      5,
      cv.GC_INIT_WITH_MASK,
    );
    updateRgbPreview();
    $("rgbStatus").textContent =
      "GrabCut refined. Continue marking or click Finalize.";
  } catch (e) {
    console.error("GrabCut failed:", e);
    $("rgbStatus").textContent = "GrabCut failed - see the browser console.";
  } finally {
    rgbBusy = false;
  }
}

function finalizeRgb() {
  if (!rgbMask) {
    $("rgbStatus").textContent = "Draw the rectangle first.";
    return;
  }

  const target = rgbTarget(); // "body" or "face"
  const P = target === "face" ? "rgbFace" : "rgb"; // element-id prefix ("rgb..." = whole body)

  const raw = binaryFromGrabcut(rgbMask);
  const largest = largestComponent(raw);

  // Mild 3x3 closing, then fill interior holes (same as clean_mask() in the Python script).
  const kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(3, 3));
  const closed = new cv.Mat();
  cv.morphologyEx(largest, closed, cv.MORPH_CLOSE, kernel);
  const cleaned = fillExternalContours(closed);

  deleteMat(rgbFinalMasks[target]);
  rgbFinalMasks[target] = cleaned.clone();

  const seg = segmentedRGBA(rgbSource, cleaned);
  const boundary = drawBoundary(rgbSource, cleaned);

  cv.imshow(P + "MaskCanvas", cleaned);
  cv.imshow(P + "SegmentedCanvas", seg);
  cv.imshow(P + "BoundaryCanvas", boundary.out);

  $(P + "Area").textContent = boundary.area.toFixed(1) + " px";
  $(P + "Perimeter").textContent = boundary.perimeter.toFixed(1) + " px";
  $(P + "Points").textContent = boundary.points;

  // Download the same files the Python script writes (the original only once per image).
  const files = [];
  if (!rgbOriginalSaved) {
    const original = document.createElement("canvas");
    cv.imshow(original, rgbSource);
    files.push(["rgb_original.png", original]);
    rgbOriginalSaved = true;
  }
  files.push([`rgb_${target}_mask.png`, $(P + "MaskCanvas")]);
  files.push([`rgb_${target}_segmented.png`, $(P + "SegmentedCanvas")]);
  files.push([`rgb_${target}_boundary.png`, $(P + "BoundaryCanvas")]);
  files.forEach(([name, canvas], i) =>
    setTimeout(() => downloadCanvas(canvas, name), i * 300),
  );

  $("rgbStatus").textContent =
    `${target === "face" ? "Face" : "Whole-body"} result finalized and saved (rgb_${target}_mask.png ...). ` +
    `Switch "Segment" to ${target === "face" ? "Whole body" : "Face"} to do the other one on the same image.`;

  raw.delete();
  largest.delete();
  kernel.delete();
  closed.delete();
  cleaned.delete();
  seg.delete();
  boundary.out.delete();
}

$("rgbStartCamera").addEventListener("click", startRgbCamera);
$("rgbStopCamera").addEventListener("click", stopRgbCamera);
$("rgbCapture").addEventListener("click", captureRgb);
$("rgbImageInput").addEventListener("change", (e) => {
  const file = e.target.files && e.target.files[0];
  if (file) loadRgbUploadedImage(file);
});
$("rgbReset").addEventListener("click", resetRgbSegmentation);
$("rgbRectMode").addEventListener("click", () => {
  rgbMode = "rect";
  $("rgbStatus").textContent =
    rgbTarget() === "face"
      ? "Drag a rectangle around the HEAD (hair to chin)."
      : "Drag a rectangle around the WHOLE person.";
});
$("rgbTarget").addEventListener("change", () => {
  resetRgbSegmentation(); // start a fresh rectangle for the other target (finished results are kept)
  $("rgbStatus").textContent =
    rgbTarget() === "face"
      ? "Face: drag a rectangle around the HEAD (hair to chin)."
      : "Whole body: drag a rectangle around the WHOLE person, head to bottom edge.";
});
$("rgbHumanMode").addEventListener("click", () => {
  rgbMode = "human";
  $("rgbStatus").textContent = "Paint definite human regions.";
});
$("rgbBackgroundMode").addEventListener("click", () => {
  rgbMode = "background";
  $("rgbStatus").textContent = "Paint definite background regions.";
});
$("rgbRefine").addEventListener("click", refineRgb);
$("rgbFinish").addEventListener("click", finalizeRgb);

// ============================================================
// Thermal: upload / camera -> Generate -> Refine -> Save
// ============================================================

// Free every thermal Mat AND set the variable to null, so nothing is ever deleted twice.
function clearThermalMats() {
  deleteMat(thermalSource);
  thermalSource = null;
  deleteMat(thermalSmooth);
  thermalSmooth = null;
  deleteMat(thermalCandidates);
  thermalCandidates = null;
  deleteMat(thermalLabels);
  thermalLabels = null;
  deleteMat(thermalMask);
  thermalMask = null;
  deleteMat(thermalFinalMask);
  thermalFinalMask = null;
  thermalLabelCount = 0;
  thermalSelectedLabels.clear();
}

function clearThermalOutputs() {
  [
    "thermalMaskCanvas",
    "thermalSegmentedCanvas",
    "thermalBoundaryCanvas",
  ].forEach((id) => {
    const c = $(id);
    c.getContext("2d").clearRect(0, 0, c.width, c.height);
  });
  $("thermalArea").textContent = "-";
  $("thermalPerimeter").textContent = "-";
  $("thermalPoints").textContent = "-";
}

// Read the canvas, scale it down if it is large, and make it the thermal source.
function setThermalSourceFromCanvas(canvas) {
  let full = cv.imread(canvas);
  let work = full;
  const maxSide = Math.max(full.cols, full.rows);
  if (maxSide > RGB_MAX_SIDE) {
    const s = RGB_MAX_SIDE / maxSide;
    work = new cv.Mat();
    cv.resize(
      full,
      work,
      new cv.Size(Math.round(full.cols * s), Math.round(full.rows * s)),
      0,
      0,
      cv.INTER_AREA,
    );
    full.delete();
  }

  clearThermalMats();
  clearThermalOutputs();
  thermalSource = work;
  thermalSmooth = thermalPreprocess(work);
  cv.imshow("thermalCanvas", work);
  cv.imshow("thermalPreview", work);
  return `${work.cols}x${work.rows}`;
}

// ---------- input: upload or camera ----------

function loadThermalFile(file) {
  if (!file || !cvOk()) return;
  stopThermalCamera();

  const img = new Image();
  const url = URL.createObjectURL(file);

  img.onload = () => {
    const c = $("thermalCanvas");
    c.width = img.naturalWidth;
    c.height = img.naturalHeight;
    c.getContext("2d").drawImage(img, 0, 0);

    const size = setThermalSourceFromCanvas(c);
    $("thermalStatus").textContent =
      `Thermal image loaded (${size}). Click Generate.`;
    URL.revokeObjectURL(url);
  };
  img.onerror = () => {
    $("thermalStatus").textContent =
      "Could not load the selected thermal image.";
    URL.revokeObjectURL(url);
  };
  img.src = url;
}

async function startThermalCamera() {
  try {
    stopRgbCamera();
    stopThermalCamera();
    const deviceId = $("thermalCameraSelect").value;

    thermalStream = await navigator.mediaDevices.getUserMedia({
      video: deviceId
        ? {
            deviceId: { exact: deviceId },
            width: { ideal: 1920 },
            height: { ideal: 1080 },
          }
        : { width: { ideal: 1920 }, height: { ideal: 1080 } },
      audio: false,
    });

    $("thermalVideo").srcObject = thermalStream;
    await $("thermalVideo").play();
    $("thermalStatus").textContent = "Camera running. Click Capture.";
  } catch (e) {
    $("thermalStatus").textContent = "Could not start camera: " + e.message;
  }
}

function stopThermalCamera() {
  if (thermalStream) thermalStream.getTracks().forEach((t) => t.stop());
  thermalStream = null;
  $("thermalVideo").srcObject = null;
}

function captureThermal() {
  if (!cvOk()) return;
  const video = $("thermalVideo");
  if (!video.videoWidth) {
    $("thermalStatus").textContent = "Start the camera first.";
    return;
  }

  const c = $("thermalCanvas");
  c.width = video.videoWidth;
  c.height = video.videoHeight;
  c.getContext("2d").drawImage(video, 0, 0);

  const size = setThermalSourceFromCanvas(c);
  stopThermalCamera();
  $("thermalStatus").textContent = `Captured (${size}). Click Generate.`;
}

// ---------- classical processing helpers ----------

function thermalPreprocess(src) {
  const gray = new cv.Mat();
  const smooth = new cv.Mat();
  const norm = new cv.Mat();

  cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
  cv.GaussianBlur(gray, smooth, new cv.Size(5, 5), 0, 0, cv.BORDER_DEFAULT);
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
  cv.threshold(
    gray,
    out,
    threshold,
    255,
    invert ? cv.THRESH_BINARY_INV : cv.THRESH_BINARY,
  );
  return out;
}

function cleanupThermal(mask, closeSize) {
  const k3 = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(3, 3));
  const opened = new cv.Mat();
  cv.morphologyEx(mask, opened, cv.MORPH_OPEN, k3);
  k3.delete();

  if (closeSize <= 0) return opened;

  const size = 2 * closeSize + 1;
  const k = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(size, size));
  const closed = new cv.Mat();
  cv.morphologyEx(opened, closed, cv.MORPH_CLOSE, k);
  opened.delete();
  k.delete();
  return closed;
}

function fillExternalContours(mask) {
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  cv.findContours(
    mask,
    contours,
    hierarchy,
    cv.RETR_EXTERNAL,
    cv.CHAIN_APPROX_NONE,
  );
  const filled = cv.Mat.zeros(mask.rows, mask.cols, cv.CV_8UC1);
  cv.drawContours(filled, contours, -1, new cv.Scalar(255), -1);
  contours.delete();
  hierarchy.delete();
  return filled;
}

function removeSmallBlobs(mask, minArea) {
  const labels = new cv.Mat(),
    stats = new cv.Mat(),
    cents = new cv.Mat();
  const n = cv.connectedComponentsWithStats(
    mask,
    labels,
    stats,
    cents,
    8,
    cv.CV_32S,
  );
  const keep = new Uint8Array(n);
  for (let i = 1; i < n; i++)
    keep[i] = stats.intAt(i, cv.CC_STAT_AREA) >= minArea ? 1 : 0;

  const out = cv.Mat.zeros(mask.rows, mask.cols, cv.CV_8UC1);
  const lab = labels.data32S,
    o = out.data;
  for (let i = 0; i < lab.length; i++) if (keep[lab[i]]) o[i] = 255;

  labels.delete();
  stats.delete();
  cents.delete();
  return out;
}

function largestBlobFraction(mask) {
  const labels = new cv.Mat(),
    stats = new cv.Mat(),
    cents = new cv.Mat();
  const n = cv.connectedComponentsWithStats(
    mask,
    labels,
    stats,
    cents,
    8,
    cv.CV_32S,
  );
  let best = 0;
  for (let i = 1; i < n; i++)
    best = Math.max(best, stats.intAt(i, cv.CC_STAT_AREA));
  labels.delete();
  stats.delete();
  cents.delete();
  return best / (mask.rows * mask.cols);
}

// Otsu threshold -> open/close -> fill holes -> drop tiny specks.
function thermalHotRegions(invert) {
  const closeSize = Math.max(
    2,
    Math.round(Math.max(thermalSmooth.cols, thermalSmooth.rows) / 128),
  );
  const thr = getOtsuThreshold(thermalSmooth);
  const t = thresholdThermal(thermalSmooth, thr, invert);
  const c = cleanupThermal(t, closeSize);
  const f = fillExternalContours(c);
  const minArea = Math.max(30, Math.round(0.002 * f.rows * f.cols));
  const g = removeSmallBlobs(f, minArea);
  t.delete();
  c.delete();
  f.delete();
  return g;
}

// Edge refinement: inside the mask = sure human, far outside = sure background,
// a thin band along the edge is left for GrabCut to decide.
function thermalGrabcutRefine(source, binary) {
  const maxDim = Math.max(binary.rows, binary.cols);
  const band = Math.max(3, Math.floor(0.01 * maxDim));
  const size = 2 * band + 1;
  const k = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(size, size));

  const sureFg = new cv.Mat();
  const maybe = new cv.Mat();
  cv.erode(binary, sureFg, k);
  cv.dilate(binary, maybe, k);

  const gc = new cv.Mat(
    binary.rows,
    binary.cols,
    cv.CV_8UC1,
    new cv.Scalar(cv.GC_BGD),
  );
  const g = gc.data,
    bd = binary.data,
    sf = sureFg.data,
    mb = maybe.data;
  let fgN = 0,
    bgN = 0;
  for (let i = 0; i < g.length; i++) {
    let v = cv.GC_BGD;
    if (mb[i] > 0) v = cv.GC_PR_BGD;
    if (bd[i] > 0) v = cv.GC_PR_FGD;
    if (sf[i] > 0) v = cv.GC_FGD;
    g[i] = v;
    if (v === cv.GC_FGD || v === cv.GC_PR_FGD) fgN++;
    else bgN++;
  }

  const rgb = rgbaToRgb(source); // grabCut needs 3 channels, imread gives RGBA
  const bg = new cv.Mat(),
    fg = new cv.Mat();
  let result;
  try {
    if (fgN < 100 || bgN < 100) throw new Error("mask too small for GrabCut");
    cv.grabCut(
      rgb,
      gc,
      new cv.Rect(0, 0, 1, 1),
      bg,
      fg,
      5,
      cv.GC_INIT_WITH_MASK,
    );
    result = binaryFromGrabcut(gc);
  } catch (e) {
    console.warn("Thermal GrabCut refinement skipped:", e);
    result = binary.clone();
  }
  k.delete();
  sureFg.delete();
  maybe.delete();
  gc.delete();
  bg.delete();
  fg.delete();
  rgb.delete();
  return result;
}

function drawOutlines(img, binary, color, thickness) {
  const cs = new cv.MatVector(),
    h = new cv.Mat();
  cv.findContours(binary, cs, h, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE);
  cv.drawContours(img, cs, -1, color, thickness);
  cs.delete();
  h.delete();
}

// Build the human mask from the currently selected region labels.
function updateThermalMask() {
  deleteMat(thermalMask);
  thermalMask = cv.Mat.zeros(
    thermalSource.rows,
    thermalSource.cols,
    cv.CV_8UC1,
  );
  if (!thermalLabels) return;

  const keep = new Uint8Array(Math.max(1, thermalLabelCount));
  thermalSelectedLabels.forEach((l) => {
    if (l > 0 && l < keep.length) keep[l] = 1;
  });

  const lab = thermalLabels.data32S,
    o = thermalMask.data;
  for (let i = 0; i < lab.length; i++) if (keep[lab[i]]) o[i] = 255;
}

// Left: warm regions (yellow) + selected person (green).  Right: segmented person.
function renderThermal() {
  if (!thermalSource || !thermalMask) return;

  const left = thermalSource.clone();
  if (thermalCandidates)
    drawOutlines(left, thermalCandidates, new cv.Scalar(255, 255, 0, 255), 1);
  drawOutlines(left, thermalMask, new cv.Scalar(0, 255, 0, 255), 2);
  const seg = segmentedRGBA(thermalSource, thermalMask);

  cv.imshow("thermalCanvas", left);
  cv.imshow("thermalPreview", seg);

  const info = contourMetrics(thermalMask);
  $("thermalArea").textContent = info.area.toFixed(1) + " px";
  $("thermalPerimeter").textContent = info.perimeter.toFixed(1) + " px";
  $("thermalPoints").textContent = info.points;
  info.contours.delete();

  deleteMat(thermalFinalMask);
  thermalFinalMask = thermalMask.clone();

  left.delete();
  seg.delete();
}

// ---------- the three buttons ----------

function generateThermal() {
  if (!cvOk()) return;
  if (!thermalSource || !thermalSmooth) {
    $("thermalStatus").textContent = "Upload or capture a thermal image first.";
    return;
  }

  // Warm regions. If the bright side covers most of the frame it is the background
  // (black-hot image), so use the other polarity.
  deleteMat(thermalCandidates);
  thermalCandidates = thermalHotRegions(false);
  let darkIsHot = false;
  const frac = largestBlobFraction(thermalCandidates);
  if (frac > 0.6) {
    const alt = thermalHotRegions(true);
    if (largestBlobFraction(alt) < frac) {
      deleteMat(thermalCandidates);
      thermalCandidates = alt;
      darkIsHot = true;
    } else {
      alt.delete();
    }
  }

  // Label the regions; the largest one is the default person.
  deleteMat(thermalLabels);
  thermalLabels = new cv.Mat();
  const stats = new cv.Mat(),
    cents = new cv.Mat();
  thermalLabelCount = cv.connectedComponentsWithStats(
    thermalCandidates,
    thermalLabels,
    stats,
    cents,
    8,
    cv.CV_32S,
  );

  thermalSelectedLabels.clear();
  if (thermalLabelCount > 1) {
    let best = 1,
      bestArea = stats.intAt(1, cv.CC_STAT_AREA);
    for (let i = 2; i < thermalLabelCount; i++) {
      const a = stats.intAt(i, cv.CC_STAT_AREA);
      if (a > bestArea) {
        bestArea = a;
        best = i;
      }
    }
    thermalSelectedLabels.add(best);
  }
  stats.delete();
  cents.delete();

  updateThermalMask();
  renderThermal();

  const regions = Math.max(0, thermalLabelCount - 1);
  $("thermalStatus").textContent = regions
    ? `Generated${darkIsHot ? " (dark-is-hot image detected)" : ""}: ${regions} warm region(s). ` +
      "Green = person. Click a yellow region to add or remove it, then Refine, then Save."
    : "No warm region found. Try a different image.";
}

async function refineThermal() {
  if (!thermalMask) {
    $("thermalStatus").textContent = "Click Generate first.";
    return;
  }
  if (thermalBusy) return;
  if (cv.countNonZero(thermalMask) === 0) {
    $("thermalStatus").textContent =
      "Nothing is selected. Click a yellow region first.";
    return;
  }

  thermalBusy = true;
  $("thermalStatus").textContent = "Refining the boundary with GrabCut...";
  await nextTick();

  try {
    const r = thermalGrabcutRefine(thermalSource, thermalMask);
    const filled = fillExternalContours(r);
    r.delete();
    deleteMat(thermalMask);
    thermalMask = filled;
    renderThermal();
    $("thermalStatus").textContent = "Boundary refined. Click Save.";
  } catch (e) {
    console.error("Thermal refine failed:", e);
    $("thermalStatus").textContent = "Refine failed - see the browser console.";
  } finally {
    thermalBusy = false;
  }
}

function downloadCanvas(canvas, name) {
  canvas.toBlob((blob) => {
    if (!blob) return;
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 3000);
  }, "image/png");
}

function saveThermal() {
  if (!thermalMask || !thermalSource || cv.countNonZero(thermalMask) === 0) {
    $("thermalStatus").textContent = "Click Generate first.";
    return;
  }

  const seg = segmentedRGBA(thermalSource, thermalMask);
  const boundary = drawBoundary(thermalSource, thermalMask);

  cv.imshow("thermalMaskCanvas", thermalMask);
  cv.imshow("thermalSegmentedCanvas", seg);
  cv.imshow("thermalBoundaryCanvas", boundary.out);

  $("thermalArea").textContent = boundary.area.toFixed(1) + " px";
  $("thermalPerimeter").textContent = boundary.perimeter.toFixed(1) + " px";
  $("thermalPoints").textContent = boundary.points;

  deleteMat(thermalFinalMask);
  thermalFinalMask = thermalMask.clone();

  // Download the same four files the Python script writes.
  const original = document.createElement("canvas");
  cv.imshow(original, thermalSource);
  const files = [
    ["thermal_original.png", original],
    ["thermal_human_mask.png", $("thermalMaskCanvas")],
    ["thermal_segmented_human.png", $("thermalSegmentedCanvas")],
    ["thermal_human_boundary.png", $("thermalBoundaryCanvas")],
  ];
  files.forEach(([name, canvas], i) =>
    setTimeout(() => downloadCanvas(canvas, name), i * 300),
  );

  $("thermalStatus").textContent =
    "Saved: thermal_original.png, thermal_human_mask.png, thermal_segmented_human.png, thermal_human_boundary.png";

  seg.delete();
  boundary.out.delete();
}

// ---------- wiring ----------

$("thermalStartCamera").addEventListener("click", startThermalCamera);
$("thermalCapture").addEventListener("click", captureThermal);
$("thermalImageInput").addEventListener("change", (e) => {
  const file = e.target.files && e.target.files[0];
  e.target.value = ""; // so choosing the same file again still fires "change"
  if (file) loadThermalFile(file);
});
$("thermalGenerate").addEventListener("click", generateThermal);
$("thermalRefine").addEventListener("click", refineThermal);
$("thermalSave").addEventListener("click", saveThermal);

// click a region on the left image to add / remove it as the person
$("thermalCanvas").addEventListener("click", (ev) => {
  if (!thermalLabels || !thermalMask) return;
  const p = canvasPoint($("thermalCanvas"), ev);
  if (
    p.x < 0 ||
    p.y < 0 ||
    p.x >= thermalLabels.cols ||
    p.y >= thermalLabels.rows
  )
    return;
  const lab = thermalLabels.intAt(p.y, p.x);
  if (lab <= 0) return;

  if (thermalSelectedLabels.has(lab)) thermalSelectedLabels.delete(lab);
  else thermalSelectedLabels.add(lab);

  updateThermalMask();
  renderThermal();
  $("thermalStatus").textContent =
    "Selection changed. Click Refine (optional), then Save.";
});

// Preloaded SAM2 reference masks (base64 PNG), so the person doesn't have to
// re-upload them every time. Selecting one from the dropdown loads it exactly the
// way an uploaded file would.
const PRELOADED_SAM2_MASKS = {
  rgb_body:
    "data:image/png;base64,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",
  thermal:
    "data:image/png;base64,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",
};

function loadSam2MaskFromDataUrl(dataUrl) {
  if (!cvOk()) return;
  const img = new Image();
  img.onload = () => {
    const c = $("samUploadedCanvas");
    c.width = img.naturalWidth;
    c.height = img.naturalHeight;
    c.getContext("2d").drawImage(img, 0, 0);

    deleteMat(samUploadedMask);
    const src = cv.imread(c);
    const gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
    samUploadedMask = new cv.Mat();
    cv.threshold(gray, samUploadedMask, 127, 255, cv.THRESH_BINARY);
    src.delete();
    gray.delete();
  };
  img.src = dataUrl;
}

// ============================================================
// SAM2 comparison
// ============================================================

$("samMaskSelect").addEventListener("change", (e) => {
  const key = e.target.value;
  if (key === "upload") return; // "Upload my own file" - do nothing here
  loadSam2MaskFromDataUrl(PRELOADED_SAM2_MASKS[key]);
  $("samMaskInput").value = ""; // any previously chosen file no longer applies
});

$("samMaskInput").addEventListener("change", (e) => {
  $("samMaskSelect").value = "upload"; // picking a file overrides a preloaded selection
  const f = e.target.files[0];
  if (!f || !cvOk()) return;

  const img = new Image();
  img.onload = () => {
    const c = $("samUploadedCanvas");
    c.width = img.naturalWidth;
    c.height = img.naturalHeight;
    c.getContext("2d").drawImage(img, 0, 0);

    deleteMat(samUploadedMask);
    const src = cv.imread(c);
    const gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
    samUploadedMask = new cv.Mat();
    cv.threshold(gray, samUploadedMask, 127, 255, cv.THRESH_BINARY);
    src.delete();
    gray.delete();
  };
  img.src = URL.createObjectURL(f);
});

function resizeMaskNearest(mask, w, h) {
  const out = new cv.Mat();
  cv.resize(mask, out, new cv.Size(w, h), 0, 0, cv.INTER_NEAREST);
  return out;
}

function compareMasks() {
  if (!samUploadedMask) {
    alert("Upload a SAM2 mask first.");
    return;
  }

  const which = $("samTarget").value;
  const classical =
    which === "rgb_body"
      ? rgbFinalMasks.body
      : which === "rgb_face"
        ? rgbFinalMasks.face
        : thermalFinalMask;
  if (!classical) {
    alert("Finalize the selected classical segmentation first.");
    return;
  }

  const sam = resizeMaskNearest(
    samUploadedMask,
    classical.cols,
    classical.rows,
  );

  let tp = 0,
    fp = 0,
    fn = 0,
    union = 0,
    inter = 0;
  const diff = cv.Mat.zeros(classical.rows, classical.cols, cv.CV_8UC4);

  for (let y = 0; y < classical.rows; y++) {
    for (let x = 0; x < classical.cols; x++) {
      const a = classical.ucharPtr(y, x)[0] > 0;
      const b = sam.ucharPtr(y, x)[0] > 0;

      if (a && b) {
        tp++;
        inter++;
      }
      if (a && !b) fp++;
      if (!a && b) fn++;
      if (a || b) union++;

      const px = diff.ucharPtr(y, x);
      if (a && b) {
        px[0] = 0;
        px[1] = 255;
        px[2] = 0;
        px[3] = 255;
      } else if (a) {
        px[0] = 255;
        px[1] = 0;
        px[2] = 0;
        px[3] = 255;
      } else if (b) {
        px[0] = 0;
        px[1] = 0;
        px[2] = 255;
        px[3] = 255;
      } else {
        px[0] = 0;
        px[1] = 0;
        px[2] = 0;
        px[3] = 255;
      }
    }
  }

  const iou = union ? inter / union : 1;
  const dice = 2 * tp + fp + fn ? (2 * tp) / (2 * tp + fp + fn) : 1;
  const precision = tp + fp ? tp / (tp + fp) : 1;
  const recall = tp + fn ? tp / (tp + fn) : 1;

  cv.imshow("samClassicalCanvas", classical);
  cv.imshow("samUploadedCanvas", sam);
  cv.imshow("samDiffCanvas", diff);

  $("samIoU").textContent = iou.toFixed(4);
  $("samDice").textContent = dice.toFixed(4);
  $("samPrecision").textContent = precision.toFixed(4);
  $("samRecall").textContent = recall.toFixed(4);

  sam.delete();
  diff.delete();
}

$("compareSam").addEventListener("click", compareMasks);

// ============================================================
// Startup
// ============================================================

window.addEventListener("load", async () => {
  if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
    await enumerateCameras();
  }
});
