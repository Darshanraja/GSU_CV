// ============================================================
// Module 5 - Vercel-compatible browser implementation
// Problem 1: Motion-masked Lucas-Kanade Optical Flow
// Problem 2: Planar Homography + Median Fusion
// ============================================================

const $ = id => document.getElementById(id);

let cvReady = false;

// ============================================================
// PAGE / TAB SETUP
// ============================================================

window.addEventListener("load", () => {
  document.querySelectorAll(".tab").forEach(button => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach(tab => {
        tab.classList.remove("active");
      });

      document.querySelectorAll(".tab-panel").forEach(panel => {
        panel.classList.remove("active-panel");
      });

      button.classList.add("active");
      $(button.dataset.tab).classList.add("active-panel");
    });
  });

  const timer = setInterval(() => {
    if (window.cv && cv.Mat) {
      cvReady = true;
      clearInterval(timer);

      if ($("opencvStatus")) {
        $("opencvStatus").textContent = "OpenCV.js ready";
      }
    }
  }, 250);
});


// ============================================================
// ============================================================
// PROBLEM 1 - OPTICAL FLOW
// ============================================================
// ============================================================

let flowRunning = false;
let flowTimer = null;

let previousGray = null;
let previousPoints = null;

let latestOldPoints = [];
let latestNewPoints = [];

let latestOldFrame = null;
let latestNewFrame = null;

let previousCleanFrameImageData = null;

// Best validation pair for the CURRENT video.
// The best pair is retained even if the tracker loses points near the end.
let currentVideoId = null;
let currentBestValidation = null;

const VALIDATION_SESSION_KEY = "module5_optical_flow_validation_results_v1";

// Used for motion-mask generation
let motionReferenceGray = null;
let frameCounter = 0;

// Motion-mask settings
const MOTION_THRESHOLD = 18;
const MIN_TRACKED_POINTS = 8;
const REDETECT_EVERY = 8;

// Shi-Tomasi settings
const FEATURE_MAX_CORNERS = 120;
const FEATURE_QUALITY = 0.08;
const FEATURE_MIN_DISTANCE = 5;
const FEATURE_BLOCK_SIZE = 7;


// ============================================================
// SESSION-STORAGE HELPERS FOR PROBLEM 1
// ============================================================

function getValidationSessionResults() {
  try {
    return JSON.parse(
      sessionStorage.getItem(
        VALIDATION_SESSION_KEY
      ) || "{}"
    );
  } catch (error) {
    return {};
  }
}


function storeValidationForVideo(
  videoId,
  result
) {
  if (
    !videoId ||
    !result
  ) {
    return;
  }

  const allResults =
    getValidationSessionResults();

  allResults[videoId] =
    result;

  try {
    sessionStorage.setItem(
      VALIDATION_SESSION_KEY,
      JSON.stringify(
        allResults
      )
    );
  } catch (error) {
    console.warn(
      "Could not store validation result in sessionStorage:",
      error
    );
  }
}


function getStoredValidationForVideo(
  videoId
) {
  if (
    !videoId
  ) {
    return null;
  }

  const allResults =
    getValidationSessionResults();

  return (
    allResults[videoId]
    ||
    null
  );
}


function imageDataToDataURL(
  imageData
) {
  const canvas =
    document.createElement(
      "canvas"
    );

  canvas.width =
    imageData.width;

  canvas.height =
    imageData.height;

  canvas
    .getContext("2d")
    .putImageData(
      imageData,
      0,
      0
    );

  return canvas.toDataURL(
    "image/jpeg",
    0.88
  );
}


function calculateValidationRows(
  oldPoints,
  newPoints
) {
  return oldPoints
    .map(
      (
        oldPoint,
        index
      ) => {
        const newPoint =
          newPoints[index];

        const dx =
          newPoint.x
          -
          oldPoint.x;

        const dy =
          newPoint.y
          -
          oldPoint.y;

        const magnitude =
          Math.hypot(
            dx,
            dy
          );

        return {
          oldPoint,
          newPoint,
          dx,
          dy,
          magnitude
        };
      }
    )
    .sort(
      (
        a,
        b
      ) =>
        b.magnitude
        -
        a.magnitude
    )
    .slice(
      0,
      10
    );
}


function considerBestValidationPair(
  oldFrame,
  newFrame,
  oldPoints,
  newPoints
) {
  if (
    !oldFrame ||
    !newFrame ||
    oldPoints.length === 0 ||
    oldPoints.length !== newPoints.length
  ) {
    return;
  }

  const rows =
    calculateValidationRows(
      oldPoints,
      newPoints
    );

  if (
    rows.length === 0
  ) {
    return;
  }

  // Score = average displacement of the strongest tracked points.
  const score =
    rows.reduce(
      (
        sum,
        row
      ) =>
        sum
        +
        row.magnitude,
      0
    )
    /
    rows.length;

  // Ignore effectively motionless pairs.
  if (
    score < 0.02
  ) {
    return;
  }

  if (
    !currentBestValidation
    ||
    score > currentBestValidation.score
  ) {
    currentBestValidation = {
      score,
      frame1:
        imageDataToDataURL(
          oldFrame
        ),
      frame2:
        imageDataToDataURL(
          newFrame
        ),
      rows
    };

    // Persist every new best pair for this browser session.
    storeValidationForVideo(
      currentVideoId,
      currentBestValidation
    );
  }
}


async function drawDataURLToCanvas(
  dataURL,
  canvas
) {
  return new Promise(
    (
      resolve,
      reject
    ) => {
      const image =
        new Image();

      image.onload =
        () => {
          canvas.width =
            image.naturalWidth;

          canvas.height =
            image.naturalHeight;

          canvas
            .getContext("2d")
            .drawImage(
              image,
              0,
              0
            );

          resolve();
        };

      image.onerror =
        reject;

      image.src =
        dataURL;
    }
  );
}


async function renderValidationResult(
  result
) {
  if (
    !result ||
    !result.frame1 ||
    !result.frame2 ||
    !result.rows
  ) {
    return;
  }

  const canvas1 =
    $("validation1");

  const canvas2 =
    $("validation2");

  await Promise.all([
    drawDataURLToCanvas(
      result.frame1,
      canvas1
    ),
    drawDataURLToCanvas(
      result.frame2,
      canvas2
    )
  ]);

  const context1 =
    canvas1.getContext(
      "2d"
    );

  const context2 =
    canvas2.getContext(
      "2d"
    );

  result.rows.forEach(
    (
      row,
      index
    ) => {
      drawNumberedPoint(
        context1,
        row.oldPoint.x,
        row.oldPoint.y,
        index + 1,
        "#00ff55"
      );

      drawNumberedPoint(
        context2,
        row.newPoint.x,
        row.newPoint.y,
        index + 1,
        "#ff3030"
      );
    }
  );

  $("validationTable").innerHTML =
    result.rows
      .map(
        (
          row,
          index
        ) => `
          <tr>
            <td>${index + 1}</td>
            <td>${row.oldPoint.x.toFixed(2)}</td>
            <td>${row.oldPoint.y.toFixed(2)}</td>
            <td>${row.newPoint.x.toFixed(2)}</td>
            <td>${row.newPoint.y.toFixed(2)}</td>
            <td>${row.dx.toFixed(2)}</td>
            <td>${row.dy.toFixed(2)}</td>
            <td>${row.magnitude.toFixed(2)}</td>
          </tr>
        `
      )
      .join("");
}


// ============================================================
// LOAD PRELOADED VIDEOS
// ============================================================

document
  .querySelectorAll(
    ".load-video"
  )
  .forEach(
    button => {
      button.addEventListener(
        "click",
        () => {
          const source =
            button.dataset.video;

          loadVideo(
            source,
            `preloaded:${source}`
          );
        }
      );
    }
  );


// ============================================================
// UPLOAD VIDEO
// ============================================================

$("videoUpload")
  .addEventListener(
    "change",
    event => {
      const file =
        event.target.files[0];

      if (
        !file
      ) {
        return;
      }

      const source =
        URL.createObjectURL(
          file
        );

      const videoId =
        `upload:${file.name}:${file.size}:${file.lastModified}`;

      loadVideo(
        source,
        videoId
      );
    }
  );


// ============================================================
// LOAD VIDEO
// ============================================================

function loadVideo(
  source,
  videoId = source
) {
  // Preserve the current video's best result before switching.
  if (
    currentVideoId &&
    currentBestValidation
  ) {
    storeValidationForVideo(
      currentVideoId,
      currentBestValidation
    );
  }

  stopFlow();

  currentVideoId =
    videoId;

  currentBestValidation =
    getStoredValidationForVideo(
      currentVideoId
    );

  const video =
    $("sourceVideo");

  video.src =
    source;

  video.load();

  video.onloadedmetadata =
    async () => {
      $("videoInfo").innerHTML = [
        [
          "Duration",
          `${video.duration.toFixed(2)} sec`
        ],
        [
          "Resolution",
          `${video.videoWidth} × ${video.videoHeight}`
        ],
        [
          "Status",
          "Ready"
        ],
        [
          "Method",
          "Motion Mask + Lucas-Kanade"
        ]
      ]
        .map(
          item => `
            <div class="info-box">
              ${item[0]}
              <b>${item[1]}</b>
            </div>
          `
        )
        .join("");

      const flowCanvas =
        $("flowCanvas");

      flowCanvas.width =
        video.videoWidth;

      flowCanvas.height =
        video.videoHeight;

      $("validation1").width =
        video.videoWidth;

      $("validation1").height =
        video.videoHeight;

      $("validation2").width =
        video.videoWidth;

      $("validation2").height =
        video.videoHeight;

      // If this video was already processed during this browser session,
      // immediately restore its saved validation pair.
      if (
        currentBestValidation
      ) {
        await renderValidationResult(
          currentBestValidation
        );
      } else {
        const c1 =
          $("validation1")
            .getContext("2d");

        const c2 =
          $("validation2")
            .getContext("2d");

        c1.clearRect(
          0,
          0,
          $("validation1").width,
          $("validation1").height
        );

        c2.clearRect(
          0,
          0,
          $("validation2").width,
          $("validation2").height
        );

        $("validationTable").innerHTML =
          "";
      }
    };
}


// ============================================================
// BUTTON EVENTS
// ============================================================

$("startFlow").addEventListener("click", startFlow);
$("stopFlow").addEventListener("click", stopFlow);
$("captureValidation").addEventListener("click", captureValidationPair);


// ============================================================
// CLEAN OLD OPENCV MATRICES
// ============================================================

function clearFlowMats() {
  if (previousGray) {
    previousGray.delete();
    previousGray = null;
  }

  if (previousPoints) {
    previousPoints.delete();
    previousPoints = null;
  }

  if (motionReferenceGray) {
    motionReferenceGray.delete();
    motionReferenceGray = null;
  }
}


// ============================================================
// BUILD MOTION MASK
//
// Browser-safe alternative to Python MOG2:
// compares current grayscale frame against a slowly changing
// reference frame, thresholds differences, then cleans mask.
// ============================================================

function buildMotionMask(currentGray) {
  const diff = new cv.Mat();
  const mask = new cv.Mat();

  // First frame becomes the motion reference.
  if (!motionReferenceGray) {
    motionReferenceGray = currentGray.clone();

    mask.create(
      currentGray.rows,
      currentGray.cols,
      cv.CV_8UC1
    );

    mask.setTo(new cv.Scalar(0));

    diff.delete();

    return mask;
  }

  // Absolute difference between current frame and reference.
  cv.absdiff(
    currentGray,
    motionReferenceGray,
    diff
  );

  // Threshold strong changes as motion.
  cv.threshold(
    diff,
    mask,
    MOTION_THRESHOLD,
    255,
    cv.THRESH_BINARY
  );

  // Morphology removes tiny noise and connects moving regions.
  const kernel = cv.getStructuringElement(
    cv.MORPH_ELLIPSE,
    new cv.Size(5, 5)
  );

  cv.morphologyEx(
    mask,
    mask,
    cv.MORPH_OPEN,
    kernel,
    new cv.Point(-1, -1),
    1
  );

  cv.morphologyEx(
    mask,
    mask,
    cv.MORPH_CLOSE,
    kernel,
    new cv.Point(-1, -1),
    2
  );

  cv.dilate(
    mask,
    mask,
    kernel,
    new cv.Point(-1, -1),
    2
  );

  kernel.delete();
  diff.delete();

  // Slowly update background reference.
  // This keeps long-term lighting change from becoming permanent motion.
  if (frameCounter % 12 === 0) {
    motionReferenceGray.delete();
    motionReferenceGray = currentGray.clone();
  }

  return mask;
}


// ============================================================
// DETECT FEATURES ONLY INSIDE MOTION MASK
// ============================================================

function detectMotionFeatures(gray, motionMask) {
  const corners = new cv.Mat();

  cv.goodFeaturesToTrack(
    gray,
    corners,
    FEATURE_MAX_CORNERS,
    FEATURE_QUALITY,
    FEATURE_MIN_DISTANCE,
    motionMask,
    FEATURE_BLOCK_SIZE,
    false,
    0.04
  );

  return corners;
}


// ============================================================
// START OPTICAL FLOW
// ============================================================

async function startFlow() {
  if (!cvReady) {
    alert("OpenCV.js is still loading.");
    return;
  }

  const video = $("sourceVideo");

  if (!video.src) {
    alert("Load a video first.");
    return;
  }

  stopFlow();
  clearFlowMats();

  latestOldPoints = [];
  latestNewPoints = [];
  latestOldFrame = null;
  latestNewFrame = null;
  previousCleanFrameImageData = null;
  frameCounter = 0;

  const canvas = $("flowCanvas");
  const context = canvas.getContext("2d");

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  await video.play();

  flowRunning = true;

  function processFrame() {
    if (
      !flowRunning ||
      video.paused ||
      video.ended
    ) {
      if (video.ended) {
        stopFlow();
      }

      return;
    }

    frameCounter++;

    // --------------------------------------------------------
    // CLEAN CURRENT FRAME
    // --------------------------------------------------------

    context.drawImage(
      video,
      0,
      0,
      canvas.width,
      canvas.height
    );

    const cleanImageData = context.getImageData(
      0,
      0,
      canvas.width,
      canvas.height
    );

    const frame = cv.imread(canvas);
    const gray = new cv.Mat();

    cv.cvtColor(
      frame,
      gray,
      cv.COLOR_RGBA2GRAY
    );

    // --------------------------------------------------------
    // MOTION MASK
    // --------------------------------------------------------

    const motionMask = buildMotionMask(gray);

    // --------------------------------------------------------
    // INITIAL / RE-DETECTION
    //
    // Features are detected ONLY inside moving regions.
    // --------------------------------------------------------

    const needRedetect =
      !previousGray ||
      !previousPoints ||
      previousPoints.rows < MIN_TRACKED_POINTS ||
      frameCounter % REDETECT_EVERY === 0;

    if (needRedetect) {
      if (previousPoints) {
        previousPoints.delete();
      }

      const freshPoints = detectMotionFeatures(
        gray,
        motionMask
      );

      previousPoints = freshPoints;

      if (previousGray) {
        previousGray.delete();
      }

      previousGray = gray.clone();

      previousCleanFrameImageData = cleanImageData;

      // Draw motion mask outline for user feedback
      drawMotionMaskOverlay(
        context,
        motionMask
      );

      frame.delete();
      gray.delete();
      motionMask.delete();

      flowTimer = requestAnimationFrame(
        processFrame
      );

      return;
    }

    // --------------------------------------------------------
    // LUCAS-KANADE TRACKING
    // --------------------------------------------------------

    const nextPoints = new cv.Mat();
    const status = new cv.Mat();
    const error = new cv.Mat();

    const winSize = new cv.Size(
      21,
      21
    );

    const criteria = new cv.TermCriteria(
      cv.TermCriteria_EPS |
      cv.TermCriteria_COUNT,
      30,
      0.01
    );

    cv.calcOpticalFlowPyrLK(
      previousGray,
      gray,
      previousPoints,
      nextPoints,
      status,
      error,
      winSize,
      3,
      criteria
    );

    latestOldPoints = [];
    latestNewPoints = [];

    // Redraw clean frame before drawing vectors.
    context.putImageData(
      cleanImageData,
      0,
      0
    );

    // --------------------------------------------------------
    // DRAW MOTION REGION
    // --------------------------------------------------------

    drawMotionMaskOverlay(
      context,
      motionMask
    );

    // --------------------------------------------------------
    // FILTER TRACKED POINTS
    //
    // Keep point only if current location is inside motion mask.
    // This removes stationary house/car/window features.
    // --------------------------------------------------------

    for (
      let i = 0;
      i < status.rows;
      i++
    ) {
      if (status.data[i] !== 1) {
        continue;
      }

      const x1 =
        previousPoints.data32F[i * 2];

      const y1 =
        previousPoints.data32F[i * 2 + 1];

      const x2 =
        nextPoints.data32F[i * 2];

      const y2 =
        nextPoints.data32F[i * 2 + 1];

      const mx =
        Math.max(
          0,
          Math.min(
            motionMask.cols - 1,
            Math.round(x2)
          )
        );

      const my =
        Math.max(
          0,
          Math.min(
            motionMask.rows - 1,
            Math.round(y2)
          )
        );

      const maskValue =
        motionMask.ucharPtr(
          my,
          mx
        )[0];

      // Reject tracked points outside moving foreground.
      if (maskValue === 0) {
        continue;
      }

      latestOldPoints.push({
        x: x1,
        y: y1
      });

      latestNewPoints.push({
        x: x2,
        y: y2
      });

      const dx = x2 - x1;
      const dy = y2 - y1;

      // Green flow line.
      context.strokeStyle =
        "#00ff55";

      context.lineWidth =
        2;

      context.beginPath();

      context.moveTo(
        x1,
        y1
      );

      context.lineTo(
        x2,
        y2
      );

      context.stroke();

      // Arrow head.
      const angle =
        Math.atan2(
          dy,
          dx
        );

      const arrowLength =
        8;

      context.beginPath();

      context.moveTo(
        x2,
        y2
      );

      context.lineTo(
        x2 -
        arrowLength *
        Math.cos(angle - 0.5),

        y2 -
        arrowLength *
        Math.sin(angle - 0.5)
      );

      context.moveTo(
        x2,
        y2
      );

      context.lineTo(
        x2 -
        arrowLength *
        Math.cos(angle + 0.5),

        y2 -
        arrowLength *
        Math.sin(angle + 0.5)
      );

      context.stroke();

      // Red current point.
      context.fillStyle =
        "#ff3030";

      context.beginPath();

      context.arc(
        x2,
        y2,
        4,
        0,
        Math.PI * 2
      );

      context.fill();
    }

    // --------------------------------------------------------
    // SAVE CLEAN VALIDATION PAIR
    // --------------------------------------------------------

    latestOldFrame =
      previousCleanFrameImageData;

    latestNewFrame =
      cleanImageData;

    // Keep the strongest valid pair seen anywhere in this video.
    // This prevents the 29-second video from losing its result
    // if tracking disappears on the final frames.
    considerBestValidationPair(
      latestOldFrame,
      latestNewFrame,
      latestOldPoints,
      latestNewPoints
    );

    // --------------------------------------------------------
    // KEEP ONLY VALID MOTION POINTS
    // --------------------------------------------------------

    if (previousPoints) {
      previousPoints.delete();
    }

    if (latestNewPoints.length > 0) {
      const flat = [];

      latestNewPoints.forEach(point => {
        flat.push(
          point.x,
          point.y
        );
      });

      previousPoints =
        cv.matFromArray(
          latestNewPoints.length,
          1,
          cv.CV_32FC2,
          flat
        );
    } else {
      previousPoints =
        new cv.Mat();
    }

    if (previousGray) {
      previousGray.delete();
    }

    previousGray =
      gray.clone();

    previousCleanFrameImageData =
      cleanImageData;

    // --------------------------------------------------------
    // CLEAN TEMP MATRICES
    // --------------------------------------------------------

    frame.delete();
    gray.delete();
    motionMask.delete();
    nextPoints.delete();
    status.delete();
    error.delete();

    flowTimer =
      requestAnimationFrame(
        processFrame
      );
  }

  processFrame();
}


// ============================================================
// DRAW MOTION MASK OVERLAY
//
// Blue contours show where motion is detected.
// ============================================================

function drawMotionMaskOverlay(
  context,
  motionMask
) {
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();

  const temp =
    motionMask.clone();

  cv.findContours(
    temp,
    contours,
    hierarchy,
    cv.RETR_EXTERNAL,
    cv.CHAIN_APPROX_SIMPLE
  );

  context.strokeStyle =
    "#00aaff";

  context.lineWidth =
    2;

  for (
    let i = 0;
    i < contours.size();
    i++
  ) {
    const contour =
      contours.get(i);

    const rect =
      cv.boundingRect(
        contour
      );

    // Ignore tiny motion blobs.
    if (
      rect.width *
      rect.height <
      250
    ) {
      contour.delete();
      continue;
    }

    context.strokeRect(
      rect.x,
      rect.y,
      rect.width,
      rect.height
    );

    contour.delete();
  }

  temp.delete();
  contours.delete();
  hierarchy.delete();
}


// ============================================================
// STOP OPTICAL FLOW
// ============================================================

function stopFlow() {
  flowRunning = false;

  if (
    flowTimer
  ) {
    cancelAnimationFrame(
      flowTimer
    );
  }

  flowTimer =
    null;

  const video =
    $("sourceVideo");

  if (
    video &&
    !video.paused
  ) {
    video.pause();
  }

  if (
    currentVideoId &&
    currentBestValidation
  ) {
    storeValidationForVideo(
      currentVideoId,
      currentBestValidation
    );
  }

  clearFlowMats();
}


// ============================================================
// CAPTURE VALIDATION PAIR
// ============================================================

function captureValidationPair() {
  const saved =
    currentBestValidation
    ||
    getStoredValidationForVideo(
      currentVideoId
    );

  if (
    !saved
  ) {
    alert(
      "No valid moving-point pair has been saved yet. Start optical flow and let it track motion for a few seconds."
    );

    return;
  }

  currentBestValidation =
    saved;

  renderValidationResult(
    saved
  );
}


// ============================================================
// DRAW NUMBERED VALIDATION POINT
// ============================================================

function drawNumberedPoint(
  context,
  x,
  y,
  number,
  color
) {
  context.fillStyle =
    color;

  context.beginPath();

  context.arc(
    x,
    y,
    7,
    0,
    Math.PI * 2
  );

  context.fill();

  context.fillStyle =
    "#111";

  context.font =
    "bold 14px Arial";

  context.fillText(
    String(number),
    x + 9,
    y - 6
  );
}


// ============================================================
// ============================================================
// PROBLEM 2 - PLANAR HOMOGRAPHY RECONSTRUCTION
// ============================================================
// ============================================================

const PRELOADED_BOOK_VIEWS = [
  "module5_assets/problem2/view1.png",
  "module5_assets/problem2/view2.png",
  "module5_assets/problem2/view3.png",
  "module5_assets/problem2/view4.png"
];

let bookImages = [];


// ============================================================
// LOAD PRELOADED BOOK VIEWS
// ============================================================

$("loadBookViews").addEventListener(
  "click",
  async () => {
    bookImages = [];

    for (
      const source of
      PRELOADED_BOOK_VIEWS
    ) {
      const image =
        await loadImage(
          source
        );

      bookImages.push({
        image,
        source,
        points: []
      });
    }

    renderBookViews();
  }
);


// ============================================================
// UPLOAD FOUR BOOK IMAGES
// ============================================================

$("imageUpload").addEventListener(
  "change",
  async event => {
    const files = [
      ...event.target.files
    ];

    if (
      files.length !== 4
    ) {
      alert(
        "Select exactly four images."
      );

      return;
    }

    bookImages = [];

    for (
      const file of files
    ) {
      const source =
        URL.createObjectURL(
          file
        );

      const image =
        await loadImage(
          source
        );

      bookImages.push({
        image,
        source,
        points: []
      });
    }

    renderBookViews();
  }
);


// ============================================================
// CLEAR BOOK VIEWS
// ============================================================

$("clearBookViews").addEventListener(
  "click",
  () => {
    bookImages = [];

    $("bookViews").innerHTML =
      "";

    $("homographyOutput")
      .classList
      .add(
        "hidden"
      );
  }
);


// ============================================================
// LOAD IMAGE
// ============================================================

function loadImage(source) {
  return new Promise(
    (
      resolve,
      reject
    ) => {
      const image =
        new Image();

      image.onload =
        () => resolve(image);

      image.onerror =
        reject;

      image.src =
        source;
    }
  );
}


// ============================================================
// RENDER FOUR BOOK VIEWS
// ============================================================

function renderBookViews() {
  const container =
    $("bookViews");

  container.innerHTML =
    "";

  bookImages.forEach(
    (
      item,
      viewIndex
    ) => {
      const card =
        document.createElement(
          "div"
        );

      card.className =
        "view-card";

      card.innerHTML = `
        <h3>
          View ${viewIndex + 1}
        </h3>

        <canvas></canvas>

        <div class="view-instructions">
          Click:
          1 Top Left →
          2 Top Right →
          3 Bottom Right →
          4 Bottom Left
        </div>

        <button
          class="secondary reset-view"
          style="margin-top:8px"
        >
          Reset Points
        </button>
      `;

      container.appendChild(
        card
      );

      const canvas =
        card.querySelector(
          "canvas"
        );

      const context =
        canvas.getContext(
          "2d"
        );

      const image =
        item.image;

      canvas.width =
        image.naturalWidth;

      canvas.height =
        image.naturalHeight;


      function redraw() {
        context.drawImage(
          image,
          0,
          0,
          canvas.width,
          canvas.height
        );

        if (
          item.points.length > 1
        ) {
          context.strokeStyle =
            "#00ff55";

          context.lineWidth =
            Math.max(
              3,
              canvas.width / 500
            );

          context.beginPath();

          context.moveTo(
            item.points[0].x,
            item.points[0].y
          );

          for (
            let i = 1;
            i < item.points.length;
            i++
          ) {
            context.lineTo(
              item.points[i].x,
              item.points[i].y
            );
          }

          if (
            item.points.length === 4
          ) {
            context.closePath();
          }

          context.stroke();
        }

        item.points.forEach(
          (
            point,
            pointIndex
          ) => {
            context.fillStyle =
              "#ff3030";

            context.beginPath();

            context.arc(
              point.x,
              point.y,
              Math.max(
                7,
                canvas.width / 250
              ),
              0,
              Math.PI * 2
            );

            context.fill();

            context.fillStyle =
              "#00ff55";

            context.font =
              `bold ${
                Math.max(
                  18,
                  canvas.width / 80
                )
              }px Arial`;

            context.fillText(
              String(
                pointIndex + 1
              ),
              point.x + 12,
              point.y - 10
            );
          }
        );
      }

      redraw();


      canvas.addEventListener(
        "click",
        event => {
          if (
            item.points.length >= 4
          ) {
            return;
          }

          const rect =
            canvas.getBoundingClientRect();

          const x =
            (
              event.clientX -
              rect.left
            )
            *
            canvas.width
            /
            rect.width;

          const y =
            (
              event.clientY -
              rect.top
            )
            *
            canvas.height
            /
            rect.height;

          item.points.push({
            x,
            y
          });

          redraw();
        }
      );


      card
        .querySelector(
          ".reset-view"
        )
        .addEventListener(
          "click",
          () => {
            item.points = [];
            redraw();
          }
        );
    }
  );
}


// ============================================================
// RECONSTRUCT BUTTON
// ============================================================

$("reconstructBook").addEventListener(
  "click",
  reconstructBook
);


// ============================================================
// CALIBRATED CAMERA PARAMETERS
// Source: user's OpenCV checkerboard calibration at 1920 x 1080
// ============================================================

const CAMERA_CALIBRATION = {
  fx: 2301.4244,
  fy: 2304.4655,
  cx: 973.7959,
  cy: 494.0908,
  width: 1920,
  height: 1080
};


// Estimate the physical cover aspect ratio from the four clicked
// quadrilaterals. This avoids inventing a book width/height.
// Object width is then defined as 1 normalized unit.
function estimateBookAspectRatio() {
  const ratios =
    bookImages.map(
      item => {
        const p =
          item.points;

        const distance =
          (
            a,
            b
          ) =>
            Math.hypot(
              b.x - a.x,
              b.y - a.y
            );

        const width =
          (
            distance(
              p[0],
              p[1]
            )
            +
            distance(
              p[3],
              p[2]
            )
          )
          /
          2;

        const height =
          (
            distance(
              p[0],
              p[3]
            )
            +
            distance(
              p[1],
              p[2]
            )
          )
          /
          2;

        return (
          width
          /
          height
        );
      }
    )
    .filter(
      value =>
        Number.isFinite(
          value
        )
        &&
        value > 0.2
        &&
        value < 2.0
    )
    .sort(
      (
        a,
        b
      ) =>
        a - b
    );

  if (
    ratios.length === 0
  ) {
    return 2 / 3;
  }

  const mid =
    Math.floor(
      ratios.length / 2
    );

  if (
    ratios.length % 2 === 1
  ) {
    return ratios[mid];
  }

  return (
    ratios[mid - 1]
    +
    ratios[mid]
  )
  /
  2;
}


// Scale the supplied 1920x1080 calibration to each image.
// Portrait images are treated as a 90-degree clockwise rotation
// of the calibrated landscape camera.
function getCalibratedCameraMatrix(
  imageWidth,
  imageHeight
) {
  let fx;
  let fy;
  let cx;
  let cy;
  let note;

  if (
    imageWidth >= imageHeight
  ) {
    const sx =
      imageWidth
      /
      CAMERA_CALIBRATION.width;

    const sy =
      imageHeight
      /
      CAMERA_CALIBRATION.height;

    fx =
      CAMERA_CALIBRATION.fx
      *
      sx;

    fy =
      CAMERA_CALIBRATION.fy
      *
      sy;

    cx =
      CAMERA_CALIBRATION.cx
      *
      sx;

    cy =
      CAMERA_CALIBRATION.cy
      *
      sy;

    note =
      "Landscape calibration scaled from 1920x1080.";
  } else {
    // Rotate calibrated intrinsics into portrait coordinates.
    const rotatedWidth =
      CAMERA_CALIBRATION.height;

    const rotatedHeight =
      CAMERA_CALIBRATION.width;

    const rotatedFx =
      CAMERA_CALIBRATION.fy;

    const rotatedFy =
      CAMERA_CALIBRATION.fx;

    const rotatedCx =
      CAMERA_CALIBRATION.height
      -
      1
      -
      CAMERA_CALIBRATION.cy;

    const rotatedCy =
      CAMERA_CALIBRATION.cx;

    const sx =
      imageWidth
      /
      rotatedWidth;

    const sy =
      imageHeight
      /
      rotatedHeight;

    fx =
      rotatedFx
      *
      sx;

    fy =
      rotatedFy
      *
      sy;

    cx =
      rotatedCx
      *
      sx;

    cy =
      rotatedCy
      *
      sy;

    note =
      "Portrait image: 1920x1080 calibration rotated 90 degrees clockwise, then scaled.";
  }

  const K =
    cv.matFromArray(
      3,
      3,
      cv.CV_64F,
      [
        fx,
        0,
        cx,

        0,
        fy,
        cy,

        0,
        0,
        1
      ]
    );

  return {
    K,
    fx,
    fy,
    cx,
    cy,
    note
  };
}


function estimateCameraPoseForView(
  item,
  aspectRatio,
  viewIndex
) {
  // Width is 1 normalized cover unit.
  // Height is determined from the measured image aspect ratio.
  // Translation and camera center are therefore reported in
  // normalized cover-width units (not centimeters).
  const objectWidth =
    1.0;

  const objectHeight =
    1.0
    /
    aspectRatio;

  const objectPoints =
    cv.matFromArray(
      4,
      3,
      cv.CV_64F,
      [
        0,
        0,
        0,

        objectWidth,
        0,
        0,

        objectWidth,
        objectHeight,
        0,

        0,
        objectHeight,
        0
      ]
    );

  const imagePoints =
    cv.matFromArray(
      4,
      2,
      cv.CV_64F,
      item.points.flatMap(
        point => [
          point.x,
          point.y
        ]
      )
    );

  const calibration =
    getCalibratedCameraMatrix(
      item.image.naturalWidth,
      item.image.naturalHeight
    );

  // Distortion coefficients were not supplied in the current
  // calibration values, so zero distortion is used here.
  const distortion =
    cv.Mat.zeros(
      4,
      1,
      cv.CV_64F
    );

  const rvec =
    new cv.Mat();

  const tvec =
    new cv.Mat();

  let success =
    false;

  // IPPE is specifically designed for coplanar points.
  if (
    typeof cv.SOLVEPNP_IPPE !== "undefined"
  ) {
    try {
      success =
        cv.solvePnP(
          objectPoints,
          imagePoints,
          calibration.K,
          distortion,
          rvec,
          tvec,
          false,
          cv.SOLVEPNP_IPPE
        );
    } catch (error) {
      success =
        false;
    }
  }

  // Fallback for OpenCV.js builds without IPPE support.
  if (
    !success
  ) {
    success =
      cv.solvePnP(
        objectPoints,
        imagePoints,
        calibration.K,
        distortion,
        rvec,
        tvec,
        false,
        cv.SOLVEPNP_ITERATIVE
      );
  }

  if (
    !success
  ) {
    objectPoints.delete();
    imagePoints.delete();
    calibration.K.delete();
    distortion.delete();
    rvec.delete();
    tvec.delete();

    throw new Error(
      `Camera pose estimation failed for View ${viewIndex}.`
    );
  }

  const rotation =
    new cv.Mat();

  cv.Rodrigues(
    rvec,
    rotation
  );

  const R =
    Array.from(
      rotation.data64F
    );

  const t =
    Array.from(
      tvec.data64F
    );

  // Camera center in object-plane coordinates:
  // C = -R^T t
  const cameraCenter = [
    -(
      R[0] * t[0]
      +
      R[3] * t[1]
      +
      R[6] * t[2]
    ),

    -(
      R[1] * t[0]
      +
      R[4] * t[1]
      +
      R[7] * t[2]
    ),

    -(
      R[2] * t[0]
      +
      R[5] * t[1]
      +
      R[8] * t[2]
    )
  ];

  const result = {
    viewIndex,
    imageWidth:
      item.image.naturalWidth,
    imageHeight:
      item.image.naturalHeight,
    fx:
      calibration.fx,
    fy:
      calibration.fy,
    cx:
      calibration.cx,
    cy:
      calibration.cy,
    calibrationNote:
      calibration.note,
    R,
    t,
    cameraCenter
  };

  objectPoints.delete();
  imagePoints.delete();
  calibration.K.delete();
  distortion.delete();
  rvec.delete();
  tvec.delete();
  rotation.delete();

  return result;
}


function ensureCameraPoseSection() {
  let section =
    $("cameraPoseSection");

  if (
    section
  ) {
    return section;
  }

  section =
    document.createElement(
      "div"
    );

  section.id =
    "cameraPoseSection";

  section.innerHTML = `
    <h3>Calibrated Camera Parameters and Estimated Camera Poses</h3>

    <p>
      Camera intrinsics use the supplied OpenCV calibration:
      fx = 2301.4244, fy = 2304.4655,
      cx = 973.7959, cy = 494.0908 at 1920 × 1080.
      Translation is reported in normalized cover-width units because
      the physical book dimensions were not supplied.
    </p>

    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>View</th>
            <th>fx</th>
            <th>fy</th>
            <th>cx</th>
            <th>cy</th>
            <th>Cx</th>
            <th>Cy</th>
            <th>Cz</th>
          </tr>
        </thead>

        <tbody id="cameraPoseTable"></tbody>
      </table>
    </div>

    <canvas
      id="cameraPoseCanvas"
      width="800"
      height="420"
      style="margin-top:16px; background:white;"
    ></canvas>

    <pre id="cameraPoseOutput"></pre>
  `;

  $("homographyOutput")
    .appendChild(
      section
    );

  return section;
}


function drawCameraPosePlot(
  poses,
  aspectRatio
) {
  const canvas =
    $("cameraPoseCanvas");

  const context =
    canvas.getContext(
      "2d"
    );

  const width =
    canvas.width;

  const height =
    canvas.height;

  context.clearRect(
    0,
    0,
    width,
    height
  );

  context.fillStyle =
    "#ffffff";

  context.fillRect(
    0,
    0,
    width,
    height
  );

  const centers =
    poses.map(
      pose =>
        pose.cameraCenter
    );

  const allX = [
    0,
    1,
    ...centers.map(
      c => c[0]
    )
  ];

  const allZ = [
    0,
    ...centers.map(
      c => c[2]
    )
  ];

  let minX =
    Math.min(
      ...allX
    );

  let maxX =
    Math.max(
      ...allX
    );

  let minZ =
    Math.min(
      ...allZ
    );

  let maxZ =
    Math.max(
      ...allZ
    );

  if (
    Math.abs(
      maxX - minX
    ) < 0.1
  ) {
    minX -= 1;
    maxX += 1;
  }

  if (
    Math.abs(
      maxZ - minZ
    ) < 0.1
  ) {
    minZ -= 1;
    maxZ += 1;
  }

  const pad =
    50;

  const mapX =
    x =>
      pad
      +
      (
        x - minX
      )
      /
      (
        maxX - minX
      )
      *
      (
        width - 2 * pad
      );

  const mapZ =
    z =>
      height
      -
      pad
      -
      (
        z - minZ
      )
      /
      (
        maxZ - minZ
      )
      *
      (
        height - 2 * pad
      );

  // Object plane shown in top-down X-Z view at Z = 0.
  context.strokeStyle =
    "#00aa44";

  context.lineWidth =
    5;

  context.beginPath();

  context.moveTo(
    mapX(0),
    mapZ(0)
  );

  context.lineTo(
    mapX(1),
    mapZ(0)
  );

  context.stroke();

  context.fillStyle =
    "#006b2f";

  context.font =
    "14px Arial";

  context.fillText(
    "Book plane (normalized width = 1)",
    mapX(0),
    mapZ(0) - 12
  );

  poses.forEach(
    pose => {
      const center =
        pose.cameraCenter;

      const x =
        mapX(
          center[0]
        );

      const z =
        mapZ(
          center[2]
        );

      context.fillStyle =
        "#0039A6";

      context.beginPath();

      context.arc(
        x,
        z,
        8,
        0,
        Math.PI * 2
      );

      context.fill();

      context.fillStyle =
        "#111827";

      context.fillText(
        `Camera ${pose.viewIndex}`,
        x + 10,
        z - 8
      );

      context.strokeStyle =
        "#8a94a3";

      context.lineWidth =
        1;

      context.beginPath();

      context.moveTo(
        x,
        z
      );

      context.lineTo(
        mapX(0.5),
        mapZ(0)
      );

      context.stroke();
    }
  );

  context.fillStyle =
    "#4b5563";

  context.fillText(
    "Top-down X-Z view; camera positions are relative and scaled in cover-width units.",
    20,
    height - 15
  );
}


// ============================================================
// PLANAR RECONSTRUCTION
// ============================================================

function reconstructBook() {
  if (
    !cvReady
  ) {
    alert(
      "OpenCV.js is still loading."
    );

    return;
  }

  if (
    bookImages.length !== 4
  ) {
    alert(
      "Load four images first."
    );

    return;
  }

  if (
    bookImages.some(
      view =>
        view.points.length !== 4
    )
  ) {
    alert(
      "Select four corners in every view first."
    );

    return;
  }

  // ========================================================
  // ESTIMATE PLANAR COVER ASPECT RATIO
  // ========================================================

  const aspectRatio =
    estimateBookAspectRatio();

  // Preserve a natural portrait shape instead of forcing 600x400.
  const referenceHeight =
    600;

  const referenceWidth =
    Math.max(
      280,
      Math.min(
        700,
        Math.round(
          referenceHeight
          *
          aspectRatio
        )
      )
    );

  const referencePoints =
    cv.matFromArray(
      4,
      1,
      cv.CV_32FC2,
      [
        0,
        0,

        referenceWidth - 1,
        0,

        referenceWidth - 1,
        referenceHeight - 1,

        0,
        referenceHeight - 1
      ]
    );

  const warpedMats =
    [];

  const homographyMatrices =
    [];

  const coordinateRows =
    [];

  const rectifiedContainer =
    $("rectifiedViews");

  rectifiedContainer.innerHTML =
    "";

  // ========================================================
  // CAMERA POSE ESTIMATION USING CALIBRATED K
  // ========================================================

  const cameraPoses =
    [];

  for (
    let viewIndex = 0;
    viewIndex < bookImages.length;
    viewIndex++
  ) {
    try {
      const pose =
        estimateCameraPoseForView(
          bookImages[viewIndex],
          aspectRatio,
          viewIndex + 1
        );

      cameraPoses.push(
        pose
      );
    } catch (error) {
      console.warn(
        error
      );
    }
  }

  // ========================================================
  // HOMOGRAPHY RECTIFICATION FOR EACH VIEW
  // ========================================================

  bookImages.forEach(
    (
      item,
      viewIndex
    ) => {
      const sourceCanvas =
        document.createElement(
          "canvas"
        );

      sourceCanvas.width =
        item.image.naturalWidth;

      sourceCanvas.height =
        item.image.naturalHeight;

      sourceCanvas
        .getContext(
          "2d"
        )
        .drawImage(
          item.image,
          0,
          0
        );

      const sourceMat =
        cv.imread(
          sourceCanvas
        );

      const sourcePoints =
        cv.matFromArray(
          4,
          1,
          cv.CV_32FC2,
          item.points.flatMap(
            point => [
              point.x,
              point.y
            ]
          )
        );

      const homography =
        cv.getPerspectiveTransform(
          sourcePoints,
          referencePoints
        );

      homographyMatrices.push(
        Array.from(
          homography.data64F
        )
      );

      const warped =
        new cv.Mat();

      cv.warpPerspective(
        sourceMat,
        warped,
        homography,
        new cv.Size(
          referenceWidth,
          referenceHeight
        ),
        cv.INTER_LINEAR,
        cv.BORDER_CONSTANT,
        new cv.Scalar(
          0,
          0,
          0,
          255
        )
      );

      warpedMats.push(
        warped
      );

      // ----------------------------------------------------
      // Display each rectified view
      // ----------------------------------------------------

      const card =
        document.createElement(
          "div"
        );

      card.className =
        "view-card";

      card.innerHTML = `
        <h3>
          Rectified View ${viewIndex + 1}
        </h3>

        <canvas></canvas>
      `;

      rectifiedContainer.appendChild(
        card
      );

      cv.imshow(
        card.querySelector(
          "canvas"
        ),
        warped
      );

      // ----------------------------------------------------
      // Save clicked image coordinates
      // ----------------------------------------------------

      const cornerNames = [
        "Top Left",
        "Top Right",
        "Bottom Right",
        "Bottom Left"
      ];

      item.points.forEach(
        (
          point,
          pointIndex
        ) => {
          coordinateRows.push([
            viewIndex + 1,
            pointIndex + 1,
            cornerNames[
              pointIndex
            ],
            point.x,
            point.y
          ]);
        }
      );

      sourceMat.delete();
      sourcePoints.delete();
      homography.delete();
    }
  );

  // ========================================================
  // MEDIAN FUSION OF RECTIFIED VIEWS
  // ========================================================

  const reconstructed =
    new cv.Mat(
      referenceHeight,
      referenceWidth,
      cv.CV_8UC4
    );

  const totalPixels =
    referenceWidth
    *
    referenceHeight;

  for (
    let pixelIndex = 0;
    pixelIndex < totalPixels;
    pixelIndex++
  ) {
    const base =
      pixelIndex
      *
      4;

    for (
      let channel = 0;
      channel < 4;
      channel++
    ) {
      const values = [
        warpedMats[0].data[
          base + channel
        ],
        warpedMats[1].data[
          base + channel
        ],
        warpedMats[2].data[
          base + channel
        ],
        warpedMats[3].data[
          base + channel
        ]
      ]
        .sort(
          (
            a,
            b
          ) =>
            a - b
        );

      reconstructed.data[
        base + channel
      ] =
        Math.round(
          (
            values[1]
            +
            values[2]
          )
          /
          2
        );
    }
  }

  cv.imshow(
    "reconstructedCanvas",
    reconstructed
  );

  // ========================================================
  // CANONICAL PLANAR BOUNDARY
  // ========================================================

  const boundary =
    reconstructed.clone();

  cv.rectangle(
    boundary,
    new cv.Point(
      1,
      1
    ),
    new cv.Point(
      referenceWidth - 2,
      referenceHeight - 2
    ),
    new cv.Scalar(
      0,
      255,
      0,
      255
    ),
    5
  );

  cv.imshow(
    "boundaryCanvas",
    boundary
  );

  // Add an explicit label so this is not misreported as a
  // triangulated/automatically discovered 3D boundary.
  if (
    !$(
      "canonicalBoundaryNote"
    )
  ) {
    const note =
      document.createElement(
        "p"
      );

    note.id =
      "canonicalBoundaryNote";

    note.textContent =
      "Green outline: canonical planar boundary obtained from the four corresponding cover corners. Camera poses below are estimated separately using the calibrated camera matrix.";

    $("boundaryCanvas")
      .parentElement
      .appendChild(
        note
      );
  }

  // ========================================================
  // HOMOGRAPHY MATRICES
  // ========================================================

  $("homographyMatrices").textContent =
    homographyMatrices
      .map(
        (
          matrix,
          index
        ) => {
          const rows = [
            matrix.slice(
              0,
              3
            ),
            matrix.slice(
              3,
              6
            ),
            matrix.slice(
              6,
              9
            )
          ];

          return (
            `View ${index + 1}\n`
            +
            rows
              .map(
                row =>
                  row
                    .map(
                      value =>
                        value.toExponential(
                          6
                        )
                    )
                    .join(
                      "   "
                    )
              )
              .join(
                "\n"
              )
          );
        }
      )
      .join(
        "\n\n"
      );

  // ========================================================
  // SELECTED PIXEL COORDINATES
  // ========================================================

  $("pointCoordinates").innerHTML =
    coordinateRows
      .map(
        row => `
          <tr>
            <td>${row[0]}</td>
            <td>${row[1]}</td>
            <td>${row[2]}</td>
            <td>${row[3].toFixed(2)}</td>
            <td>${row[4].toFixed(2)}</td>
          </tr>
        `
      )
      .join("");

  // ========================================================
  // DISPLAY CALIBRATED CAMERA POSES
  // ========================================================

  ensureCameraPoseSection();

  $("cameraPoseTable").innerHTML =
    cameraPoses
      .map(
        pose => `
          <tr>
            <td>${pose.viewIndex}</td>
            <td>${pose.fx.toFixed(2)}</td>
            <td>${pose.fy.toFixed(2)}</td>
            <td>${pose.cx.toFixed(2)}</td>
            <td>${pose.cy.toFixed(2)}</td>
            <td>${pose.cameraCenter[0].toFixed(4)}</td>
            <td>${pose.cameraCenter[1].toFixed(4)}</td>
            <td>${pose.cameraCenter[2].toFixed(4)}</td>
          </tr>
        `
      )
      .join("");

  $("cameraPoseOutput").textContent =
    [
      "BASE CAMERA CALIBRATION",
      "K_calibrated =",
      `[${CAMERA_CALIBRATION.fx}, 0, ${CAMERA_CALIBRATION.cx}]`,
      `[0, ${CAMERA_CALIBRATION.fy}, ${CAMERA_CALIBRATION.cy}]`,
      "[0, 0, 1]",
      "",
      `Calibration resolution: ${CAMERA_CALIBRATION.width} x ${CAMERA_CALIBRATION.height}`,
      `Estimated planar cover aspect ratio (width / height): ${aspectRatio.toFixed(4)}`,
      "Object-plane scale: width = 1 normalized cover-width unit.",
      "Distortion: set to zero here because distortion coefficients were not supplied with the current calibration values.",
      "",
      ...cameraPoses.flatMap(
        pose => [
          `VIEW ${pose.viewIndex}`,
          `Image resolution: ${pose.imageWidth} x ${pose.imageHeight}`,
          `Scaled K =`,
          `[${pose.fx.toFixed(4)}, 0, ${pose.cx.toFixed(4)}]`,
          `[0, ${pose.fy.toFixed(4)}, ${pose.cy.toFixed(4)}]`,
          `[0, 0, 1]`,
          `Calibration handling: ${pose.calibrationNote}`,
          "Rotation R =",
          `[${pose.R.slice(0,3).map(v=>v.toFixed(6)).join(", ")}]`,
          `[${pose.R.slice(3,6).map(v=>v.toFixed(6)).join(", ")}]`,
          `[${pose.R.slice(6,9).map(v=>v.toFixed(6)).join(", ")}]`,
          `Translation t = [${pose.t.map(v=>v.toFixed(6)).join(", ")}]`,
          `Camera center C = -R^T t = [${pose.cameraCenter.map(v=>v.toFixed(6)).join(", ")}]`,
          ""
        ]
      )
    ]
      .join(
        "\n"
      );

  if (
    cameraPoses.length > 0
  ) {
    drawCameraPosePlot(
      cameraPoses,
      aspectRatio
    );
  }

  $("homographyOutput")
    .classList
    .remove(
      "hidden"
    );

  // ========================================================
  // CLEAN MEMORY
  // ========================================================

  boundary.delete();
  reconstructed.delete();

  warpedMats.forEach(
    matrix =>
      matrix.delete()
  );

  referencePoints.delete();
}

