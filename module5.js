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
// LOAD PRELOADED VIDEOS
// ============================================================

document.querySelectorAll(".load-video").forEach(button => {
  button.addEventListener("click", () => {
    loadVideo(button.dataset.video);
  });
});


// ============================================================
// UPLOAD VIDEO
// ============================================================

$("videoUpload").addEventListener("change", event => {
  const file = event.target.files[0];

  if (!file) {
    return;
  }

  const url = URL.createObjectURL(file);

  loadVideo(url);
});


// ============================================================
// LOAD VIDEO
// ============================================================

function loadVideo(source) {
  stopFlow();

  const video = $("sourceVideo");

  video.src = source;
  video.load();

  video.onloadedmetadata = () => {
    $("videoInfo").innerHTML = [
      ["Duration", `${video.duration.toFixed(2)} sec`],
      ["Resolution", `${video.videoWidth} × ${video.videoHeight}`],
      ["Status", "Ready"],
      ["Method", "Motion Mask + Lucas-Kanade"]
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

    const flowCanvas = $("flowCanvas");

    flowCanvas.width = video.videoWidth;
    flowCanvas.height = video.videoHeight;

    $("validation1").width = video.videoWidth;
    $("validation1").height = video.videoHeight;

    $("validation2").width = video.videoWidth;
    $("validation2").height = video.videoHeight;
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

  if (flowTimer) {
    cancelAnimationFrame(
      flowTimer
    );
  }

  flowTimer = null;

  const video =
    $("sourceVideo");

  if (
    video &&
    !video.paused
  ) {
    video.pause();
  }

  clearFlowMats();
}


// ============================================================
// CAPTURE VALIDATION PAIR
// ============================================================

function captureValidationPair() {
  if (
    !latestOldFrame ||
    !latestNewFrame ||
    latestOldPoints.length === 0
  ) {
    alert(
      "Start optical flow first and wait until moving points are tracked."
    );

    return;
  }

  const canvas1 =
    $("validation1");

  const canvas2 =
    $("validation2");

  const context1 =
    canvas1.getContext("2d");

  const context2 =
    canvas2.getContext("2d");

  // Show CLEAN underlying frames.
  context1.putImageData(
    latestOldFrame,
    0,
    0
  );

  context2.putImageData(
    latestNewFrame,
    0,
    0
  );

  // Compute displacement and choose largest-moving points.
  const rows =
    latestOldPoints
      .map(
        (
          oldPoint,
          index
        ) => {
          const newPoint =
            latestNewPoints[index];

          const dx =
            newPoint.x -
            oldPoint.x;

          const dy =
            newPoint.y -
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
        (a, b) =>
          b.magnitude -
          a.magnitude
      )
      .slice(
        0,
        10
      );

  rows.forEach(
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
    rows
      .map(
        (
          row,
          index
        ) => `
          <tr>
            <td>${index + 1}</td>

            <td>
              ${row.oldPoint.x.toFixed(2)}
            </td>

            <td>
              ${row.oldPoint.y.toFixed(2)}
            </td>

            <td>
              ${row.newPoint.x.toFixed(2)}
            </td>

            <td>
              ${row.newPoint.y.toFixed(2)}
            </td>

            <td>
              ${row.dx.toFixed(2)}
            </td>

            <td>
              ${row.dy.toFixed(2)}
            </td>

            <td>
              ${row.magnitude.toFixed(2)}
            </td>
          </tr>
        `
      )
      .join("");
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
// PLANAR RECONSTRUCTION
// ============================================================

function reconstructBook() {
  if (!cvReady) {
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


  // --------------------------------------------------------
  // Reference plane
  // --------------------------------------------------------

  const referenceWidth =
    600;

  const referenceHeight =
    400;

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


  const warpedMats = [];
  const homographyMatrices = [];

  const rectifiedContainer =
    $("rectifiedViews");

  rectifiedContainer.innerHTML =
    "";

  const coordinateRows = [];


  // ========================================================
  // HOMOGRAPHY FOR EACH VIEW
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
      // Display rectified image
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
      // Coordinates
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
  // MEDIAN FUSION
  //
  // This replaces simple averaging.
  // It reduces ghosting from small alignment differences.
  // ========================================================

  const reconstructed =
    new cv.Mat(
      referenceHeight,
      referenceWidth,
      cv.CV_8UC4
    );


  const totalPixels =
    referenceWidth *
    referenceHeight;


  for (
    let pixelIndex = 0;
    pixelIndex < totalPixels;
    pixelIndex++
  ) {
    const base =
      pixelIndex * 4;


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
            values[1] +
            values[2]
          )
          /
          2
        );
    }
  }


  // ========================================================
  // DISPLAY RECONSTRUCTED OBJECT
  // ========================================================

  cv.imshow(
    "reconstructedCanvas",
    reconstructed
  );


  // ========================================================
  // DRAW RECONSTRUCTED BOUNDARY
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
