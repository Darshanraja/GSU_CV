// ============================================================
// Module 5 - Static Vercel-compatible browser implementation
// ============================================================

const $ = id => document.getElementById(id);

let cvReady = false;
let flowRunning = false;
let flowTimer = null;
let previousGray = null;
let previousPoints = null;
let previousFrameImageData = null;
let latestOldPoints = [];
let latestNewPoints = [];
let latestOldFrame = null;
let latestNewFrame = null;

const FLOW_FEATURE_PARAMS = {
  maxCorners: 100,
  qualityLevel: 0.1,
  minDistance: 5,
  blockSize: 7
};

window.addEventListener("load", () => {
  // Tabs
  document.querySelectorAll(".tab").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach(x => x.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach(x => x.classList.remove("active-panel"));
      btn.classList.add("active");
      $(btn.dataset.tab).classList.add("active-panel");
    });
  });

  // OpenCV readiness poll because async script timing differs by browser.
  const timer = setInterval(() => {
    if (window.cv && cv.Mat) {
      cvReady = true;
      clearInterval(timer);
      $("opencvStatus").textContent = "OpenCV.js ready";
    }
  }, 250);
});

// ============================================================
// Problem 1 - Optical Flow
// ============================================================

document.querySelectorAll(".load-video").forEach(btn => {
  btn.addEventListener("click", () => loadVideo(btn.dataset.video));
});

$("videoUpload").addEventListener("change", e => {
  const file = e.target.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  loadVideo(url, true);
});

function loadVideo(src, objectUrl=false) {
  stopFlow();
  const video = $("sourceVideo");
  video.src = src;
  video.load();

  video.onloadedmetadata = () => {
    $("videoInfo").innerHTML = [
      ["Duration", `${video.duration.toFixed(2)} sec`],
      ["Resolution", `${video.videoWidth} × ${video.videoHeight}`],
      ["Status", "Ready"],
      ["Mode", "Lucas-Kanade"]
    ].map(([a,b]) => `<div class="info-box">${a}<b>${b}</b></div>`).join("");

    const canvas = $("flowCanvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    $("validation1").width = video.videoWidth;
    $("validation1").height = video.videoHeight;
    $("validation2").width = video.videoWidth;
    $("validation2").height = video.videoHeight;
  };
}

$("startFlow").addEventListener("click", startFlow);
$("stopFlow").addEventListener("click", stopFlow);
$("captureValidation").addEventListener("click", captureValidationPair);

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

  const canvas = $("flowCanvas");
  const ctx = canvas.getContext("2d");

  if (!canvas.width) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }

  await video.play();
  flowRunning = true;

  // reset matrices
  if (previousGray) previousGray.delete();
  if (previousPoints) previousPoints.delete();
  previousGray = null;
  previousPoints = null;

  function processFrame() {
    if (!flowRunning || video.paused || video.ended) {
      if (video.ended) stopFlow();
      return;
    }

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const frame = cv.imread(canvas);
    const gray = new cv.Mat();
    cv.cvtColor(frame, gray, cv.COLOR_RGBA2GRAY);

    // Detect new features if needed.
    if (!previousGray || !previousPoints || previousPoints.rows < 6) {
      if (previousGray) previousGray.delete();
      if (previousPoints) previousPoints.delete();

      previousGray = gray.clone();

      const corners = new cv.Mat();
      cv.goodFeaturesToTrack(
        gray,
        corners,
        FLOW_FEATURE_PARAMS.maxCorners,
        FLOW_FEATURE_PARAMS.qualityLevel,
        FLOW_FEATURE_PARAMS.minDistance,
        new cv.Mat(),
        FLOW_FEATURE_PARAMS.blockSize,
        false,
        0.04
      );
      previousPoints = corners;
      previousFrameImageData = ctx.getImageData(0,0,canvas.width,canvas.height);

      frame.delete();
      gray.delete();
      flowTimer = requestAnimationFrame(processFrame);
      return;
    }

    const nextPts = new cv.Mat();
    const status = new cv.Mat();
    const err = new cv.Mat();
    const winSize = new cv.Size(21,21);
    const criteria = new cv.TermCriteria(
      cv.TermCriteria_EPS | cv.TermCriteria_COUNT,
      30,
      0.01
    );

    cv.calcOpticalFlowPyrLK(
      previousGray,
      gray,
      previousPoints,
      nextPts,
      status,
      err,
      winSize,
      3,
      criteria
    );

    latestOldPoints = [];
    latestNewPoints = [];

    // Draw optical-flow vectors.
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    for (let i=0; i<status.rows; i++) {
      if (status.data[i] !== 1) continue;

      const x1 = previousPoints.data32F[i*2];
      const y1 = previousPoints.data32F[i*2+1];
      const x2 = nextPts.data32F[i*2];
      const y2 = nextPts.data32F[i*2+1];

      latestOldPoints.push({x:x1,y:y1});
      latestNewPoints.push({x:x2,y:y2});

      ctx.strokeStyle = "#00ff55";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x1,y1);
      ctx.lineTo(x2,y2);
      ctx.stroke();

      // arrow head
      const ang = Math.atan2(y2-y1,x2-x1);
      const len = 7;
      ctx.beginPath();
      ctx.moveTo(x2,y2);
      ctx.lineTo(x2-len*Math.cos(ang-.5), y2-len*Math.sin(ang-.5));
      ctx.moveTo(x2,y2);
      ctx.lineTo(x2-len*Math.cos(ang+.5), y2-len*Math.sin(ang+.5));
      ctx.stroke();

      ctx.fillStyle = "#ff3030";
      ctx.beginPath();
      ctx.arc(x2,y2,3.5,0,Math.PI*2);
      ctx.fill();
    }

    latestOldFrame = previousFrameImageData;
    latestNewFrame = ctx.getImageData(0,0,canvas.width,canvas.height);

    // Keep valid points only.
    const valid = latestNewPoints;
    if (previousPoints) previousPoints.delete();

    if (valid.length > 0) {
      const arr = [];
      valid.forEach(p => { arr.push(p.x,p.y); });
      previousPoints = cv.matFromArray(valid.length,1,cv.CV_32FC2,arr);
    } else {
      previousPoints = new cv.Mat();
    }

    previousGray.delete();
    previousGray = gray.clone();
    previousFrameImageData = latestNewFrame;

    frame.delete();
    gray.delete();
    nextPts.delete();
    status.delete();
    err.delete();

    flowTimer = requestAnimationFrame(processFrame);
  }

  processFrame();
}

function stopFlow() {
  flowRunning = false;
  if (flowTimer) cancelAnimationFrame(flowTimer);
  flowTimer = null;

  const video = $("sourceVideo");
  if (video && !video.paused) video.pause();
}

function captureValidationPair() {
  if (!latestOldFrame || !latestNewFrame || latestOldPoints.length === 0) {
    alert("Start optical flow first and wait until points are tracked.");
    return;
  }

  const c1 = $("validation1");
  const c2 = $("validation2");
  const ctx1 = c1.getContext("2d");
  const ctx2 = c2.getContext("2d");

  ctx1.putImageData(latestOldFrame,0,0);
  ctx2.putImageData(latestNewFrame,0,0);

  const rows = latestOldPoints.map((p,i) => {
    const q = latestNewPoints[i];
    const dx = q.x-p.x;
    const dy = q.y-p.y;
    return {i,p,q,dx,dy,mag:Math.hypot(dx,dy)};
  }).sort((a,b)=>b.mag-a.mag).slice(0,10);

  rows.forEach((r,idx) => {
    drawNumberedPoint(ctx1,r.p.x,r.p.y,idx+1,"#00ff55");
    drawNumberedPoint(ctx2,r.q.x,r.q.y,idx+1,"#ff3030");
  });

  $("validationTable").innerHTML = rows.map((r,idx)=>`
    <tr>
      <td>${idx+1}</td>
      <td>${r.p.x.toFixed(2)}</td>
      <td>${r.p.y.toFixed(2)}</td>
      <td>${r.q.x.toFixed(2)}</td>
      <td>${r.q.y.toFixed(2)}</td>
      <td>${r.dx.toFixed(2)}</td>
      <td>${r.dy.toFixed(2)}</td>
      <td>${r.mag.toFixed(2)}</td>
    </tr>
  `).join("");
}

function drawNumberedPoint(ctx,x,y,n,color) {
  ctx.fillStyle=color;
  ctx.beginPath();
  ctx.arc(x,y,7,0,Math.PI*2);
  ctx.fill();
  ctx.fillStyle="#111";
  ctx.font="bold 14px Arial";
  ctx.fillText(String(n),x+9,y-6);
}

// ============================================================
// Problem 2 - Four-corner planar homography reconstruction
// ============================================================

const PRELOADED_BOOK_VIEWS = [
  "module5_assets/problem2/view1.png",
  "module5_assets/problem2/view2.png",
  "module5_assets/problem2/view3.png",
  "module5_assets/problem2/view4.png"
];

let bookImages = []; // {img, src, points:[]}

$("loadBookViews").addEventListener("click", async () => {
  bookImages = [];
  for (const src of PRELOADED_BOOK_VIEWS) {
    const img = await loadImage(src);
    bookImages.push({img,src,points:[]});
  }
  renderBookViews();
});

$("imageUpload").addEventListener("change", async e => {
  const files=[...e.target.files];
  if (files.length !== 4) {
    alert("Select exactly four images.");
    return;
  }
  bookImages=[];
  for (const f of files) {
    const src=URL.createObjectURL(f);
    const img=await loadImage(src);
    bookImages.push({img,src,points:[]});
  }
  renderBookViews();
});

$("clearBookViews").addEventListener("click", () => {
  bookImages=[];
  $("bookViews").innerHTML="";
  $("homographyOutput").classList.add("hidden");
});

function loadImage(src) {
  return new Promise((resolve,reject)=>{
    const img=new Image();
    img.onload=()=>resolve(img);
    img.onerror=reject;
    img.src=src;
  });
}

function renderBookViews() {
  const host=$("bookViews");
  host.innerHTML="";

  bookImages.forEach((item,index)=>{
    const card=document.createElement("div");
    card.className="view-card";
    card.innerHTML=`
      <h3>View ${index+1}</h3>
      <canvas></canvas>
      <div class="view-instructions">
        Click: 1 Top Left → 2 Top Right → 3 Bottom Right → 4 Bottom Left.
        Click "Reset Points" if needed.
      </div>
      <button class="secondary reset-view" style="margin-top:8px">Reset Points</button>
    `;
    host.appendChild(card);

    const canvas=card.querySelector("canvas");
    const ctx=canvas.getContext("2d");
    const img=item.img;

    canvas.width=img.naturalWidth;
    canvas.height=img.naturalHeight;

    function redraw() {
      ctx.drawImage(img,0,0,canvas.width,canvas.height);

      if (item.points.length>1) {
        ctx.strokeStyle="#00ff55";
        ctx.lineWidth=Math.max(3,canvas.width/500);
        ctx.beginPath();
        ctx.moveTo(item.points[0].x,item.points[0].y);
        for (let i=1;i<item.points.length;i++) ctx.lineTo(item.points[i].x,item.points[i].y);
        if (item.points.length===4) ctx.closePath();
        ctx.stroke();
      }

      item.points.forEach((p,i)=>{
        ctx.fillStyle="#ff3030";
        ctx.beginPath();
        ctx.arc(p.x,p.y,Math.max(7,canvas.width/250),0,Math.PI*2);
        ctx.fill();
        ctx.fillStyle="#00ff55";
        ctx.font=`bold ${Math.max(18,canvas.width/80)}px Arial`;
        ctx.fillText(String(i+1),p.x+12,p.y-10);
      });
    }

    redraw();

    canvas.addEventListener("click",e=>{
      if (item.points.length>=4) return;
      const r=canvas.getBoundingClientRect();
      const x=(e.clientX-r.left)*canvas.width/r.width;
      const y=(e.clientY-r.top)*canvas.height/r.height;
      item.points.push({x,y});
      redraw();
    });

    card.querySelector(".reset-view").addEventListener("click",()=>{
      item.points=[];
      redraw();
    });
  });
}

$("reconstructBook").addEventListener("click", reconstructBook);

function reconstructBook() {
  if (!cvReady) {
    alert("OpenCV.js is still loading.");
    return;
  }
  if (bookImages.length!==4) {
    alert("Load four images first.");
    return;
  }
  if (bookImages.some(v=>v.points.length!==4)) {
    alert("Select four corners in every view first.");
    return;
  }

  const refW=600, refH=400;
  const refPts = cv.matFromArray(4,1,cv.CV_32FC2,[
    0,0,
    refW-1,0,
    refW-1,refH-1,
    0,refH-1
  ]);

  const warpedMats=[];
  const matrices=[];
  const rectifiedHost=$("rectifiedViews");
  rectifiedHost.innerHTML="";
  const coordRows=[];

  bookImages.forEach((item,index)=>{
    const srcCanvas=document.createElement("canvas");
    srcCanvas.width=item.img.naturalWidth;
    srcCanvas.height=item.img.naturalHeight;
    srcCanvas.getContext("2d").drawImage(item.img,0,0);

    const src=cv.imread(srcCanvas);

    const srcPts=cv.matFromArray(4,1,cv.CV_32FC2,item.points.flatMap(p=>[p.x,p.y]));
    const H=cv.getPerspectiveTransform(srcPts,refPts);

    matrices.push(Array.from(H.data64F));

    const dst=new cv.Mat();
    cv.warpPerspective(
      src,dst,H,new cv.Size(refW,refH),
      cv.INTER_LINEAR,
      cv.BORDER_CONSTANT,
      new cv.Scalar()
    );
    warpedMats.push(dst);

    const card=document.createElement("div");
    card.className="view-card";
    card.innerHTML=`<h3>Rectified View ${index+1}</h3><canvas></canvas>`;
    rectifiedHost.appendChild(card);
    cv.imshow(card.querySelector("canvas"),dst);

    const names=["Top Left","Top Right","Bottom Right","Bottom Left"];
    item.points.forEach((p,j)=>{
      coordRows.push([index+1,j+1,names[j],p.x,p.y]);
    });

    src.delete();
    srcPts.delete();
    H.delete();
  });

  // Average the four rectified views manually in RGBA.
  const avg = new cv.Mat(refH,refW,cv.CV_8UC4);
  const total = refW*refH*4;
  for (let i=0;i<total;i++) {
    let s=0;
    for (const m of warpedMats) s += m.data[i];
    avg.data[i]=Math.round(s/warpedMats.length);
  }

  cv.imshow("reconstructedCanvas",avg);

  const boundary=avg.clone();
  cv.rectangle(
    boundary,
    new cv.Point(1,1),
    new cv.Point(refW-2,refH-2),
    new cv.Scalar(0,255,0,255),
    5
  );
  cv.imshow("boundaryCanvas",boundary);

  $("homographyMatrices").textContent = matrices.map((m,i)=>{
    const rows=[
      m.slice(0,3),
      m.slice(3,6),
      m.slice(6,9)
    ];
    return `View ${i+1}\n` + rows.map(r=>r.map(v=>v.toExponential(6)).join("   ")).join("\n");
  }).join("\n\n");

  $("pointCoordinates").innerHTML=coordRows.map(r=>`
    <tr>
      <td>${r[0]}</td>
      <td>${r[1]}</td>
      <td>${r[2]}</td>
      <td>${r[3].toFixed(2)}</td>
      <td>${r[4].toFixed(2)}</td>
    </tr>
  `).join("");

  $("homographyOutput").classList.remove("hidden");

  boundary.delete();
  avg.delete();
  warpedMats.forEach(m=>m.delete());
  refPts.delete();
}
