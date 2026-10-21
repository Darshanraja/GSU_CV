const $ = id => document.getElementById(id);

// ---------- Tabs ----------
document.querySelectorAll('.tab-button').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-button').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(x => x.classList.remove('active'));
    btn.classList.add('active');
    $(btn.dataset.tab).classList.add('active');
  });
});

// ---------- Calibration ----------
const calIds = ['fx','fy','cx','cy','calibWidth','calibHeight','k1','k2','p1','p2','k3','baseline'];
function getCalibration(frameW, frameH) {
  const base = Object.fromEntries(calIds.map(id => [id, Number($(id).value)]));
  const sx = frameW / base.calibWidth;
  const sy = frameH / base.calibHeight;
  return {
    fx: base.fx * sx,
    fy: base.fy * sy,
    cx: base.cx * sx,
    cy: base.cy * sy,
    k1: base.k1, k2: base.k2, p1: base.p1, p2: base.p2, k3: base.k3,
    baseline: base.baseline
  };
}
$('saveCalibration').onclick = () => {
  const obj = Object.fromEntries(calIds.map(id => [id, $(id).value]));
  localStorage.setItem('assignment7Calibration', JSON.stringify(obj));
  alert('Calibration saved in this browser.');
};
$('restoreCalibration').onclick = () => {
  const raw = localStorage.getItem('assignment7Calibration');
  if (!raw) return alert('No saved calibration found.');
  const obj = JSON.parse(raw);
  calIds.forEach(id => { if (obj[id] !== undefined) $(id).value = obj[id]; });
};

// Brown-Conrady inverse distortion by fixed-point iteration.
function undistortPixel(pt, c) {
  const xd = (pt.x - c.cx) / c.fx;
  const yd = (pt.y - c.cy) / c.fy;
  let x = xd, y = yd;
  for (let i = 0; i < 8; i++) {
    const r2 = x*x + y*y;
    const radial = 1 + c.k1*r2 + c.k2*r2*r2 + c.k3*r2*r2*r2;
    const dx = 2*c.p1*x*y + c.p2*(r2 + 2*x*x);
    const dy = c.p1*(r2 + 2*y*y) + 2*c.p2*x*y;
    x = (xd - dx) / radial;
    y = (yd - dy) / radial;
  }
  return { x: c.fx*x + c.cx, y: c.fy*y + c.cy };
}
function reconstruct(leftRaw, rightRaw, c) {
  const l = undistortPixel(leftRaw, c);
  const r = undistortPixel(rightRaw, c);
  const signed = l.x - r.x;
  const d = Math.abs(signed);
  if (d < 0.05) throw new Error('Disparity is too small. Select the same physical point in both views.');
  const Z = c.fx * c.baseline / d;
  const X = (l.x - c.cx) * Z / c.fx;
  const Y = (l.y - c.cy) * Z / c.fy;
  return {X,Y,Z,d,signed,l,r};
}
const dist3 = (a,b) => Math.hypot(a.X-b.X, a.Y-b.Y, a.Z-b.Z);
const avg = arr => arr.reduce((s,v)=>s+v,0)/arr.length;

// ---------- Reusable point canvas ----------
function pointCanvas(canvasId, statusId, pointColor='#16a34a') {
  const canvas = $(canvasId), ctx = canvas.getContext('2d'), status = $(statusId);
  let base = null, points = [], count = 4, labels = ['TL','TR','BR','BL'], close = true;
  function setConfig(n, labs, closeShape) { count=n; labels=labs; close=closeShape; points=[]; redraw(); updateStatus(); }
  function updateStatus() { status.textContent = base ? `${points.length}/${count} points selected.` : 'No image/frame loaded.'; }
  function redraw() {
    ctx.clearRect(0,0,canvas.width,canvas.height);
    if (base) ctx.drawImage(base,0,0,canvas.width,canvas.height);
    if (points.length > 1) {
      ctx.beginPath(); ctx.moveTo(points[0].x, points[0].y);
      for (let i=1;i<points.length;i++) ctx.lineTo(points[i].x, points[i].y);
      if (close && points.length===count && count>=3) ctx.closePath();
      ctx.lineWidth=3; ctx.strokeStyle='#facc15'; ctx.stroke();
    }
    points.forEach((p,i)=>{
      ctx.beginPath(); ctx.arc(p.x,p.y,7,0,Math.PI*2); ctx.fillStyle=pointColor; ctx.fill();
      ctx.lineWidth=2; ctx.strokeStyle='white'; ctx.stroke();
      ctx.font='bold 16px sans-serif'; ctx.fillStyle='white'; ctx.fillText(labels[i] || `${i+1}`, p.x+10, p.y-10);
    });
  }
  function setImageElement(img) {
    const maxSide = 1000;
    const scale = Math.min(1, maxSide / Math.max(img.naturalWidth || img.videoWidth, img.naturalHeight || img.videoHeight));
    const w = Math.round((img.naturalWidth || img.videoWidth)*scale);
    const h = Math.round((img.naturalHeight || img.videoHeight)*scale);
    canvas.width=w; canvas.height=h;
    const temp=document.createElement('canvas'); temp.width=w; temp.height=h;
    temp.getContext('2d').drawImage(img,0,0,w,h);
    const copy=new Image();
    copy.onload=()=>{ base=copy; points=[]; redraw(); updateStatus(); };
    copy.src=temp.toDataURL('image/jpeg',0.95);
  }
  function loadUrl(url) { return new Promise((resolve,reject)=>{ const img=new Image(); img.onload=()=>{setImageElement(img);resolve();}; img.onerror=reject; img.src=url; }); }
  function loadFile(file) { return new Promise((resolve,reject)=>{ const url=URL.createObjectURL(file); const img=new Image(); img.onload=()=>{setImageElement(img);URL.revokeObjectURL(url);resolve();}; img.onerror=reject; img.src=url; }); }
  canvas.addEventListener('click', e=>{
    if (!base || points.length>=count) return;
    const r=canvas.getBoundingClientRect();
    points.push({x:(e.clientX-r.left)*canvas.width/r.width, y:(e.clientY-r.top)*canvas.height/r.height});
    redraw(); updateStatus();
  });
  return {
    setConfig, loadUrl, loadFile, setVideoFrame: setImageElement,
    clear(){points=[];redraw();updateStatus();},
    getPoints(){return points.map(p=>({...p}));},
    getSize(){return {width:canvas.width,height:canvas.height};},
    hasImage(){return !!base;}
  };
}

// ---------- Problem 1 ----------
const p1Left = pointCanvas('p1LeftCanvas','p1LeftStatus','#16a34a');
const p1Right = pointCanvas('p1RightCanvas','p1RightStatus','#dc2626');
const preload = {
  speaker: {
    name: 'Speaker',
    left: 'assets/left_speaker.jpg',
    right: 'assets/right_speaker.jpg',
    shape: 'rectangle',
    actual: { width: 8.0, height: 9.5 }
  },
  circle: {
    name: 'Circular object',
    left: 'assets/left_circle.jpg',
    right: 'assets/right_circle.jpg',
    shape: 'circle',
    actual: { diameter: 4.0 }
  },
  polygon: {
    name: 'TUMS bottle - Polygon',
    left: 'assets/left_polygon.jpg',
    right: 'assets/right_polygon.jpg',
    shape: 'polygon',
    actual: { width: 5.5, length: 12.5 }
  }
};

function percentError(estimated, actual) {
  return Math.abs(estimated - actual) / actual * 100;
}

function updateGroundTruth() {
  const key = $('preloadedObject').value;
  const item = preload[key];
  if (!item) return;
  $('groundTruthObject').textContent = item.name;
  if (key === 'speaker') {
    $('groundTruthValues').innerHTML = 'Width = <strong>8.00 cm</strong> &nbsp;|&nbsp; Height = <strong>9.50 cm</strong>';
  } else if (key === 'circle') {
    $('groundTruthValues').innerHTML = 'Diameter = <strong>4.00 cm</strong>';
  } else {
    $('groundTruthValues').innerHTML = 'Width = <strong>5.50 cm</strong> &nbsp;|&nbsp; Length = <strong>12.50 cm</strong>';
  }
}

// Estimate the two principal dimensions of a planar polygon in 3D.
// This is used only for validating the preloaded polygon's overall width/length;
// individual polygon edge lengths are still reported above as required by the assignment.
function polygonPrincipalDimensions(pts) {
  if (pts.length < 3) return { width: NaN, length: NaN };
  const center = {
    X: avg(pts.map(p => p.X)),
    Y: avg(pts.map(p => p.Y)),
    Z: avg(pts.map(p => p.Z))
  };
  const centered = pts.map(p => [p.X-center.X, p.Y-center.Y, p.Z-center.Z]);

  // First principal direction by power iteration on the 3x3 covariance matrix.
  const C = [[0,0,0],[0,0,0],[0,0,0]];
  centered.forEach(v => {
    for (let i=0;i<3;i++) for (let j=0;j<3;j++) C[i][j] += v[i]*v[j];
  });
  let e1 = [1,0,0];
  for (let k=0;k<24;k++) {
    const w = [
      C[0][0]*e1[0]+C[0][1]*e1[1]+C[0][2]*e1[2],
      C[1][0]*e1[0]+C[1][1]*e1[1]+C[1][2]*e1[2],
      C[2][0]*e1[0]+C[2][1]*e1[1]+C[2][2]*e1[2]
    ];
    const n = Math.hypot(...w) || 1;
    e1 = w.map(x => x/n);
  }

  // Remove the first component and find a second in-plane principal direction.
  const C2 = C.map((row,i) => row.map((val,j) => {
    const lambda = e1.reduce((s,x,r)=>s+x*C[r][j],0);
    return val - lambda*e1[i];
  }));
  let e2 = [0,1,0];
  // Orthogonalize initial vector.
  let dot = e2[0]*e1[0]+e2[1]*e1[1]+e2[2]*e1[2];
  e2 = e2.map((x,i)=>x-dot*e1[i]);
  let en = Math.hypot(...e2);
  if (en < 1e-8) e2 = [0,0,1]; else e2 = e2.map(x=>x/en);
  for (let k=0;k<24;k++) {
    let w = [
      C2[0][0]*e2[0]+C2[0][1]*e2[1]+C2[0][2]*e2[2],
      C2[1][0]*e2[0]+C2[1][1]*e2[1]+C2[1][2]*e2[2],
      C2[2][0]*e2[0]+C2[2][1]*e2[1]+C2[2][2]*e2[2]
    ];
    const d = w[0]*e1[0]+w[1]*e1[1]+w[2]*e1[2];
    w = w.map((x,i)=>x-d*e1[i]);
    const n = Math.hypot(...w) || 1;
    e2 = w.map(x=>x/n);
  }

  const p1 = centered.map(v => v[0]*e1[0]+v[1]*e1[1]+v[2]*e1[2]);
  const p2 = centered.map(v => v[0]*e2[0]+v[1]*e2[1]+v[2]*e2[2]);
  const dim1 = Math.max(...p1)-Math.min(...p1);
  const dim2 = Math.max(...p2)-Math.min(...p2);
  return { width: Math.min(dim1,dim2), length: Math.max(dim1,dim2) };
}

function renderValidation(rows, objectName) {
  $('p1ValidationObject').textContent = objectName;
  $('p1ValidationTable').innerHTML = rows.map(row => {
    const absErr = Math.abs(row.estimated - row.actual);
    const pctErr = percentError(row.estimated, row.actual);
    return `<tr>
      <td>${row.label}</td>
      <td>${row.estimated.toFixed(2)} cm</td>
      <td>${row.actual.toFixed(2)} cm</td>
      <td>${absErr.toFixed(2)} cm</td>
      <td><strong>${pctErr.toFixed(2)}%</strong></td>
    </tr>`;
  }).join('');
}
function updateShape() {
  const s=$('shapeType').value;
  let n=4, labs=['TL','TR','BR','BL'], close=true, title='Rectangle point order', text='Click TL, TR, BR, BL on LEFT, then the exact same physical corners in the same order on RIGHT.';
  $('polygonCountWrap').classList.add('hidden');
  if (s==='circle') { n=2; labs=['D1','D2']; close=false; title='Circle diameter'; text='Click two opposite points across the visible diameter: D1 then D2. Repeat the exact same physical points on RIGHT.'; }
  if (s==='polygon') { $('polygonCountWrap').classList.remove('hidden'); n=Math.max(3,Number($('polygonCount').value)||3); labs=Array.from({length:n},(_,i)=>`P${i+1}`); title='Polygon vertex order'; text='Click every visible polygon vertex around the boundary in order. Repeat the same vertices in the same order on RIGHT.'; }
  p1Left.setConfig(n,labs,close); p1Right.setConfig(n,labs,close);
  $('pointInstructionTitle').textContent=title; $('pointInstruction').textContent=text;
}
async function loadPreloadedObject() {
  const key=$('preloadedObject').value, item=preload[key];
  $('shapeType').value=item.shape; if(item.shape==='polygon') $('polygonCount').value=8;
  updateShape();
  updateGroundTruth();
  await Promise.all([p1Left.loadUrl(item.left), p1Right.loadUrl(item.right)]);
}
$('shapeType').onchange=updateShape;
$('polygonCount').onchange=updateShape;
$('preloadedObject').onchange=loadPreloadedObject;
$('p1Source').onchange=async()=>{
  const v=$('p1Source').value;
  $('uploadControls').classList.toggle('hidden',v!=='upload');
  $('cameraControls').classList.toggle('hidden',v!=='camera');
  $('preloadedObjectWrap').classList.toggle('hidden',v!=='preloaded');
  $('groundTruthCard').classList.toggle('hidden',v!=='preloaded');
  if(v==='preloaded') await loadPreloadedObject();
};
$('leftUpload').onchange=e=>e.target.files[0]&&p1Left.loadFile(e.target.files[0]);
$('rightUpload').onchange=e=>e.target.files[0]&&p1Right.loadFile(e.target.files[0]);
$('clearP1Points').onclick=()=>{p1Left.clear();p1Right.clear();$('p1Results').classList.add('hidden');};

let cameraStream=null;
async function refreshCameras() {
  try {
    const temp=await navigator.mediaDevices.getUserMedia({video:true,audio:false});
    temp.getTracks().forEach(t=>t.stop());
    const devices=(await navigator.mediaDevices.enumerateDevices()).filter(d=>d.kind==='videoinput');
    $('cameraSelect').innerHTML=devices.map((d,i)=>`<option value="${d.deviceId}">${d.label||`Camera ${i+1}`}</option>`).join('');
  } catch(err) { alert('Camera permission is required: '+err.message); }
}
$('refreshCameras').onclick=refreshCameras;
$('startCamera').onclick=async()=>{
  if(cameraStream) cameraStream.getTracks().forEach(t=>t.stop());
  const id=$('cameraSelect').value;
  cameraStream=await navigator.mediaDevices.getUserMedia({video:id?{deviceId:{exact:id}}:true,audio:false});
  $('cameraPreview').srcObject=cameraStream;
};
$('stopCamera').onclick=()=>{ if(cameraStream){cameraStream.getTracks().forEach(t=>t.stop());cameraStream=null;$('cameraPreview').srcObject=null;} };
$('captureLeft').onclick=()=>{ if(!$('cameraPreview').videoWidth) return alert('Start the camera first.'); p1Left.setVideoFrame($('cameraPreview')); };
$('captureRight').onclick=()=>{ if(!$('cameraPreview').videoWidth) return alert('Start the camera first.'); p1Right.setVideoFrame($('cameraPreview')); };

$('calculateP1').onclick=()=>{
  try {
    const lp=p1Left.getPoints(), rp=p1Right.getPoints();
    if(!p1Left.hasImage()||!p1Right.hasImage()) throw new Error('Load or capture both LEFT and RIGHT images.');
    if(lp.length===0||lp.length!==rp.length) throw new Error('Select the same number of corresponding points in both views.');
    const size=p1Left.getSize(), sizeR=p1Right.getSize();
    if(size.width!==sizeR.width||size.height!==sizeR.height) throw new Error('LEFT and RIGHT displayed images must use the same resolution.');
    const c=getCalibration(size.width,size.height);
    const pts=lp.map((p,i)=>reconstruct(p,rp[i],c));
    $('p1MeanDisparity').textContent=`${avg(pts.map(p=>p.d)).toFixed(2)} px`;
    $('p1MeanZ').textContent=`${avg(pts.map(p=>p.Z)).toFixed(2)} cm`;
    $('p1Baseline').textContent=`${c.baseline.toFixed(2)} cm`;
    $('p1Fx').textContent=`${c.fx.toFixed(2)} px`;
    const labels=$('shapeType').value==='rectangle'?['TL','TR','BR','BL']:$('shapeType').value==='circle'?['D1','D2']:Array.from({length:pts.length},(_,i)=>`P${i+1}`);
    $('p1PointTable').innerHTML=pts.map((p,i)=>`<tr><td>${labels[i]}</td><td>${lp[i].x.toFixed(2)}</td><td>${lp[i].y.toFixed(2)}</td><td>${rp[i].x.toFixed(2)}</td><td>${rp[i].y.toFixed(2)}</td><td>${p.d.toFixed(2)}</td><td>${p.X.toFixed(2)}</td><td>${p.Y.toFixed(2)}</td><td>${p.Z.toFixed(2)}</td></tr>`).join('');
    let html='';
    let validationRows=[];
    const objectKey=$('preloadedObject').value;
    const objectInfo=preload[objectKey];

    if($('shapeType').value==='rectangle') {
      if(pts.length!==4) throw new Error('Rectangle requires exactly four points.');
      const top=dist3(pts[0],pts[1]), right=dist3(pts[1],pts[2]), bottom=dist3(pts[3],pts[2]), left=dist3(pts[0],pts[3]);
      const estimatedWidth=(top+bottom)/2;
      const estimatedHeight=(left+right)/2;
      html=`Top edge = <strong>${top.toFixed(2)} cm</strong><br>Bottom edge = <strong>${bottom.toFixed(2)} cm</strong><br>Left edge = <strong>${left.toFixed(2)} cm</strong><br>Right edge = <strong>${right.toFixed(2)} cm</strong><br><br>Estimated width = <strong>${estimatedWidth.toFixed(2)} cm</strong><br>Estimated height = <strong>${estimatedHeight.toFixed(2)} cm</strong>`;
      if(objectKey==='speaker') {
        validationRows=[
          {label:'Width',estimated:estimatedWidth,actual:8.0},
          {label:'Height',estimated:estimatedHeight,actual:9.5}
        ];
      }
    } else if($('shapeType').value==='circle') {
      const diameter=dist3(pts[0],pts[1]);
      html=`Estimated diameter = <strong>${diameter.toFixed(2)} cm</strong>`;
      if(objectKey==='circle') validationRows=[{label:'Diameter',estimated:diameter,actual:4.0}];
    } else {
      html='';
      for(let i=0;i<pts.length;i++){
        const j=(i+1)%pts.length;
        html+=`Edge P${i+1}-P${j+1} = <strong>${dist3(pts[i],pts[j]).toFixed(2)} cm</strong><br>`;
      }
      const dims=polygonPrincipalDimensions(pts);
      html+=`<br>Overall polygon width = <strong>${dims.width.toFixed(2)} cm</strong><br>Overall polygon length = <strong>${dims.length.toFixed(2)} cm</strong>`;
      if(objectKey==='polygon') {
        validationRows=[
          {label:'Width',estimated:dims.width,actual:5.5},
          {label:'Length',estimated:dims.length,actual:12.5}
        ];
      }
    }

    $('p1Dimensions').innerHTML=html;
    if(validationRows.length) {
      renderValidation(validationRows, objectInfo?.name || 'Object');
      $('p1ValidationCard').classList.remove('hidden');
    } else {
      $('p1ValidationObject').textContent='No reference selected';
      $('p1ValidationTable').innerHTML='<tr><td colspan="5">Ground-truth error calculation is configured for the three preloaded Assignment 7 objects.</td></tr>';
    }
    $('p1Results').classList.remove('hidden');
  } catch(err) { alert(err.message); }
};

// ---------- Problem 3 ----------
const p3Left=pointCanvas('p3LeftCanvas','p3LeftStatus','#16a34a');
const p3Right=pointCanvas('p3RightCanvas','p3RightStatus','#dc2626');
p3Left.setConfig(2,['LEFT','RIGHT'],false); p3Right.setConfig(2,['LEFT','RIGHT'],false);
function setVideoSource(video, src) { video.src=src; video.load(); }
setVideoSource($('leftVideo'),'assets/room_left.mp4');
setVideoSource($('rightVideo'),'assets/room_right.mp4');
$('p3Source').onchange=()=>{
  const up=$('p3Source').value==='upload'; $('p3UploadControls').classList.toggle('hidden',!up);
  if(!up){ setVideoSource($('leftVideo'),'assets/room_left.mp4'); setVideoSource($('rightVideo'),'assets/room_right.mp4'); }
};
$('leftVideoUpload').onchange=e=>{if(e.target.files[0]) setVideoSource($('leftVideo'),URL.createObjectURL(e.target.files[0]));};
$('rightVideoUpload').onchange=e=>{if(e.target.files[0]) setVideoSource($('rightVideo'),URL.createObjectURL(e.target.files[0]));};
function bindVideoSlider(video, slider, label) {
  video.addEventListener('loadedmetadata',()=>{slider.max=Math.max(video.duration||0,0.01);slider.value=(video.duration||0)/2;video.currentTime=Number(slider.value);label.textContent=`${Number(slider.value).toFixed(2)} s`;});
  slider.oninput=()=>{video.currentTime=Number(slider.value);label.textContent=`${Number(slider.value).toFixed(2)} s`;};
  video.ontimeupdate=()=>{ if(!slider.matches(':active')) { slider.value=video.currentTime; label.textContent=`${video.currentTime.toFixed(2)} s`; } };
}
bindVideoSlider($('leftVideo'),$('leftTime'),$('leftTimeLabel'));
bindVideoSlider($('rightVideo'),$('rightTime'),$('rightTimeLabel'));
$('captureLeftFrame').onclick=()=>{if(!$('leftVideo').videoWidth)return alert('Load the LEFT video first.');p3Left.setVideoFrame($('leftVideo'));};
$('captureRightFrame').onclick=()=>{if(!$('rightVideo').videoWidth)return alert('Load the RIGHT video first.');p3Right.setVideoFrame($('rightVideo'));};
function updateRoomMeasurement(){
  const m=$('roomMeasurement').value;
  let labs,title,text;
  if(m==='width'){labs=['LEFT','RIGHT'];title='Width endpoints';text='Click the left endpoint, then the right endpoint, on both stereo frames.';}
  if(m==='height'){labs=['BOTTOM','TOP'];title='Height endpoints';text='Click the floor/bottom endpoint, then the ceiling/top endpoint, on both stereo frames.';}
  if(m==='length'){labs=['NEAR','FAR'];title='Length endpoints';text='Click the near endpoint, then the far endpoint along the same room edge, on both stereo frames.';}
  p3Left.setConfig(2,labs,false);p3Right.setConfig(2,labs,false);$('roomInstructionTitle').textContent=title;$('roomInstruction').textContent=text;
}
$('roomMeasurement').onchange=updateRoomMeasurement;
$('clearP3Points').onclick=()=>{p3Left.clear();p3Right.clear();$('p3Results').classList.add('hidden');};
$('calculateP3').onclick=()=>{
  try{
    const lp=p3Left.getPoints(),rp=p3Right.getPoints();
    if(!p3Left.hasImage()||!p3Right.hasImage()) throw new Error('Select one frame from each video first.');
    if(lp.length!==2||rp.length!==2) throw new Error('Select both corresponding endpoints in LEFT and RIGHT frames.');
    const s=p3Left.getSize(),sr=p3Right.getSize(); if(s.width!==sr.width||s.height!==sr.height) throw new Error('Selected LEFT and RIGHT frames must use the same resolution.');
    const c=getCalibration(s.width,s.height); const pts=[reconstruct(lp[0],rp[0],c),reconstruct(lp[1],rp[1],c)];
    const dim=dist3(pts[0],pts[1]); const m=$('roomMeasurement').value.toUpperCase();
    $('p3Z1').textContent=`${pts[0].Z.toFixed(2)} cm`; $('p3Z2').textContent=`${pts[1].Z.toFixed(2)} cm`;
    $('p3DimensionCm').textContent=`${dim.toFixed(2)} cm`; $('p3DimensionFt').textContent=`${(dim/30.48).toFixed(2)} ft`;
    const labels=$('roomMeasurement').value==='width'?['LEFT','RIGHT']:$('roomMeasurement').value==='height'?['BOTTOM','TOP']:['NEAR','FAR'];
    $('p3PointTable').innerHTML=pts.map((p,i)=>`<tr><td>${labels[i]}</td><td>${p.d.toFixed(2)}</td><td>${p.X.toFixed(2)}</td><td>${p.Y.toFixed(2)}</td><td>${p.Z.toFixed(2)}</td></tr>`).join('');
    $('p3Results').classList.remove('hidden');
  }catch(err){alert(err.message);}
};

// Initial state
updateShape();
loadPreloadedObject();
updateRoomMeasurement();
if(navigator.mediaDevices) refreshCameras();
