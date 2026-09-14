let stream=null;
let calibration=null;
let measurementFrame=null,measurementPoints=[];
let validationFrame=null,validationPoints=[];
let validationResults=[],validationIndex=0,validationStarted=false;

const OBJECTS=[
{name:"Car",width:5,height:3},{name:"AirPods",width:5.5,height:2},
{name:"Bigfoot",width:4.3,height:6.5},{name:"Ruby",width:3.5,height:6},
{name:"Pepper",width:3.7,height:8.7},{name:"Shoe polish",width:10,height:4.5},
{name:"Container",width:8,height:5.5},{name:"Sardines",width:7,height:2.4},
{name:"Key",width:2.5,height:1},{name:"Mouse",width:5,height:2.5},
{name:"Playdoh",width:4.7,height:3.8},{name:"Bose",width:8,height:9.5},
{name:"Oil",width:6.5,height:11.5},{name:"Juice",width:5.5,height:10.5},
{name:"Roku",width:4,height:1.6},{name:"Lotion",width:3.5,height:14},
{name:"Vitamin",width:9,height:10.5},{name:"Cup",width:7,height:13},
{name:"Brownie",width:9,height:1.5},{name:"Egg cup",width:5,height:7.5}
];

const $=id=>document.getElementById(id);

document.querySelectorAll(".step-tab").forEach(btn=>{
  btn.addEventListener("click",()=>showStep(btn.dataset.step));
});

function showStep(id){
  document.querySelectorAll(".step-tab").forEach(b=>b.classList.toggle("active",b.dataset.step===id));
  document.querySelectorAll(".step-panel").forEach(p=>p.classList.toggle("active-panel",p.id===id));
}

// ---------------- CAMERA ----------------
$("startCameraBtn").onclick=startCamera;
$("refreshCamerasBtn").onclick=loadCameras;
$("stopCameraBtn").onclick=stopCamera;
$("cameraSelect").onchange=()=>{if(stream)startCamera();};

async function loadCameras(){
  const devices=await navigator.mediaDevices.enumerateDevices();
  const cams=devices.filter(d=>d.kind==="videoinput");
  const select=$("cameraSelect");
  const old=select.value;
  select.innerHTML="";
  cams.forEach((c,i)=>{
    const o=document.createElement("option");
    o.value=c.deviceId;
    o.textContent=c.label||`Camera ${i+1}`;
    select.appendChild(o);
  });
  if(old&&[...select.options].some(o=>o.value===old))select.value=old;
}

async function startCamera(){
  try{
    stopCamera();
    const video={width:{ideal:1920},height:{ideal:1080}};
    if($("cameraSelect").value)video.deviceId={exact:$("cameraSelect").value};

    stream=await navigator.mediaDevices.getUserMedia({video,audio:false});
    $("video").srcObject=stream;
    await $("video").play();
    await loadCameras();

    $("cameraStatus").textContent=`Camera started: ${$("video").videoWidth} × ${$("video").videoHeight}`;
  }catch(e){
    $("cameraStatus").textContent=`Camera error: ${e.message}`;
  }
}

function stopCamera(){
  if(stream)stream.getTracks().forEach(t=>t.stop());
  stream=null;
  $("video").srcObject=null;
}

function capture(canvas){
  if(!stream||!$("video").videoWidth)throw new Error("Start the camera first.");

  canvas.width=$("video").videoWidth;
  canvas.height=$("video").videoHeight;

  canvas.getContext("2d").drawImage(
    $("video"),
    0,0,
    canvas.width,
    canvas.height
  );
}

// ---------------- STEP 1 ----------------
async function loadCalibrationAutomatically(){
  try{
    const response=await fetch("module2/calibration.json",{cache:"no-store"});
    if(!response.ok)throw new Error("Saved calibration file not found.");

    const data=await response.json();

    const fx=Number(data.fx);
    const fy=Number(data.fy);
    const cx=Number(data.cx);
    const cy=Number(data.cy);
    const width=Number(data.width ?? data.image_width);
    const height=Number(data.height ?? data.image_height);

    if(![fx,fy,cx,cy,width,height].every(Number.isFinite)){
      throw new Error("Saved calibration data is invalid.");
    }

    calibration={
      fx,fy,cx,cy,width,height,
      rms:data.rms ?? null,
      distortion:Array.isArray(data.distortion)?data.distortion:[]
    };

    localStorage.setItem("module2Calibration",JSON.stringify(calibration));
    showCalibration();

    $("calibrationStatus").textContent=
      "Saved OpenCV calibration loaded successfully. Step 1 is complete.";
  }catch(e){
    const cached=localStorage.getItem("module2Calibration");

    if(cached){
      calibration=JSON.parse(cached);
      showCalibration();
      $("calibrationStatus").textContent=
        "Previously saved calibration loaded. Step 1 is complete.";
    }else{
      $("calibrationStatus").textContent=
        "Calibration could not be loaded: "+e.message;
    }
  }
}

function showCalibration(){
  $("fxValue").textContent=calibration.fx.toFixed(4);
  $("fyValue").textContent=calibration.fy.toFixed(4);
  $("cxValue").textContent=calibration.cx.toFixed(4);
  $("cyValue").textContent=calibration.cy.toFixed(4);
  $("resolutionValue").textContent=`${calibration.width} × ${calibration.height}`;
  $("calibrationResults").classList.remove("hidden");
}

$("continueStep2Btn").onclick=()=>showStep("step2");

// ---------------- COMMON GEOMETRY ----------------
function getCanvasPoint(e,canvas){
  const rect=canvas.getBoundingClientRect();

  return{
    x:(e.clientX-rect.left)*canvas.width/rect.width,
    y:(e.clientY-rect.top)*canvas.height/rect.height
  };
}

function distance(a,b){
  return Math.hypot(b.x-a.x,b.y-a.y);
}

function drawPoints(canvas,points){
  const ctx=canvas.getContext("2d");

  ctx.strokeStyle="#00ff66";
  ctx.fillStyle="#00ff66";
  ctx.lineWidth=4;
  ctx.font="24px Arial";

  points.forEach((p,i)=>{
    ctx.beginPath();
    ctx.arc(p.x,p.y,8,0,Math.PI*2);
    ctx.fill();
    ctx.fillText(String(i+1),p.x+10,p.y-10);
  });

  if(points.length>1){
    ctx.beginPath();
    ctx.moveTo(points[0].x,points[0].y);

    for(let i=1;i<points.length;i++){
      ctx.lineTo(points[i].x,points[i].y);
    }

    if(points.length===4)ctx.closePath();
    ctx.stroke();
  }
}

function restoreFrame(canvas,url,points){
  if(!url)return;

  const img=new Image();

  img.onload=()=>{
    canvas.getContext("2d").drawImage(
      img,0,0,canvas.width,canvas.height
    );

    drawPoints(canvas,points);
  };

  img.src=url;
}

function calculateMeasurement(points,z,canvas){
  if(!calibration)throw new Error("Step 1 calibration is not loaded.");
  if(points.length!==4)throw new Error("Select exactly 4 corners.");

  const [tl,tr,br,bl]=points;

  const widthPixels=
    (distance(tl,tr)+distance(bl,br))/2;

  const heightPixels=
    (distance(tl,bl)+distance(tr,br))/2;

  // Scale intrinsic focal lengths if browser camera resolution differs
  // from calibration resolution.
  const fxScaled=
    calibration.fx*(canvas.width/calibration.width);

  const fyScaled=
    calibration.fy*(canvas.height/calibration.height);

  const widthCm=
    widthPixels*z/fxScaled*100;

  const heightCm=
    heightPixels*z/fyScaled*100;

  return{
    widthPixels,
    heightPixels,
    widthCm,
    heightCm
  };
}

// ---------------- STEP 2 ----------------
$("captureObjectBtn").onclick=()=>{
  try{
    const canvas=$("measurementCanvas");

    capture(canvas);

    measurementFrame=
      canvas.toDataURL("image/jpeg",0.9);

    measurementPoints=[];

    $("measurementResults").classList.add("hidden");
    $("measurementStatus").textContent=
      "Captured. Click Top-left, Top-right, Bottom-right, Bottom-left.";
  }catch(e){
    $("measurementStatus").textContent=e.message;
  }
};

$("measurementCanvas").onclick=e=>{
  if(!measurementFrame||measurementPoints.length>=4)return;

  measurementPoints.push(
    getCanvasPoint(e,$("measurementCanvas"))
  );

  restoreFrame(
    $("measurementCanvas"),
    measurementFrame,
    measurementPoints
  );

  $("measurementStatus").textContent=
    `${measurementPoints.length}/4 corners selected.`;
};

$("resetCornersBtn").onclick=()=>{
  measurementPoints=[];

  restoreFrame(
    $("measurementCanvas"),
    measurementFrame,
    measurementPoints
  );
};

$("calculateObjectBtn").onclick=()=>{
  try{
    const distanceM=Number($("step2Distance").value);

    if(!Number.isFinite(distanceM)||distanceM<=0)
      throw new Error("Enter a valid camera-to-object distance.");

    const r=calculateMeasurement(
      measurementPoints,
      distanceM,
      $("measurementCanvas")
    );

    $("pixelWidthResult").textContent=
      r.widthPixels.toFixed(2)+" px";

    $("pixelHeightResult").textContent=
      r.heightPixels.toFixed(2)+" px";

    $("widthResult").textContent=
      r.widthCm.toFixed(2)+" cm";

    $("heightResult").textContent=
      r.heightCm.toFixed(2)+" cm";

    $("measurementResults").classList.remove("hidden");

    $("measurementStatus").textContent=
      "Measurement calculated successfully.";
  }catch(e){
    $("measurementStatus").textContent=e.message;
  }
};

// ---------------- STEP 3 ----------------
$("validationMode").onchange=()=>{
  $("manualFields").classList.toggle(
    "hidden",
    $("validationMode").value!=="manual"
  );

  if(validationStarted)updateCurrentObject();
};

$("beginValidationBtn").onclick=()=>{
  if(!calibration){
    $("validationStatus").textContent=
      "Step 1 calibration is required.";
    return;
  }

  validationResults=[];
  validationIndex=0;
  validationStarted=true;
  validationFrame=null;
  validationPoints=[];

  $("validationTable").querySelector("tbody").innerHTML="";

  updateCurrentObject();

  $("validationStatus").textContent=
    "Validation started.";
};

function getCurrentTruth(){
  if($("validationMode").value==="predefined"){
    return OBJECTS[validationIndex]||null;
  }

  const name=$("manualName").value.trim();
  const width=Number($("manualWidth").value);
  const height=Number($("manualHeight").value);

  if(!name||width<=0||height<=0)return null;

  return{name,width,height};
}

function updateCurrentObject(){
  if($("validationMode").value==="predefined"){
    const o=OBJECTS[validationIndex];

    if(o){
      $("currentObject").innerHTML=
        `<b>${validationIndex+1}/20 - ${o.name}</b><br>
         Actual width: ${o.width.toFixed(2)} cm |
         Actual height: ${o.height.toFixed(2)} cm`;
    }else{
      $("currentObject").textContent=
        "All 20 predefined objects completed.";
    }
  }else{
    $("currentObject").textContent=
      "Enter the actual object information above.";
  }
}

$("captureValidationBtn").onclick=()=>{
  if(!validationStarted){
    $("validationStatus").textContent=
      "Press Begin Validation first.";
    return;
  }

  if(!getCurrentTruth()){
    $("validationStatus").textContent=
      "Enter valid object information.";
    return;
  }

  try{
    const canvas=$("validationCanvas");

    capture(canvas);

    validationFrame=
      canvas.toDataURL("image/jpeg",0.9);

    validationPoints=[];

    $("validationStatus").textContent=
      "Captured. Click the 4 corners.";
  }catch(e){
    $("validationStatus").textContent=e.message;
  }
};

$("validationCanvas").onclick=e=>{
  if(!validationFrame||validationPoints.length>=4)return;

  validationPoints.push(
    getCanvasPoint(e,$("validationCanvas"))
  );

  restoreFrame(
    $("validationCanvas"),
    validationFrame,
    validationPoints
  );

  $("validationStatus").textContent=
    `${validationPoints.length}/4 corners selected.`;
};

$("resetValidationCornersBtn").onclick=()=>{
  validationPoints=[];

  restoreFrame(
    $("validationCanvas"),
    validationFrame,
    validationPoints
  );
};

$("saveValidationBtn").onclick=()=>{
  const truth=getCurrentTruth();

  if(!truth){
    $("validationStatus").textContent=
      "Object information is missing.";
    return;
  }

  try{
    const distanceM=
      Number($("validationDistance").value);

    if(!Number.isFinite(distanceM)||distanceM<=0)
      throw new Error("Enter a valid validation distance.");

    const estimate=calculateMeasurement(
      validationPoints,
      distanceM,
      $("validationCanvas")
    );

    const widthError=
      estimate.widthCm-truth.width;

    const heightError=
      estimate.heightCm-truth.height;

    const widthAbs=
      Math.abs(widthError);

    const heightAbs=
      Math.abs(heightError);

    const widthPct=
      widthAbs/truth.width*100;

    const heightPct=
      heightAbs/truth.height*100;

    const result={
      measurement:validationResults.length+1,
      object:truth.name,
      distance:distanceM,

      actualWidth:truth.width,
      cameraWidth:estimate.widthCm,
      widthError,
      widthAbs,
      widthPct,

      actualHeight:truth.height,
      cameraHeight:estimate.heightCm,
      heightError,
      heightAbs,
      heightPct
    };

    validationResults.push(result);
    appendValidationRow(result);

    validationFrame=null;
    validationPoints=[];

    if($("validationMode").value==="predefined"){
      validationIndex++;
    }else{
      $("manualName").value="";
      $("manualWidth").value="";
      $("manualHeight").value="";
    }

    updateCurrentObject();

    $("validationStatus").textContent=
      "Measurement saved.";
  }catch(e){
    $("validationStatus").textContent=e.message;
  }
};

function appendValidationRow(r){
  const tr=document.createElement("tr");

  tr.innerHTML=`
    <td>${r.measurement}</td>
    <td>${r.object}</td>
    <td>${r.actualWidth.toFixed(2)}</td>
    <td>${r.cameraWidth.toFixed(2)}</td>
    <td>${r.actualHeight.toFixed(2)}</td>
    <td>${r.cameraHeight.toFixed(2)}</td>
    <td>${r.widthPct.toFixed(2)}%</td>
    <td>${r.heightPct.toFixed(2)}%</td>
  `;

  $("validationTable")
    .querySelector("tbody")
    .appendChild(tr);
}

$("clearValidationBtn").onclick=()=>{
  validationResults=[];
  validationIndex=0;
  validationStarted=false;
  validationFrame=null;
  validationPoints=[];

  $("validationTable")
    .querySelector("tbody")
    .innerHTML="";

  $("currentObject").textContent=
    "Press Begin Validation to start.";

  $("validationStatus").textContent=
    "Validation cleared.";
};

// ---------------- REPORT ----------------
function mean(a){
  return a.reduce((x,y)=>x+y,0)/a.length;
}

function rmse(a){
  return Math.sqrt(
    mean(a.map(v=>v*v))
  );
}

function std(a){
  const m=mean(a);

  return Math.sqrt(
    mean(a.map(v=>(v-m)**2))
  );
}

$("finishValidationBtn").onclick=()=>{
  if(!validationResults.length){
    $("validationStatus").textContent=
      "Complete validation measurements first.";
    return;
  }

  const widthErrors=
    validationResults.map(r=>r.widthError);

  const heightErrors=
    validationResults.map(r=>r.heightError);

  const widthAbs=
    validationResults.map(r=>r.widthAbs);

  const heightAbs=
    validationResults.map(r=>r.heightAbs);

  const widthPct=
    validationResults.map(r=>r.widthPct);

  const heightPct=
    validationResults.map(r=>r.heightPct);

  const wMean=mean(widthErrors);
  const hMean=mean(heightErrors);

  const wMAE=mean(widthAbs);
  const hMAE=mean(heightAbs);

  const wRMSE=rmse(widthErrors);
  const hRMSE=rmse(heightErrors);

  const wMAPE=mean(widthPct);
  const hMAPE=mean(heightPct);

  const wStd=std(widthErrors);
  const hStd=std(heightErrors);

  const overall=(wMAPE+hMAPE)/2;

  $("reportCount").textContent=
    validationResults.length;

  $("reportWMean").textContent=
    wMean.toFixed(3)+" cm";

  $("reportWMAE").textContent=
    wMAE.toFixed(3)+" cm";

  $("reportWRMSE").textContent=
    wRMSE.toFixed(3)+" cm";

  $("reportWMAPE").textContent=
    wMAPE.toFixed(2)+"%";

  $("reportWStd").textContent=
    wStd.toFixed(3)+" cm";

  $("reportHMean").textContent=
    hMean.toFixed(3)+" cm";

  $("reportHMAE").textContent=
    hMAE.toFixed(3)+" cm";

  $("reportHRMSE").textContent=
    hRMSE.toFixed(3)+" cm";

  $("reportHMAPE").textContent=
    hMAPE.toFixed(2)+"%";

  $("reportHStd").textContent=
    hStd.toFixed(3)+" cm";

  $("reportOverall").textContent=
    overall.toFixed(2)+"%";

  $("reportSummary").innerHTML=
    `Validated using <b>${validationResults.length}</b> measurement(s).
     Overall average percentage error:
     <b>${overall.toFixed(2)}%</b>.`;

  const tbody=
    $("reportTable").querySelector("tbody");

  tbody.innerHTML="";

  validationResults.forEach(r=>{
    const tr=document.createElement("tr");

    tr.innerHTML=`
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
};

$("downloadCsvBtn").onclick=()=>{
  if(!validationResults.length)return;

  const rows=[[
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
  ]];

  validationResults.forEach(r=>{
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

  const csv=
    rows.map(row=>
      row.map(v=>`"${String(v).replaceAll('"','""')}"`).join(",")
    ).join("\n");

  const blob=
    new Blob([csv],{type:"text/csv"});

  const url=
    URL.createObjectURL(blob);

  const a=
    document.createElement("a");

  a.href=url;
  a.download=
    "step3_validation_results_web.csv";

  a.click();

  URL.revokeObjectURL(url);
};

$("printReportBtn").onclick=()=>window.print();

window.addEventListener("load",async()=>{
  await loadCalibrationAutomatically();
  $("validationMode").dispatchEvent(new Event("change"));
});
