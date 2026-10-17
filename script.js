
"use strict";

const $ = id => document.getElementById(id);

const video = $("camera");
const overlay = $("overlay");
const ctx = overlay.getContext("2d");

const statusBox = $("status");
const placeholder = $("placeholder");
const connection = document.querySelector(".connection");

let stream = null;
let facingMode = "user";
let selectedFilter = "none";
let intensity = 100;
let mediaRecorder = null;
let recordedChunks = [];
let recording = false;
let faceDetector = null;
let handLandmarker = null;
let aiLoading = false;
let aiReady = false;
let detecting = false;
let lastVideoTime = -1;
let lastDetectTime = 0;
let faceCount = 0;
let handCount = 0;
let animationId = null;

const filterPresets = {
  none: "",
  vivid: "saturate(1.8) contrast(1.08)",
  mono: "grayscale(1) contrast(1.12)",
  vintage: "sepia(.65) contrast(1.05) saturate(.85)",
  dream: "saturate(1.35) brightness(1.08) hue-rotate(12deg)",
  cool: "saturate(1.15) hue-rotate(14deg) contrast(1.05)",
  warm: "sepia(.25) saturate(1.4) brightness(1.05)",
  cinema: "contrast(1.22) saturate(1.2) brightness(.94)"
};

const filterNames = {
  none: "ORIGINAL LOOK",
  vivid: "VIVID COLORS",
  mono: "MONOCHROME",
  vintage: "VINTAGE FILM",
  dream: "DREAM GLOW",
  cool: "ARCTIC BLUE",
  warm: "GOLDEN HOUR",
  cinema: "CINEMATIC"
};

function setStatus(message) {
  statusBox.textContent = message;
}

function setCameraState(active) {
  connection.classList.toggle("active", active);
  $("cameraState").textContent = active ? "LIVE" : "OFFLINE";
  $("cameraLabel").textContent = active ? "Your camera is running" : "Camera is ready";
  $("cameraDot").style.background = active ? "#54e6a6" : "";
  placeholder.classList.toggle("hidden", active);
}

function getFilterString() {
  if (!filterPresets[selectedFilter]) return "none";
  if (intensity === 0) return "none";
  if (intensity === 100) return filterPresets[selectedFilter];

  return `contrast(${1 + intensity / 500}) ${filterPresets[selectedFilter]}`;
}

function applyFilter() {
  video.style.filter = getFilterString();
  $("previewEffect").textContent = filterNames[selectedFilter] || "ORIGINAL LOOK";
  $("intensityValue").textContent = `${intensity}%`;

  document.querySelectorAll(".filter-card").forEach(button => {
    button.classList.toggle("active", button.dataset.filter === selectedFilter);
  });
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("Camera unavailable. Open this site through HTTPS in a supported browser.");
    return;
  }

  stopCamera(false);
  setStatus("Requesting camera permission...");

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: facingMode },
        width: { ideal: 1280 },
        height: { ideal: 960 }
      }
    });

    video.srcObject = stream;
    await video.play();

    setCameraState(true);
    resizeOverlay();
    applyFilter();
    setStatus("Camera ready! Choose a filter or load the AI models.");
    $("previewTime").textContent = "LIVE";
    startRenderLoop();
  } catch (error) {
    stream = null;
    setCameraState(false);
    let message = "Unable to open the camera.";

    if (error.name === "NotAllowedError") {
      message = "Camera permission denied. Allow camera access in your browser settings.";
    } else if (error.name === "NotFoundError") {
      message = "No compatible camera was found on this device.";
    } else if (error.name === "NotReadableError") {
      message = "Camera is busy. Close other camera apps and try again.";
    }

    setStatus(message);
  }
}

function stopCamera(showMessage = true) {
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }

  video.srcObject = null;
  setCameraState(false);

  faceCount = 0;
  handCount = 0;
  $("faceStatus").textContent = aiReady ? "Waiting for camera" : "Not started";
  $("handStatus").textContent = aiReady ? "Waiting for camera" : "Not started";

  $("faceLight").classList.remove("active");
  $("handLight").classList.remove("active");

  ctx.clearRect(0, 0, overlay.width, overlay.height);
  $("previewTime").textContent = "READY";

  if (showMessage) setStatus("Camera stopped.");

  if (animationId !== null) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }
}

function resizeOverlay() {
  if (!video.videoWidth || !video.videoHeight) return;

  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;
}

function startRenderLoop() {
  if (animationId !== null) cancelAnimationFrame(animationId);

  const render = async now => {
    animationId = requestAnimationFrame(render);

    if (!stream || video.readyState < 2) return;

    if (overlay.width !== video.videoWidth || overlay.height !== video.videoHeight) {
      resizeOverlay();
    }

    if (aiReady && !detecting && now - lastDetectTime > 100) {
      lastDetectTime = now;
      detecting = true;

      try {
        await detectAI(now);
      } catch (error) {
        console.warn("AI detection error:", error);
      } finally {
        detecting = false;
      }
    }

    drawDetections();
  };

  animationId = requestAnimationFrame(render);
}

async function loadAI() {
  if (aiLoading) return;

  if (aiReady) {
    setStatus("AI models are already loaded. Start the camera to detect faces and hands.");
    return;
  }

  aiLoading = true;
  $("aiBtn").disabled = true;
  $("aiBtn").textContent = "⏳ Loading AI Models...";
  $("modelBadge").textContent = "LOADING AI";
  setStatus("Loading AI libraries and models. Please keep your internet connected...");

  try {
    if (!window.vision || !window.vision.FilesetResolver ||
        !window.vision.FaceDetector || !window.vision.HandLandmarker) {
      throw new Error("AI library unavailable");
    }

    const visionFiles = await window.vision.FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm"
    );

    const face = await window.vision.FaceDetector.createFromOptions(visionFiles, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
        delegate: "CPU"
      },
      runningMode: "VIDEO",
      minDetectionConfidence: 0.5
    });

    let hands;

    try {
      hands = await window.vision.HandLandmarker.createFromOptions(visionFiles, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
          delegate: "CPU"
        },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5
      });
    } catch (handError) {
      face.close();
      throw new Error("Face model loaded, but hand model failed. Check your internet and retry.");
    }

    faceDetector = face;
    handLandmarker = hands;
    aiReady = true;

    $("modelBadge").textContent = "AI READY";
    $("aiBtn").textContent = "✓ AI Models Loaded";
    setStatus("AI models loaded successfully. Start the camera to begin detection.");

    if (stream) startRenderLoop();
  } catch (error) {
    console.error(error);
    $("modelBadge").textContent = "AI UNAVAILABLE";
    $("aiBtn").textContent = "↻ Retry AI Models";
    setStatus(error.message || "Unable to load AI models. Check your internet connection.");
  } finally {
    aiLoading = false;
    $("aiBtn").disabled = false;
  }
}

async function detectAI(now) {
  if (!stream || !aiReady || video.currentTime === lastVideoTime) return;

  lastVideoTime = video.currentTime;

  const timestamp = Math.round(now);
  const faceResult = faceDetector.detectForVideo(video, timestamp);
  const handResult = handLandmarker.detectForVideo(video, timestamp);

  faceCount = faceResult.detections?.length || 0;
  handCount = handResult.landmarks?.length || 0;

  $("faceStatus").textContent = faceCount
    ? `${faceCount} face${faceCount > 1 ? "s" : ""} detected`
    : "Looking for faces...";

  $("handStatus").textContent = handCount
    ? `${handCount} hand${handCount > 1 ? "s" : ""} detected`
    : "Looking for hands...";

  $("faceLight").classList.toggle("active", faceCount > 0);
  $("handLight").classList.toggle("active", handCount > 0);
}

function drawDetections() {
  if (!overlay.width || !overlay.height) return;

  ctx.clearRect(0, 0, overlay.width, overlay.height);

  if (!aiReady || !stream || !faceDetector || !handLandmarker) return;

  // Run detection results are obtained live; redraw using the latest results
  // on each frame when a result is available.
  // The detections themselves are refreshed by detectAI().
  if (window.lastFaceResult?.detections) {
    drawFaceBoxes(window.lastFaceResult.detections);
  }
  if (window.lastHandResult?.landmarks) {
    drawHandLandmarks(window.lastHandResult.landmarks);
  }
}

function drawFaceBoxes(detections) {
  ctx.save();
  ctx.strokeStyle = "#a78bfa";
  ctx.lineWidth = Math.max(2, overlay.width / 300);

  for (const item of detections) {
    const box = item.boundingBox;
    if (!box) continue;

    ctx.strokeRect(box.originX, box.originY, box.width, box.height);
  }

  ctx.restore();
}

function drawHandLandmarks(allHands) {
  const links = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [5,9],[9,10],[10,11],[11,12],
    [9,13],[13,14],[14,15],[15,16],
    [13,17],[17,18],[18,19],[19,20],[0,17]
  ];

  ctx.save();
  ctx.lineWidth = Math.max(2, overlay.width / 350);
  ctx.strokeStyle = "#55e8ff";
  ctx.fillStyle = "#b5f8ff";

  for (const points of allHands) {
    for (const [a, b] of links) {
      ctx.beginPath();
      ctx.moveTo(points[a].x * overlay.width, points[a].y * overlay.height);
      ctx.lineTo(points[b].x * overlay.width, points[b].y * overlay.height);
      ctx.stroke();
    }

    for (const point of points) {
      ctx.beginPath();
      ctx.arc(point.x * overlay.width, point.y * overlay.height, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  ctx.restore();
}

function capturePhoto() {
  if (!stream || video.readyState < 2) {
    setStatus("Start the camera before taking a photo.");
    return;
  }

  const photo = document.createElement("canvas");
  photo.width = video.videoWidth;
  photo.height = video.videoHeight;

  const pctx = photo.getContext("2d");
  pctx.filter = getFilterString();

  // Mirror the selfie image, matching the preview.
  pctx.translate(photo.width, 0);
  pctx.scale(-1, 1);
  pctx.drawImage(video, 0, 0, photo.width, photo.height);

  // Overlay uses the unmirrored source coordinate system.
  pctx.filter = "none";
  pctx.drawImage(overlay, 0, 0, photo.width, photo.height);

  const url = photo.toDataURL("image/png");
  const link = document.createElement("a");
  link.href = url;
  link.download = `ai-vision-${Date.now()}.png`;
  link.click();

  const item = document.createElement("div");
  item.className = "gallery-item";

  const image = document.createElement("img");
  image.src = url;
  image.alt = "Captured photo";

  const saveLink = document.createElement("a");
  saveLink.href = url;
  saveLink.download = link.download;
  saveLink.textContent = "⬇ Save photo";

  item.append(image, saveLink);
  $("gallery").prepend(item);
  setStatus("Photo captured! Your photo is ready to save.");
}

function toggleRecording() {
  if (recording) {
    mediaRecorder.stop();
    return;
  }

  if (!stream) {
    setStatus("Start the camera before recording a video.");
    return;
  }

  if (!window.MediaRecorder) {
    setStatus("Video recording is not supported in this browser.");
    return;
  }

  const mimeTypes = [
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm"
  ];

  const mimeType = mimeTypes.find(type => MediaRecorder.isTypeSupported(type));

  try {
    recordedChunks = [];
    mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);

    mediaRecorder.ondataavailable = event => {
      if (event.data.size) recordedChunks.push(event.data);
    };

    mediaRecorder.onstop = () => {
      const blob = new Blob(recordedChunks, {
        type: mediaRecorder.mimeType || "video/webm"
      });

      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `ai-vision-video-${Date.now()}.webm`;
      link.textContent = "⬇ Save recorded video";
      link.className = "video-download";
      link.style.display = "block";
      link.style.padding = "12px";
      link.style.marginTop = "12px";
      link.style.border = "1px solid #51467d";
      link.style.borderRadius = "10px";
      link.style.color = "#cbbaff";
      link.style.textAlign = "center";
      link.style.fontSize = "12px";
      link.style.textDecoration = "none";
      $("gallery").prepend(link);

      $("recordStatus").textContent = "Recording finished. Tap the link to save your video.";
      $("recordBtn").textContent = "🎥 Record Video";
      recording = false;
    };

    mediaRecorder.start();
    recording = true;
    $("recordBtn").textContent = "⏹ Stop Recording";
    $("recordStatus").textContent = "🔴 Recording... Tap Stop Recording to finish.";
    setStatus("Video recording started.");
  } catch (error) {
    setStatus("Unable to start recording on this browser.");
  }
}

$("startBtn").addEventListener("click", startCamera);
$("stopBtn").addEventListener("click", () => stopCamera());
$("switchBtn").addEventListener("click", async () => {
  facingMode = facingMode === "user" ? "environment" : "user";
  $("cameraLabel").textContent = facingMode === "user" ? "Front camera selected" : "Back camera selected";
  if (stream) await startCamera();
});
$("aiBtn").addEventListener("click", loadAI);
$("photoBtn").addEventListener("click", capturePhoto);
$("recordBtn").addEventListener("click", toggleRecording);

document.querySelectorAll(".filter-card").forEach(button => {
  button.addEventListener("click", () => {
    selectedFilter = button.dataset.filter;
    applyFilter();
    setStatus(`${filterNames[selectedFilter]} selected.`);
  });
});

$("intensity").addEventListener("input", event => {
  intensity = Number(event.target.value);
  applyFilter();
});

video.addEventListener("loadedmetadata", resizeOverlay);

window.addEventListener("pagehide", () => {
  if (stream) stream.getTracks().forEach(track => track.stop());
  if (faceDetector) faceDetector.close();
  if (handLandmarker) handLandmarker.close();
});

applyFilter();
setCameraState(false);
