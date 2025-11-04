// ==================== GLOBAL VARIABLES ====================
const canvas = document.getElementById("canvasId");
const ctx = canvas.getContext("2d");
const output = document.getElementById("output");

const MODEL_URL = "model.json";
const labels = ["0","1","2","3","4","5","6","7","8","9"];

let isDrawing = false;
let boundingBoxes = [];
let maxX = 0, maxY = 0, minX = canvas.width, minY = canvas.height;
const pad = 40;

let model = null;  // model will be loaded once

// ==================== INITIALIZE TFJS WITH WASM BACKEND ====================
async function init() {
  output.innerHTML = "Initializing TensorFlow.js...";
  await tf.setBackend("wasm");
  await tf.ready();
  console.log("Backend:", tf.getBackend()); // should log "wasm"

  output.innerHTML = "Loading model...";
  model = await tf.loadLayersModel(MODEL_URL);
  console.log("Model loaded!");

  // Warm up once for faster first inference
  tf.tidy(() => model.predict(tf.zeros([1, 28, 28, 1])));
  console.log("Model warmed up.");
  output.innerHTML = "Ready to classify!";
}

init(); // run at startup

// ==================== DRAWING LOGIC ====================
function updateBounds(x, y) {
  maxX = Math.max(maxX, x);
  maxY = Math.max(maxY, y);
  minX = Math.min(minX, x);
  minY = Math.min(minY, y);
}

canvas.addEventListener("mousedown", e => {
  isDrawing = true;
  ctx.beginPath();
  draw(e);
});

canvas.addEventListener("mousemove", draw);

canvas.addEventListener("mouseup", () => {
  isDrawing = false;
  recordBoundingBox();
});

function draw(e) {
  if (!isDrawing) return;
  ctx.lineTo(e.offsetX, e.offsetY);
  ctx.strokeStyle = "green";
  ctx.lineWidth = 5;
  ctx.stroke();
  updateBounds(e.offsetX, e.offsetY);
}

function recordBoundingBox() {
  const x0 = minX - pad / 2;
  const y0 = maxY + pad / 2;
  const w = maxX - minX + pad;
  const h = minY - maxY - pad;

  boundingBoxes.push([minY - pad / 2, minX - pad / 2, maxY + pad / 2, maxX + pad / 2]);

  ctx.beginPath();
  ctx.rect(x0, y0, w, h);
  ctx.strokeStyle = "red";
  ctx.lineWidth = 1;
  ctx.stroke();

  maxX = maxY = 0;
  minX = canvas.width;
  minY = canvas.height;
}

// ==================== CLASSIFICATION LOGIC ====================
async function classifyWriting() {
  if (!model) {
    output.innerHTML = "Model still loading...";
    return;
  }

  output.innerHTML = "Classifying...";
  const img = new Image();
  img.src = canvas.toDataURL();

  img.onload = async () => {
    const tensor = tf.browser.fromPixels(img).expandDims(0).div(255);
    const boxIndices = tf.tensor1d([0], "int32");
    const newSize = [28, 28];
    let resultString = "";

    for (const box of boundingBoxes) {
      const { index, prob } = tf.tidy(() => {
        const b = tf.tensor(box).div(canvas.width).reshape([1, 4]);
        const cropped = tf.image.cropAndResize(tensor, b, boxIndices, newSize);
        const gray = cropped.max(3).reshape([1, 28, 28, 1]);
        const prediction = model.predict(gray);
        const i = prediction.argMax(-1).dataSync()[0];
        const p = prediction.max().dataSync()[0];
        return { index: i, prob: p };
      });

      resultString += labels[index];
      output.innerHTML += `Detected: ${labels[index]} (confidence=${prob.toFixed(3)})<br>`;
    }

    output.innerHTML += `<br><b>Final Output:</b> ${resultString}<br>`;
  };
}

// ==================== BUTTON EVENT ====================
document.getElementById("classifyBtn").addEventListener("click", classifyWriting);
