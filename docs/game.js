const $ = (selector) => document.querySelector(selector);

const video = $("#cameraVideo");
const cameraCanvas = $("#cameraOverlay");
const cameraCtx = cameraCanvas.getContext("2d");
const gameCanvas = $("#gameCanvas");
const ctx = gameCanvas.getContext("2d");
const cameraBtn = $("#cameraBtn");
const cameraStatus = $("#cameraStatus");
const cameraPlaceholder = $("#cameraPlaceholder");
const personLabel = $("#personLabel");
const personDot = $("#personDot");
const handCountEl = $("#handCount");
const scoreValue = $("#scoreValue");
const levelValue = $("#levelValue");
const livesValue = $("#livesValue");
const timeValue = $("#timeValue");
const pauseBtn = $("#pauseBtn");
const fpsLabel = $("#fpsLabel");
const toastEl = $("#toast");

const W = 960;
const H = 650;
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],[0,17]
];
const POSE_CONNECTIONS = [[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24]];
const HAND_MODEL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const POSE_MODEL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

let stream = null;
let handLandmarker = null;
let poseLandmarker = null;
let cameraOn = false;
let personDetected = false;
let hands = [null, null];
let lastDetectAt = 0;
let lastVideoTime = -1;
let aiReady = false;
let audioContext = null;
let toastTimer = null;

const game = {
  state: "menu",
  score: 0,
  lives: 3,
  level: 1,
  startedAt: 0,
  stateAt: performance.now(),
  nextSpawn: 0,
  fruits: [],
  particles: [],
  halves: [],
  trails: [[], []],
  previous: [null, null],
  cursors: [null, null],
  dwellAt: null,
  dwellDuration: 900,
  elapsed: 0,
  lastFrame: performance.now(),
  fps: 0,
};

const fruitStyles = [
  {emoji:"🍎", color:"#ef5264", glow:"#ff9c9f"},
  {emoji:"🍊", color:"#ff9d42", glow:"#ffd17b"},
  {emoji:"🍉", color:"#3cc17e", glow:"#ff7691"},
  {emoji:"🍇", color:"#9873e8", glow:"#ceb8ff"},
  {emoji:"🍋", color:"#d5dc57", glow:"#f8f19d"},
  {emoji:"🍓", color:"#ef5277", glow:"#ffb2c3"},
  {emoji:"🍍", color:"#e9b94e", glow:"#fff0a3"},
];

function showToast(message, error = false) {
  toastEl.textContent = message;
  toastEl.classList.toggle("error", error);
  toastEl.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.remove("show"), 3200);
}

function setCameraStatus(label, type = "neutral") {
  cameraStatus.className = "status-pill " + type;
  cameraStatus.innerHTML = "<i></i> " + label;
}

function playTone(frequency = 440, duration = 0.08, type = "sine", volume = 0.045) {
  try {
    audioContext ??= new (window.AudioContext || window.webkitAudioContext)();
    if (audioContext.state === "suspended") audioContext.resume();
    const oscillator = audioContext.createOscillator();
    const gain = audioContext.createGain();
    oscillator.type = type;
    oscillator.frequency.setValueAtTime(frequency, audioContext.currentTime);
    gain.gain.setValueAtTime(volume, audioContext.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioContext.currentTime + duration);
    oscillator.connect(gain).connect(audioContext.destination);
    oscillator.start();
    oscillator.stop(audioContext.currentTime + duration);
  } catch (_) {}
}

function resizeCameraOverlay() {
  if (!video.videoWidth || !video.videoHeight) return;
  cameraCanvas.width = video.videoWidth;
  cameraCanvas.height = video.videoHeight;
}

async function loadVisionModels() {
  setCameraStatus("Đang tải AI", "neutral");
  cameraBtn.textContent = "ĐANG TẢI AI...";
  const visionModule = await import("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs");
  const vision = await visionModule.FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  const handOptions = {
    baseOptions: { modelAssetPath: HAND_MODEL, delegate: "GPU" },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.52,
    minHandPresenceConfidence: 0.50,
    minTrackingConfidence: 0.50,
  };
  try {
    handLandmarker = await visionModule.HandLandmarker.createFromOptions(vision, handOptions);
  } catch (error) {
    console.warn("Hand model GPU not available; retry on CPU", error);
    delete handOptions.baseOptions.delegate;
    handLandmarker = await visionModule.HandLandmarker.createFromOptions(vision, handOptions);
  }

  try {
    poseLandmarker = await visionModule.PoseLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: POSE_MODEL, delegate: "GPU" },
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: 0.48,
      minPosePresenceConfidence: 0.48,
      minTrackingConfidence: 0.48,
    });
  } catch (error) {
    console.warn("Pose model GPU not available; retry on CPU", error);
    poseLandmarker = await visionModule.PoseLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: POSE_MODEL },
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: 0.48,
      minPosePresenceConfidence: 0.48,
      minTrackingConfidence: 0.48,
    });
  }

  aiReady = true;
  cameraBtn.innerHTML = "<span>◉</span> TẮT CAMERA";
  setCameraStatus("AI đã sẵn sàng", "good");
  showToast("Đã bật camera và tải xong AI. Hãy đưa tay vào khung hình.");
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    showToast("Trình duyệt không hỗ trợ camera. Hãy mở bằng HTTPS trên Chrome hoặc Edge.", true);
    return;
  }

  cameraBtn.disabled = true;
  cameraBtn.textContent = "ĐANG MỞ CAMERA...";
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: "user",
        width: { ideal: 960 },
        height: { ideal: 540 },
        frameRate: { ideal: 30, max: 30 },
      },
    });
    video.srcObject = stream;
    await video.play();
    cameraOn = true;
    cameraPlaceholder.classList.add("hidden");
    resizeCameraOverlay();
    await loadVisionModels();
    cameraBtn.disabled = false;
    requestAnimationFrame(detectLoop);
  } catch (error) {
    console.error(error);
    stopCamera();
    cameraBtn.disabled = false;
    cameraBtn.innerHTML = "<span>◎</span> THỬ BẬT LẠI CAMERA";
    setCameraStatus("Không thể bật", "bad");
    const message = error?.name === "NotAllowedError"
      ? "Bạn chưa cho phép truy cập camera. Hãy nhấn biểu tượng khóa cạnh địa chỉ web và cho phép Camera."
      : error?.name === "NotFoundError"
        ? "Không tìm thấy webcam trên thiết bị này."
        : "Không thể khởi tạo camera/AI. Hãy kiểm tra kết nối mạng và quyền camera.";
    showToast(message, true);
  }
}

function stopCamera() {
  cameraOn = false;
  aiReady = false;
  personDetected = false;
  hands = [null, null];
  if (stream) stream.getTracks().forEach(track => track.stop());
  stream = null;
  video.srcObject = null;
  handLandmarker?.close();
  poseLandmarker?.close();
  handLandmarker = null;
  poseLandmarker = null;
  cameraCtx.clearRect(0, 0, cameraCanvas.width, cameraCanvas.height);
  cameraPlaceholder.classList.remove("hidden");
  setCameraStatus("Đã tắt", "neutral");
  personDot.classList.remove("active");
  personLabel.textContent = "Chưa phát hiện người";
  handCountEl.textContent = "0";
  cameraBtn.innerHTML = "<span>◎</span> BẬT CAMERA";
  game.previous = [null, null];
  game.cursors = [null, null];
  game.trails = [[], []];
}

function drawCameraLandmarks(poseResult, handResult) {
  const width = cameraCanvas.width;
  const height = cameraCanvas.height;
  cameraCtx.clearRect(0, 0, width, height);
  if (!width || !height) return;

  if (poseResult?.landmarks?.length) {
    const landmarks = poseResult.landmarks[0];
    cameraCtx.lineWidth = Math.max(2, width / 360);
    cameraCtx.strokeStyle = "rgba(78, 218, 179, .72)";
    for (const [a, b] of POSE_CONNECTIONS) {
      const p1 = landmarks[a], p2 = landmarks[b];
      if (p1 && p2 && p1.visibility > 0.35 && p2.visibility > 0.35) {
        cameraCtx.beginPath();
        cameraCtx.moveTo(p1.x * width, p1.y * height);
        cameraCtx.lineTo(p2.x * width, p2.y * height);
        cameraCtx.stroke();
      }
    }
    for (const id of [11, 12, 13, 14, 15, 16, 23, 24]) {
      const p = landmarks[id];
      if (p && p.visibility > 0.35) {
        cameraCtx.beginPath();
        cameraCtx.fillStyle = "#8dffda";
        cameraCtx.arc(p.x * width, p.y * height, Math.max(3, width / 150), 0, Math.PI * 2);
        cameraCtx.fill();
      }
    }
  }

  for (const hand of handResult?.landmarks || []) {
    cameraCtx.lineWidth = Math.max(2, width / 420);
    cameraCtx.strokeStyle = "rgba(77, 208, 245, .92)";
    cameraCtx.fillStyle = "#d9fbff";
    for (const [a, b] of HAND_CONNECTIONS) {
      cameraCtx.beginPath();
      cameraCtx.moveTo(hand[a].x * width, hand[a].y * height);
      cameraCtx.lineTo(hand[b].x * width, hand[b].y * height);
      cameraCtx.stroke();
    }
    for (const point of hand) {
      cameraCtx.beginPath();
      cameraCtx.arc(point.x * width, point.y * height, Math.max(2, width / 220), 0, Math.PI * 2);
      cameraCtx.fill();
    }
  }
}

function classifyHands(handResult) {
  const result = [null, null];
  const all = handResult?.landmarks || [];
  all.forEach((landmarks, index) => {
    let slot = index;
    const handedness = (
      handResult?.handednesses?.[index]?.[0]?.categoryName ||
      handResult?.handedness?.[index]?.[0]?.categoryName ||
      ""
    ).toLowerCase();
    // Video preview is mirrored; keep stable left/right slots while converting to game coordinates.
    if (handedness === "left") slot = 0;
    if (handedness === "right") slot = 1;
    const tip = landmarks[8];
    result[slot] = { x: 1 - tip.x, y: tip.y };
  });
  return result;
}

function detectLoop(now) {
  if (!cameraOn || !aiReady || video.readyState < 2) return;
  if (video.currentTime !== lastVideoTime && now - lastDetectAt > 28) {
    lastVideoTime = video.currentTime;
    lastDetectAt = now;
    try {
      const timestamp = Math.round(now);
      const handResult = handLandmarker.detectForVideo(video, timestamp);
      const poseResult = poseLandmarker.detectForVideo(video, timestamp);
      personDetected = (poseResult?.landmarks?.length || 0) > 0;
      hands = classifyHands(handResult);
      drawCameraLandmarks(poseResult, handResult);

      personDot.classList.toggle("active", personDetected);
      personLabel.textContent = personDetected ? "Đã phát hiện người" : "Chưa thấy người trong khung hình";
      handCountEl.textContent = String(hands.filter(Boolean).length);
      if (personDetected) setCameraStatus("Đang nhận diện", "good");
      else setCameraStatus("Đang tìm người", "neutral");
    } catch (error) {
      console.error("Vision frame error:", error);
    }
  }
  requestAnimationFrame(detectLoop);
}

function rand(min, max) { return min + Math.random() * (max - min); }
function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }

function beginGame() {
  game.score = 0;
  game.lives = 3;
  game.level = 1;
  game.fruits = [];
  game.particles = [];
  game.halves = [];
  game.trails = [[], []];
  game.previous = [null, null];
  game.dwellAt = null;
  game.startedAt = performance.now();
  game.stateAt = game.startedAt;
  game.nextSpawn = game.startedAt + 500;
  game.state = "countdown";
  pauseBtn.disabled = false;
  pauseBtn.textContent = "Ⅱ TẠM DỪNG";
  playTone(520, .12, "sine", .045);
}

function endGame(reason) {
  game.state = "gameover";
  game.stateAt = performance.now();
  game.previous = [null, null];
  pauseBtn.disabled = true;
  playTone(125, .55, "sawtooth", .07);
  showToast(reason || "Trò chơi kết thúc.");
}

function pauseGame() {
  if (game.state === "playing") {
    game.state = "paused";
    game.stateAt = performance.now();
    game.previous = [null, null];
    pauseBtn.textContent = "▶ TIẾP TỤC";
  } else if (game.state === "paused") {
    const pauseDuration = performance.now() - game.stateAt;
    game.startedAt += pauseDuration;
    game.nextSpawn += pauseDuration;
    game.state = "playing";
    game.previous = [null, null];
    pauseBtn.textContent = "Ⅱ TẠM DỪNG";
  }
}

const startButtonRect = () => ({ x: W / 2 - 170, y: H / 2 + 28, w: 340, h: 76 });
function isInsideStart(point) {
  if (!point) return false;
  const r = startButtonRect();
  return point.x >= r.x && point.x <= r.x + r.w && point.y >= r.y && point.y <= r.y + r.h;
}

function spawnWave(now) {
  const elapsed = (now - game.startedAt) / 1000;
  game.level = 1 + Math.floor(elapsed / 18);
  const count = Math.min(1 + Math.floor(elapsed / 14), 6);
  const speed = 780 + Math.min(game.level - 1, 10) * 65;

  for (let i = 0; i < count; i++) {
    const style = fruitStyles[Math.floor(Math.random() * fruitStyles.length)];
    game.fruits.push({
      x: rand(70, W - 70) + (i - (count - 1) / 2) * 12,
      y: H + rand(25, 95),
      vx: rand(-135, 135),
      vy: -rand(speed * 1.22, speed * 1.48),
      r: rand(27, 39),
      style,
      bomb: false,
      angle: rand(-.5, .5),
      spin: rand(-2.5, 2.5),
    });
  }

  const bombChance = Math.min(.06 + Math.max(0, elapsed - 7) * .008, .32);
  if (elapsed > 7 && Math.random() < bombChance) {
    game.fruits.push({
      x: rand(80, W - 80), y: H + 45,
      vx: rand(-110, 110), vy: -rand(speed * 1.25, speed * 1.48),
      r: 34, style: null, bomb: true, angle: 0, spin: rand(-1.4, 1.4),
    });
  }

  game.nextSpawn = now + Math.max(470, 1450 - elapsed * 12);
}

function distanceToSegment(px, py, ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay;
  const len = dx * dx + dy * dy;
  if (len < .001) return Math.hypot(px - ax, py - ay);
  const t = clamp(((px - ax) * dx + (py - ay) * dy) / len, 0, 1);
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}

function burst(x, y, color, count = 22) {
  for (let i = 0; i < count; i++) {
    const a = rand(0, Math.PI * 2), speed = rand(75, 390);
    game.particles.push({
      x, y, vx: Math.cos(a) * speed, vy: Math.sin(a) * speed,
      life: rand(.35, .8), maxLife: .8, size: rand(2, 6),
      color: [color, "#fff5d7", "#70eafa", "#ffe18b"][Math.floor(Math.random() * 4)],
    });
  }
}

function sliceFruit(fruit, from, to) {
  const angle = Math.atan2(to.y - from.y, to.x - from.x);
  const side = angle + Math.PI / 2;
  for (const sign of [-1, 1]) {
    game.halves.push({
      x: fruit.x, y: fruit.y,
      vx: fruit.vx + Math.cos(side) * sign * 180,
      vy: fruit.vy - 80 + Math.sin(side) * sign * 180,
      r: fruit.r, color: fruit.style.color, emoji: fruit.style.emoji,
      rotation: fruit.angle + sign * .5, spin: sign * 3.8, life: 1.1,
    });
  }
  burst(fruit.x, fruit.y, fruit.style.glow, 20);
  game.score += 10;
  playTone(rand(620, 900), .07, "triangle", .035);
}

function checkSlash(from, to) {
  if (Math.hypot(to.x - from.x, to.y - from.y) < 8) return;
  const hits = game.fruits.filter(f => distanceToSegment(f.x, f.y, from.x, from.y, to.x, to.y) < f.r + 13);
  for (const fruit of hits) {
    const idx = game.fruits.indexOf(fruit);
    if (idx < 0) continue;
    game.fruits.splice(idx, 1);
    if (fruit.bomb) {
      burst(fruit.x, fruit.y, "#ff526d", 52);
      endGame("BOOM! Bạn đã chém trúng bom.");
      return;
    }
    sliceFruit(fruit, from, to);
  }
}

function updateCursors(now) {
  game.cursors = hands.map(point => point ? { x: point.x * W, y: point.y * H } : null);
  game.cursors.forEach((point, index) => {
    if (!point) {
      game.previous[index] = null;
      game.trails[index] = game.trails[index].filter(p => now - p.t < 180);
      return;
    }
    game.trails[index].push({ x: point.x, y: point.y, t: now });
    game.trails[index] = game.trails[index].filter(p => now - p.t < 230);
    if (game.state === "playing" && personDetected && game.previous[index]) {
      checkSlash(game.previous[index], point);
    }
    game.previous[index] = point;
  });
}

function updateGame(now, dt) {
  updateCursors(now);
  if (game.state === "menu" || game.state === "gameover") {
    const hovering = personDetected && game.cursors.some(isInsideStart);
    if (hovering) {
      game.dwellAt ??= now;
      if (now - game.dwellAt >= game.dwellDuration) beginGame();
    } else {
      game.dwellAt = null;
    }
  }

  if (game.state === "countdown") {
    if (now - game.stateAt >= 3000) {
      game.state = "playing";
      game.startedAt = now;
      game.nextSpawn = now + 300;
    }
    return;
  }
  if (game.state !== "playing") return;

  const elapsed = (now - game.startedAt) / 1000;
  game.level = 1 + Math.floor(elapsed / 18);
  const gravity = 1450 + Math.min(game.level - 1, 10) * 55;
  if (now >= game.nextSpawn) spawnWave(now);

  for (const fruit of [...game.fruits]) {
    fruit.x += fruit.vx * dt;
    fruit.y += fruit.vy * dt;
    fruit.vy += gravity * dt;
    fruit.angle += fruit.spin * dt;
    if (fruit.x < fruit.r || fruit.x > W - fruit.r) fruit.vx *= -1;
    if (fruit.y > H + fruit.r + 30) {
      game.fruits.splice(game.fruits.indexOf(fruit), 1);
      if (!fruit.bomb) {
        game.lives--;
        playTone(190, .12, "sine", .035);
        if (game.lives <= 0) {
          endGame("Bạn đã bỏ lỡ quá nhiều trái cây.");
          break;
        }
      }
    }
  }

  for (const p of [...game.particles]) {
    p.x += p.vx * dt; p.y += p.vy * dt;
    p.vx *= .985; p.vy += 480 * dt; p.life -= dt;
    if (p.life <= 0) game.particles.splice(game.particles.indexOf(p), 1);
  }
  for (const h of [...game.halves]) {
    h.x += h.vx * dt; h.y += h.vy * dt;
    h.vy += 850 * dt; h.rotation += h.spin * dt; h.life -= dt;
    if (h.life <= 0 || h.y > H + 100) game.halves.splice(game.halves.indexOf(h), 1);
  }
}

function drawArenaBackground() {
  const gradient = ctx.createLinearGradient(0, 0, W, H);
  gradient.addColorStop(0, "#0b1c30");
  gradient.addColorStop(.55, "#071626");
  gradient.addColorStop(1, "#0b1b2d");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, W, H);

  ctx.strokeStyle = "rgba(78,133,169,.075)";
  ctx.lineWidth = 1;
  for (let x = 0; x < W; x += 48) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
  }
  for (let y = 0; y < H; y += 48) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
  }
  for (let i = 0; i < 38; i++) {
    const x = (i * 173 + 41) % W;
    const y = (i * 97 + 23) % H;
    ctx.fillStyle = "rgba(137,204,230,.18)";
    ctx.fillRect(x, y, 2, 2);
  }
  const floor = ctx.createLinearGradient(0, H - 65, 0, H);
  floor.addColorStop(0, "rgba(36,111,133,0)");
  floor.addColorStop(1, "rgba(36,111,133,.13)");
  ctx.fillStyle = floor; ctx.fillRect(0, H - 65, W, 65);
}

function drawFruit(fruit) {
  ctx.save();
  ctx.translate(fruit.x, fruit.y);
  ctx.rotate(fruit.angle);
  if (fruit.bomb) {
    ctx.shadowColor = "#ff526d"; ctx.shadowBlur = 24;
    const grad = ctx.createRadialGradient(-10, -12, 4, 0, 0, fruit.r);
    grad.addColorStop(0, "#526276"); grad.addColorStop(.4, "#202c3e"); grad.addColorStop(1, "#080e19");
    ctx.fillStyle = grad;
    ctx.beginPath(); ctx.arc(0, 0, fruit.r, 0, Math.PI * 2); ctx.fill();
    ctx.shadowBlur = 0;
    ctx.strokeStyle = "#ff6c83"; ctx.lineWidth = 3; ctx.stroke();
    ctx.strokeStyle = "#c49a64"; ctx.lineWidth = 7; ctx.beginPath(); ctx.moveTo(5, -fruit.r + 3); ctx.quadraticCurveTo(12, -fruit.r - 13, 22, -fruit.r - 12); ctx.stroke();
    ctx.fillStyle = "#ffe18c"; ctx.shadowColor = "#ffc04c"; ctx.shadowBlur = 18;
    ctx.beginPath(); ctx.arc(23, -fruit.r - 13, 5, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#f6f9ff"; ctx.font = "bold 20px sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText("!", 0, 1);
  } else {
    ctx.shadowColor = fruit.style.glow; ctx.shadowBlur = 23;
    ctx.fillStyle = fruit.style.color;
    ctx.beginPath(); ctx.arc(0, 0, fruit.r, 0, Math.PI * 2); ctx.fill();
    ctx.shadowBlur = 0;
    const shine = ctx.createRadialGradient(-fruit.r * .33, -fruit.r * .37, 1, -fruit.r * .2, -fruit.r * .2, fruit.r);
    shine.addColorStop(0, "rgba(255,255,255,.6)"); shine.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = shine; ctx.beginPath(); ctx.arc(0, 0, fruit.r, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#765339"; ctx.fillRect(-2, -fruit.r - 9, 4, 13);
    ctx.fillStyle = "#77db98"; ctx.beginPath(); ctx.ellipse(8, -fruit.r - 6, 10, 4, -.4, 0, Math.PI * 2); ctx.fill();
    ctx.font = Math.max(26, fruit.r * 1.15) + "px sans-serif";
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(fruit.style.emoji, 0, 2);
  }
  ctx.restore();
}

function drawHalf(half) {
  ctx.save();
  ctx.globalAlpha = clamp(half.life / 1.1, 0, 1);
  ctx.translate(half.x, half.y);
  ctx.rotate(half.rotation);
  ctx.shadowColor = half.color; ctx.shadowBlur = 14;
  ctx.fillStyle = half.color;
  ctx.beginPath(); ctx.arc(0, 0, half.r, -Math.PI / 2, Math.PI / 2); ctx.lineTo(0, 0); ctx.closePath(); ctx.fill();
  ctx.shadowBlur = 0;
  ctx.strokeStyle = "#fff1d9"; ctx.lineWidth = 4;
  ctx.beginPath(); ctx.moveTo(0, -half.r); ctx.lineTo(0, half.r); ctx.stroke();
  ctx.font = Math.max(24, half.r) + "px sans-serif";
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillText(half.emoji, -7, 0);
  ctx.restore();
}

function drawParticles() {
  for (const p of game.particles) {
    ctx.globalAlpha = clamp(p.life / p.maxLife, 0, 1);
    ctx.fillStyle = p.color;
    ctx.shadowColor = p.color; ctx.shadowBlur = 10;
    ctx.beginPath(); ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2); ctx.fill();
  }
  ctx.globalAlpha = 1; ctx.shadowBlur = 0;
}

function drawTrails(now) {
  for (let i = 0; i < 2; i++) {
    const trail = game.trails[i].filter(p => now - p.t < 230);
    if (trail.length < 2) continue;
    for (let j = 1; j < trail.length; j++) {
      const alpha = j / trail.length;
      ctx.strokeStyle = i === 0 ? `rgba(87,225,242,${alpha * .78})` : `rgba(185,142,255,${alpha * .78})`;
      ctx.lineWidth = 2 + alpha * 7;
      ctx.lineCap = "round";
      ctx.beginPath(); ctx.moveTo(trail[j - 1].x, trail[j - 1].y); ctx.lineTo(trail[j].x, trail[j].y); ctx.stroke();
    }
    const p = trail[trail.length - 1];
    ctx.shadowColor = i === 0 ? "#6deafa" : "#c5a1ff"; ctx.shadowBlur = 20;
    ctx.fillStyle = "#f3fdff"; ctx.beginPath(); ctx.arc(p.x, p.y, 5, 0, Math.PI * 2); ctx.fill();
    ctx.shadowBlur = 0;
    ctx.strokeStyle = i === 0 ? "#6deafa" : "#c5a1ff"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(p.x, p.y, 15, 0, Math.PI * 2); ctx.stroke();
  }
}

function drawStartButton(now) {
  const r = startButtonRect();
  const hovered = personDetected && game.cursors.some(isInsideStart);
  const progress = hovered && game.dwellAt ? clamp((now - game.dwellAt) / game.dwellDuration, 0, 1) : 0;
  ctx.save();
  ctx.shadowColor = "#5bddeb"; ctx.shadowBlur = hovered ? 30 : 16;
  const grad = ctx.createLinearGradient(r.x, r.y, r.x + r.w, r.y + r.h);
  grad.addColorStop(0, hovered ? "#246e86" : "#184c68");
  grad.addColorStop(1, hovered ? "#255f83" : "#173b5b");
  roundedRect(r.x, r.y, r.w, r.h, 18, grad);
  ctx.shadowBlur = 0;
  ctx.strokeStyle = "#6bdce9"; ctx.lineWidth = 2; roundedRectStroke(r.x, r.y, r.w, r.h, 18);
  ctx.fillStyle = "#f4fbff"; ctx.font = "800 21px 'Be Vietnam Pro', sans-serif";
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillText(game.state === "gameover" ? "CHƠI LẠI" : "BẮT ĐẦU TRÒ CHƠI", W / 2, r.y + 33);
  ctx.fillStyle = "#a9d7e2"; ctx.font = "11px 'Be Vietnam Pro', sans-serif";
  ctx.fillText("GIỮ NGÓN TRỎ ĐỂ KÍCH HOẠT", W / 2, r.y + 56);
  if (progress > 0) {
    ctx.strokeStyle = "#ffda83"; ctx.lineWidth = 5; ctx.lineCap = "round";
    ctx.beginPath(); ctx.moveTo(r.x + 17, r.y + r.h - 7); ctx.lineTo(r.x + 17 + (r.w - 34) * progress, r.y + r.h - 7); ctx.stroke();
    ctx.strokeStyle = "#ffda83"; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.arc(W / 2, r.y - 17, 16, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * progress); ctx.stroke();
  }
  ctx.restore();
}
function roundedRect(x,y,w,h,r,fill){ctx.fillStyle=fill;ctx.beginPath();ctx.roundRect(x,y,w,h,r);ctx.fill();}
function roundedRectStroke(x,y,w,h,r){ctx.beginPath();ctx.roundRect(x,y,w,h,r);ctx.stroke();}

function drawMenu(now) {
  const r = startButtonRect();
  const titleY = H * .25;
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillStyle = "#ecf6ff"; ctx.font = "900 37px 'Be Vietnam Pro', sans-serif";
  ctx.shadowColor = "#5bddeb"; ctx.shadowBlur = 24;
  ctx.fillText("SẴN SÀNG CHƯA?", W / 2, titleY);
  ctx.shadowBlur = 0;
  ctx.fillStyle = "#9eb6ce"; ctx.font = "14px 'Be Vietnam Pro', sans-serif";
  ctx.fillText("Dùng chuyển động cơ thể để chém trái cây", W / 2, titleY + 43);
  drawStartButton(now);
  ctx.fillStyle = personDetected ? "#68dfea" : "#ffd17b";
  ctx.font = "12px 'Be Vietnam Pro', sans-serif";
  ctx.fillText(personDetected ? "Đưa ngón trỏ vào nút và giữ khoảng 1 giây" : "Bật camera và đứng vào trong khung hình để bắt đầu", W / 2, r.y + r.h + 28);

  const cards = [
    {title:"CHÉM BẰNG HAI TAY", desc:"Theo dõi cử chỉ theo thời gian thực", color:"#64dbe9"},
    {title:"TỐC ĐỘ TĂNG DẦN", desc:"Càng chơi lâu, thử thách càng lớn", color:"#76e4b5"},
    {title:"TRÁNH BOM", desc:"Chạm bom là kết thúc lượt chơi", color:"#ff8094"},
  ];
  const cardW = 238, gap = 14, startX = (W - (cardW * 3 + gap * 2)) / 2;
  cards.forEach((card, i) => {
    const x = startX + i * (cardW + gap), y = H - 105;
    ctx.fillStyle = "rgba(17,38,61,.92)";
    ctx.strokeStyle = "rgba(90,135,172,.35)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.roundRect(x, y, cardW, 66, 12); ctx.fill(); ctx.stroke();
    ctx.textAlign = "left"; ctx.fillStyle = card.color;
    ctx.font = "800 11px 'Be Vietnam Pro', sans-serif"; ctx.fillText(card.title, x + 14, y + 23);
    ctx.fillStyle = "#9cb2c8"; ctx.font = "10px 'Be Vietnam Pro', sans-serif"; ctx.fillText(card.desc, x + 14, y + 45);
  });
}

function drawHudInCanvas() {
  ctx.fillStyle = "rgba(13,29,48,.72)";
  ctx.fillRect(16, 15, W - 32, 56);
  ctx.strokeStyle = "rgba(79,119,155,.45)"; ctx.lineWidth = 1; ctx.strokeRect(16, 15, W - 32, 56);
  ctx.textAlign = "left";
  ctx.fillStyle = "#8da6bf"; ctx.font = "800 9px 'Be Vietnam Pro', sans-serif"; ctx.fillText("ĐIỂM", 31, 33);
  ctx.fillStyle = "#f1f8ff"; ctx.font = "900 22px 'Be Vietnam Pro', sans-serif"; ctx.fillText(String(game.score).padStart(4, "0"), 31, 57);
  ctx.fillStyle = "#8da6bf"; ctx.font = "800 9px 'Be Vietnam Pro', sans-serif"; ctx.fillText("CẤP ĐỘ", 194, 33);
  ctx.fillStyle = "#68e3ef"; ctx.font = "900 22px 'Be Vietnam Pro', sans-serif"; ctx.fillText(String(game.level).padStart(2, "0"), 194, 57);
  ctx.textAlign = "right";
  ctx.fillStyle = "#8da6bf"; ctx.font = "800 9px 'Be Vietnam Pro', sans-serif"; ctx.fillText("MẠNG", W - 32, 33);
  ctx.fillStyle = "#ff7188"; ctx.font = "900 20px 'Be Vietnam Pro', sans-serif"; ctx.fillText("♥ ".repeat(game.lives).trim() || "—", W - 32, 57);
}

function drawOverlay(now) {
  ctx.fillStyle = "rgba(3,8,16,.68)";
  ctx.fillRect(0, 0, W, H);
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  if (game.state === "countdown") {
    const remaining = Math.max(1, 3 - Math.floor((now - game.stateAt) / 1000));
    ctx.shadowColor = "#ffd47d"; ctx.shadowBlur = 35;
    ctx.fillStyle = "#ffe09a"; ctx.font = "900 112px 'Be Vietnam Pro', sans-serif";
    ctx.fillText(String(remaining), W / 2, H / 2 - 14);
    ctx.shadowBlur = 0;
    ctx.fillStyle = "#f2f8ff"; ctx.font = "800 17px 'Be Vietnam Pro', sans-serif";
    ctx.fillText("CHUẨN BỊ CHÉM!", W / 2, H / 2 + 80);
  } else if (game.state === "paused") {
    ctx.fillStyle = "#f1f7ff"; ctx.font = "900 34px 'Be Vietnam Pro', sans-serif"; ctx.fillText("ĐÃ TẠM DỪNG", W / 2, H / 2 - 15);
    ctx.fillStyle = "#9fb4c9"; ctx.font = "14px 'Be Vietnam Pro', sans-serif"; ctx.fillText("Nhấn P hoặc nút TIẾP TỤC để chơi tiếp", W / 2, H / 2 + 35);
  } else if (game.state === "gameover") {
    ctx.fillStyle = "#ff788d"; ctx.font = "900 35px 'Be Vietnam Pro', sans-serif"; ctx.fillText("TRÒ CHƠI KẾT THÚC", W / 2, H / 2 - 120);
    ctx.fillStyle = "#f2f7ff"; ctx.font = "800 24px 'Be Vietnam Pro', sans-serif"; ctx.fillText("ĐIỂM CỦA BẠN: " + game.score, W / 2, H / 2 - 70);
    drawStartButton(now);
    ctx.fillStyle = personDetected ? "#67deeb" : "#ffd17b"; ctx.font = "12px 'Be Vietnam Pro', sans-serif";
    ctx.fillText(personDetected ? "Đưa ngón trỏ vào nút để chơi lại" : "Đứng trước camera để chơi lại", W / 2, startButtonRect().y + 106);
  }
}

function drawGame(now) {
  drawArenaBackground();
  if (game.state === "menu") {
    drawMenu(now);
  } else {
    drawHudInCanvas();
    game.fruits.forEach(drawFruit);
    game.halves.forEach(drawHalf);
    drawParticles();
    if (game.state === "countdown" || game.state === "paused" || game.state === "gameover") {
      drawOverlay(now);
    }
  }
  drawTrails(now);
  if (game.state === "menu") drawTrails(now);
}

function updateHud(now) {
  scoreValue.textContent = String(game.score).padStart(4, "0");
  levelValue.textContent = String(game.level).padStart(2, "0");
  livesValue.textContent = game.lives > 0 ? ("♥ ".repeat(game.lives).trim()) : "—";
  const secs = game.state === "playing" ? Math.floor((now - game.startedAt) / 1000) : 0;
  timeValue.textContent = String(Math.floor(secs / 60)).padStart(2, "0") + ":" + String(secs % 60).padStart(2, "0");
  pauseBtn.disabled = !["playing", "paused"].includes(game.state);
  pauseBtn.textContent = game.state === "paused" ? "▶ TIẾP TỤC" : "Ⅱ TẠM DỪNG";
}

function gameLoop(now) {
  const dt = clamp((now - game.lastFrame) / 1000, 0, .04);
  game.lastFrame = now;
  updateGame(now, dt);
  drawGame(now);
  updateHud(now);
  const fps = dt > 0 ? Math.round(1 / dt) : 0;
  game.fps = game.fps ? game.fps * .9 + fps * .1 : fps;
  fpsLabel.textContent = cameraOn ? Math.round(game.fps) + " FPS · TRACKING" : "CAM READY";
  requestAnimationFrame(gameLoop);
}

function pointFromEvent(event) {
  const rect = gameCanvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) / rect.width * W,
    y: (event.clientY - rect.top) / rect.height * H,
  };
}

gameCanvas.addEventListener("pointerdown", event => {
  if (!["menu", "gameover"].includes(game.state)) return;
  const p = pointFromEvent(event);
  if (isInsideStart(p)) beginGame();
});
pauseBtn.addEventListener("click", pauseGame);
cameraBtn.addEventListener("click", () => cameraOn ? stopCamera() : startCamera());
$("#fullscreenBtn").addEventListener("click", async () => {
  try {
    if (!document.fullscreenElement) await document.documentElement.requestFullscreen();
    else await document.exitFullscreen();
  } catch (_) { showToast("Trình duyệt không cho phép toàn màn hình.", true); }
});
document.addEventListener("keydown", event => {
  if (event.key.toLowerCase() === "p") pauseGame();
  if (event.key === "Escape" && document.fullscreenElement) document.exitFullscreen();
});
window.addEventListener("beforeunload", stopCamera);
window.addEventListener("resize", () => {
  // Canvas uses a fixed logical coordinate system and scales responsively via CSS.
  drawGame(performance.now());
});

requestAnimationFrame(gameLoop);
