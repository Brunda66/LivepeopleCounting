const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const annotatedImg = document.getElementById('annotated');
const countEl = document.getElementById('count');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const intervalInput = document.getElementById('interval');

let socket = null;
let stream = null;
let sending = false;
let sendInterval = 200;

function connectSocket() {
  socket = io(); // connects to same origin
  socket.on('connect', () => console.log('socket connected'));
  socket.on('server_message', d => console.log('srv:', d));
  socket.on('frame_result', d => {
    if (d.image) annotatedImg.src = d.image;
    if (typeof d.count !== 'undefined') countEl.innerText = 'People: ' + d.count;
  });
  socket.on('disconnect', () => console.log('socket disconnected'));
}

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
    video.srcObject = stream;
    await video.play();
    sendInterval = parseInt(intervalInput.value) || 200;
    sending = true;
    sendFramesLoop();
    startBtn.disabled = true;
    stopBtn.disabled = false;
  } catch (err) {
    alert('Could not start camera: ' + err);
  }
}

function stopCamera() {
  sending = false;
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  video.srcObject = null;
  startBtn.disabled = false;
  stopBtn.disabled = true;
}

async function sendFramesLoop() {
  if (!socket) connectSocket();
  while (sending) {
    if (video.readyState === HTMLMediaElement.HAVE_ENOUGH_DATA) {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL('image/jpeg', 0.6);
      try {
        socket.emit('frame', { image: dataUrl });
      } catch (e) {
        console.warn('Socket emit failed', e);
      }
    }
    await new Promise(r => setTimeout(r, parseInt(intervalInput.value) || sendInterval));
  }
}

startBtn.addEventListener('click', () => startCamera());
stopBtn.addEventListener('click', () => stopCamera());
intervalInput.addEventListener('change', () => { /* user changed interval */ });

// auto-connect socket so that it is ready when user starts
connectSocket();