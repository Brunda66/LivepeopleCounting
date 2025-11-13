import base64
import time
from threading import Lock

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import numpy as np
import cv2
import torch

# Ensure ultralytics classes are allowlisted for torch.load (PyTorch 2.6+ fix)
try:
    from ultralytics.nn.tasks import DetectionModel
    torch.serialization.add_safe_globals([DetectionModel])
except Exception as e:
    print("Warning - safe globals setup:", e)

from ultralytics import YOLO

# Configuration
MODEL_NAME = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.3

app = Flask(__name__)
app.config['SECRET_KEY'] = 'change-me-in-production'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

print("Loading YOLO model (this may take a few seconds to download weights on first run)...")
model = YOLO(MODEL_NAME)
print("Model loaded. Ready.")

thread_lock = Lock()

def decode_base64_image(data_url: str):
    header, encoded = data_url.split(',', 1)
    data = base64.b64decode(encoded)
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def encode_image_to_base64_bgr(img_bgr):
    _, buf = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    b64 = base64.b64encode(buf).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print("Client connected")
    emit('server_message', {'msg': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")

@socketio.on('frame')
def handle_frame(data):
    try:
        img = decode_base64_image(data['image'])
        results = model.predict(source=img, conf=CONFIDENCE_THRESHOLD, verbose=False)
        r = results[0]
        annotated = img.copy()
        person_count = 0

        try:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls == 0 and conf >= CONFIDENCE_THRESHOLD:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, f"P:{conf:.2f}", (x1, max(10, y1-6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                    person_count += 1
        except Exception as e:
            print("Warning parsing boxes:", e)

        annotated_b64 = encode_image_to_base64_bgr(annotated)
        emit('frame_result', {'image': annotated_b64, 'count': person_count})
    except Exception as e:
        print("Error processing frame:", e)

if __name__ == '__main__':
    print("Starting server on http://0.0.0.0:5000")
    socketio.run(app, host='0.0.0.0', port=5000)
