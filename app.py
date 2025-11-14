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

# ============ OPTIMIZED CONFIGURATION ============
MODEL_NAME = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.5  # Increased from 0.3 for faster processing
MAX_FRAME_WIDTH = 640  # Resize incoming frames
MAX_FRAME_HEIGHT = 480
JPEG_QUALITY = 70  # Reduced from 80 for faster encoding
SKIP_FRAMES = 2  # Process every 3rd frame (0=process all, 1=every other, 2=every third)
MAX_DETECTIONS = 15  # Limit max detections per frame

# ============ FLASK & SOCKETIO SETUP ============
app = Flask(__name__)
app.config['SECRET_KEY'] = 'change-me-in-production'
socketio = SocketIO(
    app, 
    cors_allowed_origins="*", 
    async_mode='eventlet',
    ping_timeout=60,
    ping_interval=25,
    max_http_buffer_size=1e8  # 100MB for large frames
)

print("Loading YOLO model (this may take a few seconds)...")
model = YOLO(MODEL_NAME)

# Optimize model settings
model.overrides['conf'] = CONFIDENCE_THRESHOLD
model.overrides['iou'] = 0.45
model.overrides['max_det'] = MAX_DETECTIONS
model.overrides['classes'] = [0]  # Only detect person class

print("Model loaded and optimized. Ready!")

# ============ THREAD SAFETY ============
thread_lock = Lock()
frame_counter = 0
last_detection_result = None
last_frame_time = 0
processing_time_avg = []

# ============ HELPER FUNCTIONS ============
def resize_frame(img, max_width=MAX_FRAME_WIDTH, max_height=MAX_FRAME_HEIGHT):
    """Resize frame to reduce processing time"""
    h, w = img.shape[:2]
    
    # Calculate scaling factor
    scale = min(max_width / w, max_height / h, 1.0)  # Don't upscale
    
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    return img

def decode_base64_image(data_url: str):
    """Decode base64 image with optimization"""
    try:
        # Handle both with and without header
        if ',' in data_url:
            header, encoded = data_url.split(',', 1)
        else:
            encoded = data_url
        
        data = base64.b64decode(encoded)
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
        # Resize immediately after decoding
        img = resize_frame(img)
        
        return img
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def encode_image_to_base64_bgr(img_bgr):
    """Encode image to base64 with optimized JPEG quality"""
    _, buf = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    b64 = base64.b64encode(buf).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"

def draw_detections(img, boxes, draw_boxes=True):
    """Draw detection boxes and count people"""
    person_count = 0
    annotated = img.copy() if draw_boxes else img
    
    try:
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            if cls == 0 and conf >= CONFIDENCE_THRESHOLD:
                person_count += 1
                
                if draw_boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    
                    # Draw rectangle
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw confidence label
                    label = f"Person {conf:.2f}"
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated, (x1, y1 - label_h - 4), (x1 + label_w, y1), (0, 255, 0), -1)
                    cv2.putText(annotated, label, (x1, y1 - 4),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Add person count overlay
        if draw_boxes:
            count_text = f"Count: {person_count}"
            cv2.putText(annotated, count_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    except Exception as e:
        print(f"Warning parsing boxes: {e}")
    
    return annotated, person_count

# ============ ROUTES ============
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return {'status': 'healthy', 'model_loaded': True}

# ============ SOCKETIO EVENTS ============
@socketio.on('connect')
def handle_connect():
    print("Client connected")
    emit('server_message', {'msg': 'connected', 'config': {
        'max_width': MAX_FRAME_WIDTH,
        'max_height': MAX_FRAME_HEIGHT,
        'skip_frames': SKIP_FRAMES
    }})

@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")

@socketio.on('frame')
def handle_frame(data):
    """Handle incoming frame with optimizations"""
    global frame_counter, last_detection_result, last_frame_time, processing_time_avg
    
    start_time = time.time()
    frame_counter += 1
    
    try:
        # Skip frames for better performance
        if frame_counter % (SKIP_FRAMES + 1) != 0:
            # Return last detection result without processing
            if last_detection_result:
                emit('frame_result', last_detection_result)
            return
        
        # Decode image
        img = decode_base64_image(data['image'])
        if img is None:
            emit('error', {'msg': 'Failed to decode image'})
            return
        
        # Run detection (with thread lock for safety)
        with thread_lock:
            results = model.predict(
                source=img,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False,
                stream=False,
                half=False  # Use FP32 for CPU (more stable)
            )
        
        r = results[0]
        
        # Draw detections
        annotated, person_count = draw_detections(img, r.boxes, draw_boxes=True)
        
        # Encode result
        annotated_b64 = encode_image_to_base64_bgr(annotated)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # ms
        processing_time_avg.append(processing_time)
        if len(processing_time_avg) > 30:
            processing_time_avg.pop(0)
        avg_time = sum(processing_time_avg) / len(processing_time_avg)
        
        # Prepare response
        result = {
            'image': annotated_b64,
            'count': person_count,
            'processing_time': round(processing_time, 1),
            'avg_processing_time': round(avg_time, 1),
            'fps': round(1000 / avg_time, 1) if avg_time > 0 else 0
        }
        
        # Cache result
        last_detection_result = result
        last_frame_time = time.time()
        
        # Send result
        emit('frame_result', result)
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        import traceback
        traceback.print_exc()
        emit('error', {'msg': str(e)})

@socketio.on('update_config')
def handle_config_update(data):
    """Allow client to update configuration"""
    global CONFIDENCE_THRESHOLD, SKIP_FRAMES, JPEG_QUALITY
    
    if 'confidence' in data:
        CONFIDENCE_THRESHOLD = float(data['confidence'])
        model.overrides['conf'] = CONFIDENCE_THRESHOLD
    
    if 'skip_frames' in data:
        SKIP_FRAMES = int(data['skip_frames'])
    
    if 'jpeg_quality' in data:
        JPEG_QUALITY = int(data['jpeg_quality'])
    
    emit('config_updated', {
        'confidence': CONFIDENCE_THRESHOLD,
        'skip_frames': SKIP_FRAMES,
        'jpeg_quality': JPEG_QUALITY
    })

# ============ MAIN ============
if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    
    print(f"\n{'='*50}")
    print(f"🚀 AI Crowd Monitor Starting")
    print(f"{'='*50}")
    print(f"📊 Configuration:")
    print(f"   - Max Frame Size: {MAX_FRAME_WIDTH}x{MAX_FRAME_HEIGHT}")
    print(f"   - Skip Frames: {SKIP_FRAMES} (Process every {SKIP_FRAMES+1} frames)")
    print(f"   - Confidence: {CONFIDENCE_THRESHOLD}")
    print(f"   - JPEG Quality: {JPEG_QUALITY}")
    print(f"   - Max Detections: {MAX_DETECTIONS}")
    print(f"{'='*50}\n")
    
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
