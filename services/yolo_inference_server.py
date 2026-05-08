#!/usr/bin/env python3
"""HTTP inference server used to simulate a YOLO endpoint."""

import base64
import binascii
import json
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from ultralytics import YOLO

MODEL_WEIGHTS_PATH = os.getenv(
    'YOLO_MODEL_WEIGHTS',
    '/workspace/weights/pedestrian_best.pt',
)
SERVER_HOST = os.getenv('YOLO_SERVER_HOST', '0.0.0.0')
SERVER_PORT = int(os.getenv('YOLO_SERVER_PORT', '8765'))
WEBOTS_HEALTHCHECK_URL = os.getenv('YOLO_WEBOTS_HEALTHCHECK_URL', '').strip()
WEBOTS_HEALTHCHECK_PERIOD_S = max(0.5, float(os.getenv('YOLO_WEBOTS_HEALTHCHECK_PERIOD_S', '2.0')))

if not os.path.exists(MODEL_WEIGHTS_PATH):
    raise FileNotFoundError(f"YOLO weights not found: {MODEL_WEIGHTS_PATH}")

MODEL = YOLO(MODEL_WEIGHTS_PATH)


def _json_response(handler, status_code, payload):
    body = json.dumps(payload).encode('utf-8')
    try:
        handler.send_response(status_code)
        handler.send_header('Content-Type', 'application/json')
        handler.send_header('Content-Length', str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)
        return True
    except (BrokenPipeError, ConnectionResetError):
        # Client disconnected before response was written.
        return False


def _decode_image(payload):
    expected_fields = ['image_b64', 'width', 'height', 'channels']
    for field in expected_fields:
        if field not in payload:
            raise ValueError(f"Missing required field: {field}")

    width = int(payload['width'])
    height = int(payload['height'])
    channels = int(payload['channels'])
    if width <= 0 or height <= 0 or channels != 3:
        raise ValueError('Invalid image dimensions or channels')

    image_bytes = base64.b64decode(payload['image_b64'], validate=True)
    expected_len = width * height * channels
    if len(image_bytes) != expected_len:
        raise ValueError(
            f"Invalid image size: expected {expected_len} bytes, got {len(image_bytes)}"
        )

    image = np.frombuffer(image_bytes, dtype=np.uint8).reshape((height, width, channels))
    return image, width, height


def _run_inference(image):
    results = MODEL.predict(source=image, verbose=False)
    detections = []
    for detection in results[0].boxes:
        x1, y1, x2, y2 = map(int, detection.xyxy[0])
        conf = float(detection.conf[0])
        cls = int(detection.cls[0])
        if isinstance(MODEL.names, dict):
            label = str(MODEL.names.get(cls, f'class_{cls}'))
        elif isinstance(MODEL.names, list) and cls < len(MODEL.names):
            label = str(MODEL.names[cls])
        else:
            label = f'class_{cls}'

        detections.append(
            {
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'conf': conf,
                'cls': cls,
                'label': label,
            }
        )
    return detections


def _monitor_webots_presence():
    if not WEBOTS_HEALTHCHECK_URL:
        return

    previous_state = None
    while True:
        connected = False
        try:
            with urllib.request.urlopen(WEBOTS_HEALTHCHECK_URL, timeout=1.0) as response:
                connected = response.status < 500
        except (urllib.error.URLError, TimeoutError):
            connected = False

        if connected != previous_state:
            state = 'online' if connected else 'offline'
            print(f"[yolo-server] webots health endpoint is {state}: {WEBOTS_HEALTHCHECK_URL}")
            previous_state = connected

        time.sleep(WEBOTS_HEALTHCHECK_PERIOD_S)


class InferenceHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            _json_response(self, 200, {'status': 'ok'})
            return
        _json_response(self, 404, {'error': 'Not found'})

    def do_POST(self):
        if self.path != '/detect':
            _json_response(self, 404, {'error': 'Not found'})
            return

        try:
            content_length = int(self.headers.get('Content-Length', '0'))
            if content_length <= 0:
                raise ValueError('Missing body')

            raw_payload = self.rfile.read(content_length)
            payload = json.loads(raw_payload.decode('utf-8'))
            image, width, height = _decode_image(payload)

            started_at = time.perf_counter()
            detections = _run_inference(image)
            inference_ms = (time.perf_counter() - started_at) * 1000.0

            response = {
                'frame_id': payload.get('frame_id'),
                'image_width': width,
                'image_height': height,
                'detections': detections,
                'count': len(detections),
                'inference_ms': round(inference_ms, 3),
            }
            _json_response(self, 200, response)
        except (ValueError, json.JSONDecodeError, binascii.Error) as exc:
            _json_response(self, 400, {'error': str(exc)})
        except (BrokenPipeError, ConnectionResetError):
            # Client disconnected while we were preparing/sending a response.
            return
        except Exception as exc:  # noqa: BLE001
            if not _json_response(self, 500, {'error': f'Internal error: {exc}'}):
                return

    def log_message(self, fmt, *args):
        print(f"[yolo-server] {self.address_string()} - {fmt % args}")


if __name__ == '__main__':
    print(f"[yolo-server] Loading weights: {MODEL_WEIGHTS_PATH}")
    print(f"[yolo-server] Listening on {SERVER_HOST}:{SERVER_PORT}")
    if WEBOTS_HEALTHCHECK_URL:
        print(f"[yolo-server] Monitoring webots endpoint: {WEBOTS_HEALTHCHECK_URL}")
        threading.Thread(target=_monitor_webots_presence, daemon=True).start()

    server = ThreadingHTTPServer((SERVER_HOST, SERVER_PORT), InferenceHandler)
    server.serve_forever()
