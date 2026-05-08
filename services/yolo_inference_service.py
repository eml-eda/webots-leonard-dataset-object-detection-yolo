"""
External YOLO inference service for robot controllers.
Robots send images to this service, which returns detected object positions.
"""

import os
import sys
import glob
import base64
import logging
from flask import Flask, request, jsonify
import cv2
import numpy as np
import onnxruntime as ort

# Add tristan-yolo-py-inference to path
YOLO_DIR = os.path.join(os.path.dirname(__file__), "tristan-yolo-py-inference")
sys.path.insert(0, YOLO_DIR)

from post_proc import (
    fixed_point_nms,
    float_to_qx_y_tensor,
    merge_output_int,
    qx_y_to_float_tensor,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)

# YOLO Configuration
MODEL_PATH = os.path.join(YOLO_DIR, "inputs/yolo_pruned_int_fixed.onnx")
ANCHORS_PATH = os.path.join(YOLO_DIR, "inputs/anchors.npy")
CLASS_NAMES = ["Cookie_box", "Gray_block", "Wooden_box"]
TARGET_CLASS = "Cookie_box"

# Global session for inference
session = None
anchor_grid = None


def init_inference():
    """Initialize ONNX session and load anchors."""
    global session, anchor_grid
    try:
        session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        anchor_grid = np.load(ANCHORS_PATH)
        logger.info("YOLO model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        raise


def preprocess_image(image_bgr, input_hw):
    """Preprocess image for YOLO model."""
    resized = cv2.resize(image_bgr, input_hw, interpolation=cv2.INTER_LINEAR)
    image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    image_chw = image_rgb.transpose(2, 0, 1)
    image_chw = np.expand_dims(image_chw, axis=0)
    image_input = np.ascontiguousarray(image_chw, dtype=np.float32)
    return image_input


def predict(image_bgr, conf_thres=0.25, iou_thres=0.45):
    """Run YOLO inference on image."""
    if session is None:
        raise RuntimeError("YOLO model not initialized")
    
    input_name = session.get_inputs()[0].name
    output_names = [_.name for _ in session.get_outputs()]
    input_shape = session.get_inputs()[0].shape
    
    image = preprocess_image(image_bgr, input_shape[2:])
    raw_output = session.run(output_names, {input_name: image})
    
    x_fixed = [float_to_qx_y_tensor(t, 4, 12) for t in raw_output]
    out = merge_output_int(x_fixed, anchor_grid)
    out = fixed_point_nms(out, conf_thres, iou_thres)
    out = [qx_y_to_float_tensor(t.astype(np.int32), 14, 15) for t in out]
    return out


def get_class_center(detections, target_class):
    """Get center coordinates of target class detection."""
    if detections is None or len(detections) == 0:
        return None, None, None
    
    try:
        class_idx = CLASS_NAMES.index(target_class)
        for box in detections[0]:
            x1, y1, x2, y2, conf, pred_cls = box
            if int(pred_cls) == class_idx:
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                return float(center_x), float(center_y), float(conf)
    except Exception as e:
        logger.error(f"Error getting class center: {e}")
    
    return None, None, None


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "model": "yolo"}), 200


@app.route("/detect", methods=["POST"])
def detect():
    """
    Detect objects in an image.
    
    Expected JSON payload:
    {
        "image": "<base64-encoded image>",
        "target_class": "Cookie_box",
        "conf_threshold": 0.25,
        "iou_threshold": 0.45
    }
    
    Returns:
    {
        "success": true/false,
        "target_class": "Cookie_box",
        "x": <center_x or null>,
        "y": <center_y or null>,
        "confidence": <confidence or null>,
        "image_size": [width, height],
        "detections": [...all detections...]
    }
    """
    try:
        data = request.get_json()
        logger.info("/detect: image received")
        
        if not data or "image" not in data:
            return jsonify({"success": False, "error": "Missing 'image' in payload"}), 400
        
        # Decode image
        image_b64 = data["image"]
        image_data = base64.b64decode(image_b64)
        image_array = np.frombuffer(image_data, dtype=np.uint8)
        image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image_bgr is None:
            return jsonify({"success": False, "error": "Failed to decode image"}), 400
        
        target_class = data.get("target_class", TARGET_CLASS)
        conf_thres = float(data.get("conf_threshold", 0.25))
        iou_thres = float(data.get("iou_threshold", 0.45))
        
        # Run inference
        detections = predict(image_bgr, conf_thres, iou_thres)

        # Get target class center
        center_x, center_y, confidence = get_class_center(detections, target_class)
        logger.info(f"/detect: detection result for '{target_class}': x={center_x}, y={center_y}, conf={confidence}")
        
        # Format all detections for response
        all_detections = []
        if detections and len(detections) > 0:
            for box in detections[0]:
                x1, y1, x2, y2, conf, pred_cls = box
                all_detections.append({
                    "class": CLASS_NAMES[int(pred_cls)],
                    "confidence": float(conf),
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "center_x": float((x1 + x2) / 2),
                    "center_y": float((y1 + y2) / 2),
                })
        
        response = {
            "success": True,
            "target_class": target_class,
            "x": center_x,
            "y": center_y,
            "confidence": confidence,
            "image_size": [image_bgr.shape[1], image_bgr.shape[0]],
            "detections": all_detections,
        }
        
        logger.info("/detect: sending response to client")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in /detect: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    init_inference()
    logger.info("Starting YOLO inference service on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
