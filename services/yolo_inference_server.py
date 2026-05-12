"""
External YOLO inference service for robot controllers.
Robots send images to this service, which returns detected object positions.
"""

import base64
import glob
import logging
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, request

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
MODEL_PATH = os.path.join(YOLO_DIR, "inputs_cf/yolo_pruned_int_fixed.onnx")
ANCHORS_PATH = os.path.join(YOLO_DIR, "inputs_cf/anchors.npy")
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


def get_class_center(detections, target_class, original_shape, input_shape):
    if detections is None or len(detections) == 0:
        return None, None, None
    try:
        class_idx = CLASS_NAMES.index(target_class)
        best_box = None
        best_conf = -1
        for box in detections[0]:
            x1, y1, x2, y2, conf, pred_cls = box
            if int(pred_cls) == class_idx and conf > best_conf:
                best_conf = conf
                best_box = box
        if best_box is not None:
            x1, y1, x2, y2, conf, _ = best_box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Scale to original image size
            center_x_orig = center_x * original_shape[1] / input_shape[1]
            center_y_orig = center_y * original_shape[0] / input_shape[0]
            
            return float(center_x_orig), float(center_y_orig), float(conf)
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
        
        if not data or "image_b64" not in data:
            return jsonify({"success": False, "error": "Missing 'image_b64' in payload"}), 400
        
        # Decode image
        image_b64 = data["image_b64"]
        width = int(data['width'])
        height = int(data['height'])
        channels = int(data['channels'])
        image_data = base64.b64decode(image_b64)
        image_array = np.frombuffer(image_data, dtype=np.uint8)
        image_bgr = image_array.reshape((height, width, channels))
        
        if image_bgr is None:
            return jsonify({"success": False, "error": "Failed to decode image"}), 400
        
        target_class = data.get("target_class", TARGET_CLASS)
        conf_thres = float(data.get("conf_threshold", 0.25))
        iou_thres = float(data.get("iou_threshold", 0.45))
        
        # Run inference
        detections = predict(image_bgr, conf_thres, iou_thres)

        # Get target class center
        input_shape = session.get_inputs()[0].shape[2:]
        center_x, center_y, confidence = get_class_center(detections, target_class, image_bgr.shape, input_shape)
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
        
        ### Save annotated image for debugging
        try:
            os.makedirs(os.path.join(os.path.dirname(__file__), "imgs"), exist_ok=True)
            output_path = os.path.join(os.path.dirname(__file__), "imgs", "output.png")
            annotated = image_bgr.copy()

            if detections and len(detections) > 0:
                # Find the target box with highest confidence
                best_box = None
                best_conf = -1
                for box in detections[0]:
                    x1, y1, x2, y2, conf, pred_cls = box
                    if CLASS_NAMES[int(pred_cls)] == target_class and conf > best_conf:
                        best_conf = conf
                        best_box = box

                if best_box is not None:
                    x1, y1, x2, y2, conf, pred_cls = best_box
                    
                    input_shape = session.get_inputs()[0].shape[2:]
                    x1_orig = int(x1 * annotated.shape[1] / input_shape[1])
                    y1_orig = int(y1 * annotated.shape[0] / input_shape[0])
                    x2_orig = int(x2 * annotated.shape[1] / input_shape[1])
                    y2_orig = int(y2 * annotated.shape[0] / input_shape[0])

                    print(f"Detected {CLASS_NAMES[int(pred_cls)]} with confidence {conf:.2f}"
                          f" at ({x1_orig}, {y1_orig}), ({x2_orig}, {y2_orig})")

                    cv2.rectangle(annotated, (x1_orig, y1_orig), (x2_orig, y2_orig), (0, 255, 0), 2)

                    label = f"{target_class} {conf:.2f}"
                    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated, (x1_orig, y1_orig - th - baseline - 4),
                                  (x1_orig + tw, y1_orig), (0, 255, 0), -1)
                    cv2.putText(annotated, label, (x1_orig, y1_orig - baseline - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                if center_x is not None and center_y is not None:
                    cv2.drawMarker(annotated, (int(center_x), int(center_y)), (0, 255, 0),
                                   cv2.MARKER_CROSS, markerSize=16, thickness=2)

            cv2.imwrite(output_path, annotated)
            logger.info(f"Annotated image saved to {output_path}")

        except Exception as e:
            logger.warning(f"Failed to save annotated image: {e}")
            
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in /detect: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    init_inference()
    port = int(os.environ.get("YOLO_SERVER_PORT", 8765))
    logger.info(f"Starting YOLO inference service on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
