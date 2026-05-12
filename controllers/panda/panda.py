import os
import sys
import numpy as np
import cv2
from controller import Supervisor
from transformations import calculate_viewable_area, compute_points_world_from_relative
from movement import PandaMovement
import onnxruntime as ort

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INFERENCE_DIR = os.path.join(SCRIPT_DIR, "tristan-yolo-py-inference")
sys.path.insert(0, INFERENCE_DIR)

from post_proc import (
    fixed_point_nms,
    float_to_qx_y_tensor,
    merge_output_int,
    qx_y_to_float_tensor,
)

TIME_STEP = 32
MODEL_PATH = os.path.join(INFERENCE_DIR, "inputs", "yolo_pruned_int_fixed.onnx")
ANCHORS_PATH = os.path.join(INFERENCE_DIR, "inputs", "anchors.npy")
CLASS_NAMES = ["Cookie_box", "Gray_block", "Wooden_box"]


def capture_image(robot, save_path=None):
    print("Capturing image from camera...")
    camera = robot.getDevice("camera")
    if camera is None:
        print("ERROR: camera device not found")
        return None, 0, 0
    print(f"Camera found: {camera.getName()}, {camera.getWidth()}x{camera.getHeight()}")
    camera.enable(TIME_STEP)
    print("Camera enabled, warming up...")
    robot.step(TIME_STEP)
    print("Reading image...")
    image = camera.getImage()
    if image is None:
        print("Failed to capture image")
        return None, 0, 0

    width = camera.getWidth()
    height = camera.getHeight()
    image_array = np.frombuffer(image, dtype=np.uint8).reshape((height, width, 4))
    image_bgr = image_array[:, :, :3].copy()
    print("Image shape:", image_bgr.shape)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        camera.saveImage(save_path, 100)
        print(f"Image saved to {save_path}")

    return image_bgr, width, height


def preprocess_image(image_bgr, input_hw):
    """Preprocess image for YOLO model."""
    resized = cv2.resize(image_bgr, input_hw, interpolation=cv2.INTER_LINEAR)
    image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    image_chw = image_rgb.transpose(2, 0, 1)
    image_chw = np.expand_dims(image_chw, axis=0)
    image_input = np.ascontiguousarray(image_chw, dtype=np.float32)
    return image_input

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
        print(f"Error in get_class_center: {e}")
    
    return None, None, None

def predict(image_bgr, conf_thres=0.25, iou_thres=0.45):
    """Run YOLO inference on image."""

    session = session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    if session is None:
        raise RuntimeError("YOLO model not initialized")
    print("YOLO model loaded successfully")
    
    anchor_grid = np.load(ANCHORS_PATH)
    if anchor_grid is None:
        raise RuntimeError("Failed to load anchors")
    print("Anchors loaded successfully")
    
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

def main():
    robot = Supervisor()
    target_class = "Cookie_box"
    print("Starting Panda controller...")
    image_dir = os.path.join(os.path.dirname(__file__), "all_images")
    save_path = os.path.join(image_dir, "image_0.png")
    image_bgr, width, height = capture_image(robot, save_path=save_path)
    print("Image captured successfully.")
    detections = predict(image_bgr)
    print("YOLO inference completed.")
    center_x, center_y, confidence = get_class_center(detections, target_class)
    print(f"Detection result - Center: ({center_x}, {center_y}), Confidence: {confidence:.2f}")

    center_x = center_x * (640 / 128) / 640
    center_y = center_y * (360 / 128) / 360
    print(f"Relative coordinates: ({center_x:.4f}, {center_y:.4f})")

    viewable_area = calculate_viewable_area(1.3, 69, 42)
    print("Viewable area corners (relative to camera): ", viewable_area)
    points_world = compute_points_world_from_relative(
        viewable_area, [[center_x, center_y, 0.76]]
    )[0]
    print(
        f"World coordinates of detected object: ({points_world[0]:.4f}, {points_world[1]:.4f}, {points_world[2]:.4f})"
    )

    points_world[2] = 0.86
    # Initialize Panda movement controller
    panda_movement = PandaMovement(time_step=TIME_STEP, robot=robot)
    panda_movement.init_arm_components()
    panda_movement.move_arm(final_position=points_world, time_limit=0)

if __name__ == "__main__":
    main()