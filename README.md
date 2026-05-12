# Webots Leonard Dataset — Object Detection (YOLO)

Webots simulation for generating object detection datasets. A Franka Emika Panda robot uses an onboard camera and YOLO inference to detect objects (Cookie box, Kuka box, Wooden box) on a table, then moves its arm to the detected target.

Tested with **Python 3.10.20** and **Webots R2023b**.

## Prerequisites

- [Webots R2023b](https://github.com/cyberbotics/webots/releases/tag/R2023b)
- Python 3.10+
- Git

## Setup

### 1. Clone the repository

```bash
git clone git@github.com:eml-eda/webots-leonard-dataset-object-detection-yolo.git
cd webots-leonard-dataset-object-detection-yolo
```

### 2. Initialize the submodule

The project uses a git submodule for the TRISTAN YOLO ONNX inference library:

```bash
git submodule update --init --recursive
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

For development (linting, testing, type checking):

```bash
pip install -r requirements_dev.txt
```

### 4. Preparing the Neural Network Model

The Panda controller expects a fixed ONNX model at `controllers/panda/tristan-yolo-py-inference/inputs/yolo_pruned_int_fixed.onnx`. If this file is missing, generate it by running the repair script from the submodule directory:

```bash
cd controllers/panda/tristan-yolo-py-inference
python repair_onnx.py
```

This reads `inputs/yolo_pruned_int.onnx`, reconnects dangling outputs, strips invalid attributes, re-runs ONNX shape inference, and saves the result as `inputs/yolo_pruned_int_fixed.onnx`.


## Running

Open one of the world files in Webots:

```bash
webots worlds/leonardo-panda.wbt
```

Then start the simulation (play button). The Panda controller will:

1. Capture an image from the onboard camera
2. Run YOLO inference (ONNX runtime, CPU) to detect objects
3. Compute world coordinates of the target object
4. Move the Panda arm to the detected position

### World files

| File | Description |
|------|-------------|
| `worlds/leonardo-panda.wbt` | Panda robot with camera, objects on a table |
| `worlds/leonardo-dataset.wbt` | Dataset capture scene with camera and objects |

## Project structure

```
controllers/
  panda/                        # Main controller: camera capture + YOLO + arm movement
    movement.py                 # Panda arm IK and movement (roboticstoolbox-python)
    transformations.py          # Camera-to-world coordinate transforms
    tristan-yolo-py-inference/  # Submodule: TRISTAN YOLO ONNX inference + post-processing
  biscuit_box/                  # Random biscuit box placement
  kuka_box/                     # Random kuka box placement
  wooden_box/                   # Random wooden box placement
  camera_viewable_area.py       # Camera FOV ground projection utility
  check_overlap.py              # Polygon overlap detection
  compute_corners.py            # Bounding box corner computation
  generate_bbox.py              # Bounding box visualization
meshes/                         # 3D mesh assets (Panda visual links)
protos/                         # Custom Webots PROTO definitions
weights/                        # Model weights
worlds/                         # Webots simulation worlds
```

## Detection classes

| Class | Object |
|-------|--------|
| `Cookie_box` | Biscuit box |
| `Gray_block` | Gray block |
| `Wooden_box` | Wooden box |
