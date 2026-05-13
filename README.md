# Webots Leonard Dataset — Object Detection (YOLO)

Webots simulation for generating object detection datasets. A Franka Emika Panda robot uses a camera and YOLO inference to detect objects (Cookie box, Kuka box, Wooden box) on a table, then moves its arm to the detected target.

Tested with **Python 3.10.20** and **Webots R2023b**.

## Prerequisites

- [Webots R2023b](https://github.com/cyberbotics/webots/releases/tag/R2023b)
- Python 3.10+

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

### 3. Create a virtual environment and install Python dependencies

> [!WARNING]  
> Code has been tested with python 3.10.20. It should work with other 3.10 versions, but may not be compatible with Python 3.11+ due to some dependencies.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Preparing the Neural Network Model

The Panda controller expects a fixed ONNX model at `controllers/panda/tristan-yolo-py-inference/inputs/yolo_pruned_int_fixed.onnx`. If this file is missing, generate it by running the repair script from the submodule directory:

```bash
cd controllers/panda/tristan-yolo-py-inference
python repair_onnx.py
```

This reads `inputs/yolo_pruned_int.onnx`, reconnects dangling outputs, strips invalid attributes, re-runs ONNX shape inference, and saves the result as `inputs/yolo_pruned_int_fixed.onnx`.


## Running

> [!WARNING]  
> Code has been tested with Webots R2023b. Compatibility with newer versions is not guaranteed.

The next step is to launch the Webots simulation. Open the Webots application and go into settings and General. Here you should find a field called "Python command". Set this to the path of your virtual environment's python executable, for example:

```
/home/user/webots-leonard-dataset-object-detection-yolo/venv/bin/python
```

Now you can open the Webots world file `worlds/leonardo-panda.wbt`. This will load the simulation environment.

Then start the simulation (play button). The Panda controller will:

1. Capture an image from the onboard camera
2. Run YOLO inference (ONNX runtime, CPU) to detect objects
3. Compute world coordinates of the target object
4. Move the Panda arm to the detected position