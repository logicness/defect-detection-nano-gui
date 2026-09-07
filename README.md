# Defect Detection — Jetson Orin Nano Edge Device

Real-time surface defect detection on NVIDIA Jetson Orin Nano, powered by TensorRT inference with a local PyQt5 GUI and a TCP-based inference server for host-PC integration.

## Features

- **TensorRT FP16 inference** — YOLOv8/v11 models exported to TensorRT engine, GPU latency under 10 ms
- **Local GUI** — PyQt5 interface for camera preview, detection visualization, model management, and parameter tuning (runs standalone on the Nano via VNC)
- **TCP inference server** — asyncio-based server exposing detection over a custom frame protocol (4-byte big-endian length header + UTF-8 JSON), compatible with the companion Host PC application
- **Camera integration** — Hikvision USB3 industrial camera (MV-CS050-10UC) support via MVS SDK, with configurable exposure and gain
- **Automated deployment** — one-command deployment scripts for pushing code, models, and test images to the device over SSH

## Prerequisites

| Component | Requirement |
|-----------|-------------|
| Hardware | NVIDIA Jetson Orin Nano Super 8 GB |
| JetPack | 5.x+ (CUDA, TensorRT, cuDNN pre-installed) |
| Python | 3.8+ (system Python on JetPack) |
| Camera (optional) | Hikvision USB3 industrial camera with MVS SDK |

## Quick Start

```bash
# Clone the repository on your Jetson device
git clone https://github.com/logicness/defect-detection-nano-gui.git
cd defect-detection-nano-gui

# Install dependencies
pip install -r requirements.txt  # or: pip install pyserial PyQt5 opencv-python

# Start the inference server (listens on port 8888)
python nano_src/server/infer_server.py

# Launch the local GUI (requires VNC or display)
python nano_gui_app/app/main.py
```

## Project Structure

```
├── nano_gui_app/              # Standalone GUI application (PyQt5)
│   └── app/main.py            # Main window with detection, model management, settings
├── nano_src/
│   ├── inference/
│   │   ├── trt_infer.py       # TensorRT engine loader, letterbox, NMS, coordinate restore
│   │   └── camera_test.py     # Camera capture utility for testing
│   ├── server/
│   │   ├── infer_server.py    # TCP inference server (asyncio, port 8888)
│   │   ├── camera_source.py   # Camera abstraction (MVS SDK / simulation)
│   │   ├── model_manager.py   # Model loading, switching, and engine management
│   │   ├── image_store.py     # Image storage and retrieval
│   │   └── nano_local_detect.py  # Local detection pipeline
│   └── deploy/                # Deployment and provisioning scripts
│       ├── deploy_to_nano.py  # Push code/models to device via SSH/SCP
│       ├── deploy_camera.py   # Camera SDK installation and configuration
│       └── deploy_stream.py   # Stream testing utility
├── LICENSE                    # MIT License
└── README.md
```

## Configuration

The inference server and GUI read configuration at runtime. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `host` | `<NANO_LAN_IP>` | Device LAN IP address |
| `port` | `8888` | TCP inference server port |
| `model` | `models/active` | Path to TensorRT engine file |

Deployment scripts accept configuration via environment variables:

```bash
export NANO_HOST="192.168.1.101"   # Device IP
export NANO_USER="nvidia"          # SSH username (Jetson default)
export NANO_PASS="your-password"   # SSH password
python nano_src/deploy/deploy_to_nano.py
```

## Network Notes

Default IP addresses in configuration files are placeholders for a typical LAN setup. Replace them with your actual device addresses before deployment.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
