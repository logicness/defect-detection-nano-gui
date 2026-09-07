# 缺陷检测边缘设备（Nano 下位机）

工业表面缺陷检测项目，基于 **NVIDIA Jetson Orin Nano Super 8G** 部署 YOLOv8 TensorRT 推理。

## 目录结构

```
defect_detection/
├── README.md                 本文档
├── inference/
│   └── trt_infer.py          TensorRT 推理脚本（核心）
├── server/
│   └── infer_server.py       TCP 推理服务（asyncio）
├── tests/test_images/        测试图片
│   ├── bus.jpg               COCO 示例图
│   └── random.jpg            随机图（测速用）
├── outputs/                  推理结果输出
└── models/
    ├── baseline/             YOLOv8n/s 模型产物（pt / onnx / engine）
    └── ultralytics/          Ultralytics 源码备份（参考用）
```

## 各文件/文件夹说明

| 路径 | 作用 |
|------|------|
| `inference/trt_infer.py` | TensorRT 推理封装，加载 engine、letterbox、NMS、坐标还原。CLI 单图推理测试。 |
| `server/` | 阶段五 TCP 服务代码位置（待开发）。 |
| `tests/test_images/` | 单图推理测试用图。`bus.jpg` 验证后处理正确性，`random.jpg` 跑 baseline FPS。 |
| `outputs/` | 推理结果图（画框后）保存位置。 |
| `models/baseline/` | YOLOv8n/s 三种格式产物：`.pt`（训练用）、`.onnx`（中间格式）、`_fp16.engine`（Nano 推理用）。同名 `.log` 为 trtexec 编译与基准日志。 |
| `models/ultralytics/` | Ultralytics YOLOv8 官方源码备份。实际 `import ultralytics` 走系统 pip 安装包，此目录仅供训练/导出时参考。 |

## 环境

- Jetson Orin Nano Super 8G / JetPack 6.2.1 / Ubuntu 22.04.5
- CUDA 12.6 / cuDNN 9.x / **TensorRT 10.3.0**
- PyTorch 2.5.0a0 / Ultralytics 8.3.159 / OpenCV 4.10.0 / Python 3.10

## 快速使用

### 单图推理（默认 yolov8s FP16）

```bash
cd /home/nvidia/defect_detection
python3 inference/trt_infer.py --image tests/test_images/bus.jpg --loop 5
```

### 切换 yolov8n

```bash
python3 inference/trt_infer.py \
  --engine models/baseline/yolov8n_fp16.engine \
  --image tests/test_images/random.jpg \
  --loop 10
```

### 常用参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--engine` | `models/baseline/yolov8s_fp16.engine` | engine 路径 |
| `--image` | （必填） | 输入图片 |
| `--conf` | 0.25 | 置信度阈值 |
| `--iou` | 0.45 | NMS IoU 阈值 |
| `--output` | `outputs/test_output.jpg` | 结果图保存路径 |
| `--loop` | 1 | 循环推理次数（测速） |

## 性能基准（FP16 engine）

| 模型 | GPU 推理 | 全流程 | FPS |
|------|----------|--------|-----|
| YOLOv8n | ~6 ms | ~20 ms | ~50 |
| YOLOv8s | ~20 ms | ~28–31 ms | ~32–36 |

## 重新生成 engine（如需）

```bash
# ONNX 导出（PC 端）
yolo export model=yolov8s.pt format=onnx dynamic=True

# engine 编译（必须在 Nano 上执行）
trtexec --onnx=yolov8s.onnx --saveEngine=yolov8s_fp16.engine \
  --fp16 --memPoolSize=workspace:64
```

> 注意：engine 不可跨设备使用，必须在 Orin Nano 上编译。

## 阶段进度

✅ 阶段一 硬件 + 系统 + VNC/SSH
✅ 阶段二 环境盘点 + 依赖确认
✅ 阶段三 ONNX → TensorRT 引擎
✅ 阶段四 Python 推理脚本
✅ 阶段五 TCP 通信联调
🔄 阶段六 上位机 GUI
🔄 阶段七 缺陷数据集训练
🔄 阶段八 产线对接