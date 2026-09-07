#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorRT YOLOv8 推理封装（TensorRT 10.x API）
不依赖 pycuda，使用 ctypes 直接调用 libcudart.so 进行显存管理。
适用平台：NVIDIA Jetson Orin Nano / JetPack 6.x
"""

import os
import time
import ctypes
import ctypes.util
import numpy as np
import cv2
import tensorrt as trt


# ---------- CUDA Runtime 封装 ----------
_cudart_path = ctypes.util.find_library("cudart")
if not _cudart_path:
    for _p in [
        "/usr/local/cuda/lib64/libcudart.so.12",
        "/usr/local/cuda/lib64/libcudart.so",
        "/usr/lib/aarch64-linux-gnu/libcudart.so.12",
        "/usr/lib/aarch64-linux-gnu/libcudart.so",
    ]:
        if os.path.exists(_p):
            _cudart_path = _p
            break
if not _cudart_path:
    raise RuntimeError("找不到 libcudart.so，请确认 CUDA 已安装")

cudart = ctypes.CDLL(_cudart_path)
cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
cudart.cudaMalloc.restype = ctypes.c_int
cudart.cudaFree.argtypes = [ctypes.c_void_p]
cudart.cudaFree.restype = ctypes.c_int
cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
cudart.cudaMemcpy.restype = ctypes.c_int
cudart.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
cudart.cudaMemcpyAsync.restype = ctypes.c_int
cudart.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
cudart.cudaStreamCreate.restype = ctypes.c_int
cudart.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
cudart.cudaStreamSynchronize.restype = ctypes.c_int
cudart.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
cudart.cudaStreamDestroy.restype = ctypes.c_int
cudart.cudaHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
cudart.cudaHostAlloc.restype = ctypes.c_int
cudart.cudaFreeHost.argtypes = [ctypes.c_void_p]
cudart.cudaFreeHost.restype = ctypes.c_int
# cudaGetErrorString 返回 const char*（单参数）；原实现误传两个参数，错误路径会崩溃
cudart.cudaGetErrorString.argtypes = [ctypes.c_int]
cudart.cudaGetErrorString.restype = ctypes.c_char_p

# cudaHostAlloc flags
cudaHostAllocDefault = 0x00


def host_alloc_pinned(shape, dtype):
    """分配页锁定（pinned）host 内存，加速异步拷贝。"""
    arr = np.empty(shape, dtype=dtype)
    ptr = ctypes.c_void_p()
    _check_cuda(cudart.cudaHostAlloc(ctypes.byref(ptr), arr.nbytes, cudaHostAllocDefault), "cudaHostAlloc")
    # 用 ctypes 指针创建 numpy array view
    addr = ptr.value
    pinned = np.ctypeslib.as_array(
        (ctypes.c_char * arr.nbytes).from_address(addr)
    ).view(dtype).reshape(shape)
    return pinned, ptr

cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToHost = 2


def _check_cuda(err, msg=""):
    if err != 0:
        err_str = cudart.cudaGetErrorString(err)
        raise RuntimeError(
            f"CUDA 错误 {err}: {(err_str.decode() if err_str else 'unknown')} {msg}")


def cuda_malloc(size):
    ptr = ctypes.c_void_p()
    _check_cuda(cudart.cudaMalloc(ctypes.byref(ptr), size), "cudaMalloc")
    return ptr


def cuda_free(ptr):
    if ptr and ptr.value:
        cudart.cudaFree(ptr)


def cuda_memcpy_htod(dst, src):
    _check_cuda(cudart.cudaMemcpy(dst, src.ctypes.data, src.nbytes, cudaMemcpyHostToDevice), "memcpy HtoD")


def cuda_memcpy_dtoh(dst, src):
    _check_cuda(cudart.cudaMemcpy(dst.ctypes.data, src, dst.nbytes, cudaMemcpyDeviceToHost), "memcpy DtoH")


def cuda_memcpy_async_htod(dst, src, stream):
    _check_cuda(cudart.cudaMemcpyAsync(dst, src.ctypes.data, src.nbytes, cudaMemcpyHostToDevice, stream), "memcpy async HtoD")


def cuda_memcpy_async_dtoh(dst, src, stream):
    _check_cuda(cudart.cudaMemcpyAsync(dst.ctypes.data, src, dst.nbytes, cudaMemcpyDeviceToHost, stream), "memcpy async DtoH")


# ---------- TensorRT 推理类 ----------
class TRTInfer:
    """TensorRT 推理封装：加载 .engine，完成 YOLOv8 预处理/推理/NMS 后处理。"""

    def __init__(self, engine_path, input_shape=(1, 3, 640, 640), conf_thres=0.25, iou_thres=0.45):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine_path = engine_path
        self.input_shape = input_shape
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(
                f"engine 反序列化失败（文件损坏或平台不兼容）: {engine_path}")

        self.context = self.engine.create_execution_context()

        # 绑定全部 tensor：引擎可能含中间输出（如 onnx::Shape_699），
        # execute_async_v3 要求每个输出都有地址，否则 enqueue 报
        # "Neither address or allocator is set for output tensor ..."
        n_io = self.engine.num_io_tensors
        self.input_name = None
        self.output_name = None
        self._output_host_ptrs = []  # 额外输出 pinned host 指针（__del__ 释放）
        self._output_devices = []
        self._output_hosts = []
        self._extra_names = []
        self._extra_shapes = []
        for i in range(n_io):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                if self.output_name is None:
                    self.output_name = name  # 主输出（后处理用）
                else:
                    self._extra_names.append(name)  # 额外输出（只需占位绑定）
        if self.input_name is None or self.output_name is None:
            raise RuntimeError(f"引擎无输入/输出 tensor: {engine_path}")

        # 动态读取引擎真实输入 shape（兼容非 640 静态输入，如 SDX 256x256）
        try:
            _eng_shape = tuple(self.engine.get_tensor_shape(self.input_name))
            if -1 not in _eng_shape and _eng_shape[2] > 0 and _eng_shape[3] > 0:
                self.input_shape = _eng_shape
        except Exception:
            pass

        self.context.set_input_shape(self.input_name, self.input_shape)
        self.output_shape = self.context.get_tensor_shape(self.output_name)
        if -1 in self.output_shape:
            raise RuntimeError(f"主输出 shape 未确定: {self.output_name} {self.output_shape}")

        # 输入绑定
        self.input_host, self.input_host_ptr = host_alloc_pinned(self.input_shape, np.float32)
        self.input_device = cuda_malloc(self.input_host.nbytes)
        self.context.set_tensor_address(self.input_name, self.input_device.value)

        # 主输出绑定
        self.output_host, self.output_host_ptr = host_alloc_pinned(self.output_shape, np.float32)
        self.output_device = cuda_malloc(self.output_host.nbytes)
        self.context.set_tensor_address(self.output_name, self.output_device.value)

        # 额外输出：分配最小 1 字节占位（推理只读主输出，无需真实大小）
        for name in self._extra_names:
            try:
                shp = self.context.get_tensor_shape(name)
                if -1 in shp:
                    shp = (1,)  # 动态 shape 无法解析，给最小占位
                tmp_host, tmp_host_ptr = host_alloc_pinned(shp, np.float32)
                tmp_dev = cuda_malloc(tmp_host.nbytes)
                self._output_hosts.append(tmp_host)
                self._output_host_ptrs.append(tmp_host_ptr)  # 保存指针供 __del__ 释放
                self._output_devices.append(tmp_dev)
                self._extra_shapes.append(shp)
                self.context.set_tensor_address(name, tmp_dev.value)
            except Exception as e:
                raise RuntimeError(f"额外输出 {name} 绑定失败: {e}")

        self.bindings = [self.input_device.value, self.output_device.value]

        self.stream = ctypes.c_void_p()
        _check_cuda(cudart.cudaStreamCreate(ctypes.byref(self.stream)), "cudaStreamCreate")

    def preprocess(self, img_bgr):
        """letterbox + BGR->RGB + /255 + HWC->CHW"""
        h0, w0 = img_bgr.shape[:2]
        target_h, target_w = self.input_shape[2], self.input_shape[3]

        scale = min(target_w / w0, target_h / h0)
        new_w, new_h = int(w0 * scale), int(h0 * scale)
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2

        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)
        return blob, scale, (pad_w, pad_h), (h0, w0)

    def infer(self, img_bgr):
        t0 = time.perf_counter()
        blob, scale, (pad_w, pad_h), raw_shape = self.preprocess(img_bgr)
        t1 = time.perf_counter()

        np.copyto(self.input_host, blob)
        cuda_memcpy_async_htod(self.input_device.value, self.input_host, self.stream.value)
        t1b = time.perf_counter()

        self.context.execute_async_v3(stream_handle=self.stream.value)
        _check_cuda(cudart.cudaStreamSynchronize(self.stream), "stream sync after infer")
        t2 = time.perf_counter()

        cuda_memcpy_async_dtoh(self.output_host, self.output_device.value, self.stream.value)
        _check_cuda(cudart.cudaStreamSynchronize(self.stream), "stream sync after DtoH")
        t3 = time.perf_counter()

        detections = self.postprocess(self.output_host, scale, pad_w, pad_h, raw_shape)
        t4 = time.perf_counter()

        return {
            "detections": detections,
            "timing": {
                "preprocess_ms": (t1 - t0) * 1000,
                "memcpy_htod_ms": (t1b - t1) * 1000,
                "gpu_inference_ms": (t2 - t1b) * 1000,
                "memcpy_dtoh_ms": (t3 - t2) * 1000,
                "postprocess_ms": (t4 - t3) * 1000,
                "total_ms": (t4 - t0) * 1000,
            },
        }

    def postprocess(self, output, scale, pad_w, pad_h, raw_shape):
        """YOLOv8 输出 (1, 4+num_classes, 8400) -> NMS -> 坐标还原"""
        output = np.squeeze(output)
        # 通用化：输出可能是 (anchors, 4+classes) 或 (4+classes, anchors)，
        # 用 shape 比较判断（nc 恒远小于 anchors 数），不再硬编码 10/29/39/84
        if output.ndim == 2 and output.shape[0] < output.shape[1]:
            output = output.transpose(1, 0)

        num_anchors = output.shape[0]
        boxes = output[:, :4]
        scores = output[:, 4:]
        cls_ids = np.argmax(scores, axis=1)
        max_scores = scores[np.arange(num_anchors), cls_ids]

        mask = max_scores >= self.conf_thres
        boxes = boxes[mask]
        scores = max_scores[mask]
        cls_ids = cls_ids[mask]

        if len(boxes) == 0:
            return []

        xyxy = np.zeros_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

        # ⚠️ cv2.dnn.NMSBoxes 期望 (x, y, w, h)，传 xyxy 会算错 IoU 导致 NMS 结果错误
        boxes_xywh = [[float(b[0]), float(b[1]),
                       float(b[2] - b[0]), float(b[3] - b[1])] for b in xyxy]
        indices = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(), self.conf_thres, self.iou_thres)
        indices = indices.flatten() if len(indices) > 0 else []

        h0, w0 = raw_shape
        results = []
        for i in indices:
            x1, y1, x2, y2 = xyxy[i]
            x1 -= pad_w
            y1 -= pad_h
            x2 -= pad_w
            y2 -= pad_h
            x1 /= scale
            y1 /= scale
            x2 /= scale
            y2 /= scale
            x1 = max(0, min(w0 - 1, x1))
            y1 = max(0, min(h0 - 1, y1))
            x2 = max(0, min(w0 - 1, x2))
            y2 = max(0, min(h0 - 1, y2))

            results.append({
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": float(scores[i]),
                "class_id": int(cls_ids[i]),
                "result": "NG" if scores[i] >= self.conf_thres else "OK",
            })
        return results

    def __del__(self):
        if self.stream and self.stream.value:
            cudart.cudaStreamDestroy(self.stream)
        cuda_free(self.input_device)
        cuda_free(self.output_device)
        # 额外输出：之前只释放主输入/输出，额外输出显存/页锁定内存全部泄漏
        for dev in getattr(self, "_output_devices", []):
            cuda_free(dev)
        for hp in getattr(self, "_output_host_ptrs", []):
            if hp and hp.value:
                try:
                    cudart.cudaFreeHost(hp)
                except Exception:
                    pass
        if self.input_host_ptr and self.input_host_ptr.value:
            cudart.cudaFreeHost(self.input_host_ptr)
        if self.output_host_ptr and self.output_host_ptr.value:
            cudart.cudaFreeHost(self.output_host_ptr)


def draw_detections(img_bgr, detections, labels=None):
    img = img_bgr.copy()
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        conf = d["confidence"]
        cls_id = d["class_id"]
        label = labels[cls_id] if labels and cls_id < len(labels) else f"cls{cls_id}"
        text = f"{label} {conf:.2f}"
        color = (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, text, (x1, max(y1 - 5, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TensorRT YOLOv8 推理测试")
    parser.add_argument("--engine", default="/home/nvidia/defect_detection/models/baseline/yolov8s_fp16.engine", help="TensorRT engine 路径")
    parser.add_argument("--image", required=True, help="输入图片路径")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值")
    parser.add_argument("--output", default="/home/nvidia/defect_detection/outputs/test_output.jpg", help="画框结果保存路径")
    parser.add_argument("--loop", type=int, default=1, help="循环推理次数（测速用）")
    args = parser.parse_args()

    if not os.path.exists(args.engine):
        raise FileNotFoundError(f"engine 不存在: {args.engine}")
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"图片不存在: {args.image}")

    img = cv2.imread(args.image)
    if img is None:
        raise ValueError(f"无法读取图片: {args.image}")

    print(f"[INFO] 加载 engine: {args.engine}")
    infer = TRTInfer(args.engine, conf_thres=args.conf, iou_thres=args.iou)

    for _ in range(3):
        infer.infer(img)

    times = []
    for _ in range(args.loop):
        result = infer.infer(img)
        times.append(result["timing"]["gpu_inference_ms"])

    detections = result["detections"]
    print(f"[INFO] 检测到 {len(detections)} 个目标")
    for d in detections:
        print(f"  box={d['box']}, conf={d['confidence']:.3f}, cls={d['class_id']}, result={d['result']}")

    print(f"[INFO] GPU 推理耗时: mean={np.mean(times):.2f}ms, min={np.min(times):.2f}ms, max={np.max(times):.2f}ms")
    print(f"[INFO] 全流程耗时: {result['timing']['total_ms']:.2f}ms")
    print(f"[INFO] 单次时间拆分（最后一帧）:")
    for k, v in result["timing"].items():
        print(f"  {k}: {v:.2f}ms")

    out_img = draw_detections(img, detections)
    cv2.imwrite(args.output, out_img)
    print(f"[INFO] 结果图已保存: {args.output}")
