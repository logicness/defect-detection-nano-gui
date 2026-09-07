#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下位机 TCP 推理服务（asyncio）
接收上位机发来的图片，调用 TensorRT 推理，返回检测结果。

协议：4字节大端长度包头 + UTF-8 JSON 载荷
消息类型：
  - detect_request:   { type, image_base64 }
  - detect_response:  { type, detections, timing, ok }
  - control:          { type, conf_thres, iou_thres }
  - heartbeat:        { type } / { type: heartbeat_ack }
  - model_list_request / model_list_response
  - model_load_request / model_load_response
  - model_upload_start / model_upload_chunk / model_upload_end / model_upload_response
  - model_delete_request / model_delete_response
  - model_status     （编译进度主动推送）
  - nano_image_dir_request / nano_image_dir_response   （下位机图片目录 查询/设置）
  - nano_images_request / nano_images_response         （下位机图片列表 + 缩略图）
  - nano_image_request / nano_image_response           （单张预览大图，按需拉取）
  - nano_detect_request / nano_detect_response         （单张检测：下位机本地图片 + 当前激活模型）
  - nano_batch_request / nano_batch_response           （批量检测会话 启动）
  - nano_batch_stop_request / nano_batch_stop_response （批量检测 中止）
  - nano_batch_status_request / nano_batch_status_response（批量检测 状态查询）
  - nano_batch_progress / nano_batch_item / nano_batch_done（批量检测 主动推送）
  - stream_subscribe_request / stream_subscribe_response（产线流订阅/退订）
  - stream_control_request / stream_control_response   （产线流控制 start/stop/set_fps）
  - stream_frame / stream_state                        （产线流 主动推送 帧/状态）
"""

import asyncio
import json
import re
import subprocess
import struct
import threading
import time
import base64
import logging
import argparse
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2

# jetson-stats（仅 Nano 上有）；缺失时回退 tegrastats 采样
try:
    import jtop as _jtop_mod
    HAS_JTOP = True
except ImportError:
    HAS_JTOP = False

# 添加 server / inference 目录到 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "inference"))
from model_manager import ModelManager, _rel_name
from image_store import ImageStore, DEFAULT_IMAGE_DIR, _safe_basename
from camera_sim import CameraSimulator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
# 2026-09-01：启用按天轮转文件日志（logs/infer_server_YYYYMMDD.log，保留 30 天），
# 根治 systemd append 到 server.log 无轮转导致的磁盘长期增长；
# console 输出由 systemd 接管（journal 按大小自动轮转）。
try:
    from log_config import setup_logging
    setup_logging()   # TimedRotatingFileHandler(按天, 30天) + console
except Exception as e:
    print(f"[log] setup_logging 失败(忽略): {e}", file=sys.stderr)
log = logging.getLogger("server")

# N3/N4 输入健壮性：帧大小上限（base64，原图约 30MB）与单帧推理超时（秒）
MAX_IMAGE_B64 = 40 * 1024 * 1024
DETECT_TIMEOUT_S = 10.0
# 2026-09-01 修复：事件循环冻结判定阈值——循环心跳超过该秒数未刷新，
# loop-watchdog 线程强制退出进程（systemd 自动拉起），把"永久静默"降级为"自动恢复"。
LOOP_FREEZE_EXIT_S = 60.0
# 批量检测护栏：单张超时沿用 DETECT_TIMEOUT_S；批量总时长上限（超时强制中止防挂死）
BATCH_MAX_MINUTES = 30
BATCH_MAX_ROUNDS = 20   # 轮数上限（防异常请求跑死 Nano）


class FrameProtocol:
    """4字节大端长度包头 + UTF-8 JSON 载荷"""

    MAX_FRAME = 50 * 1024 * 1024  # 50MB 上限（分片上传 1MB/片远小于此）

    @staticmethod
    def pack(obj: dict) -> bytes:
        payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        return struct.pack(">I", len(payload)) + payload

    @staticmethod
    async def read(reader: asyncio.StreamReader) -> dict:
        header = await reader.readexactly(4)
        length = struct.unpack(">I", header)[0]
        if length <= 0 or length > FrameProtocol.MAX_FRAME:
            raise ValueError(f"帧长度非法: {length}")
        payload = await reader.readexactly(length)
        return json.loads(payload.decode("utf-8"))


class DetectServer:
    def __init__(self, model_manager: ModelManager, host: str = "0.0.0.0",
                 port: int = 8888, image_dir: str = "", source: str = "sim"):
        self.mm = model_manager
        self.host = host
        self.port = port
        self._engine_lock = asyncio.Lock()   # detect / load / 编译后加载 互斥
        self._status_writer = None           # 编译进度推送目标连接
        self._status_task = None             # 编译完成后的自动加载协程
        self._loop = None                    # 主事件循环（start() 中捕获）
        # 下位机图片检测（2026-08-20）：固定图片文件夹 + 批量检测会话
        self._store = ImageStore(default_dir=image_dir or DEFAULT_IMAGE_DIR)
        self._batch = None                   # 批量会话 dict（同时只允许一个）
        # 单 worker 推理线程池：detect/load 严格串行。
        # 1) detect 超时后旧任务仍在后台跑，若直接释放锁会让新 detect 并发操作同一
        #    TRTInfer（共享 input_host/output_host/context/stream）→ 数据竞争；
        #    单 worker 队列保证串行，超时只是跳过本次响应，任务继续排队执行。
        # 2) model_load（engine 反序列化 + 预热推理）不再阻塞事件循环。
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="trt")
        # 2026-09-01 修复：状态采集 / 缩略图 / 预览 / 标注图等 CPU+SDK 重活专用线程池，
        # 全部移出事件循环执行（避免 MVS SDK 同步调用卡死 asyncio 循环，见事故记录）。
        self._status_exec = ThreadPoolExecutor(max_workers=1, thread_name_prefix="status")
        # 事件循环心跳（loop-watchdog 判据）：每次收到客户端消息刷新；
        # 独立线程检测冻结 >LOOP_FREEZE_EXIT_S 则强制退出，由 systemd 自动拉起。
        self._loop_beat = time.monotonic()
        # jtop GPU 采样：真实利用率 + 滚动窗口（捕捉推理突发，读取零阻塞）。
        # jtop.service 可能晚于本服务启动（开机/重启时约晚 10s+），
        # 因此把初始化放进采样线程持续重试，避免一次性失败后整个进程生命周期
        # 都回退到 tegrastats(GR3D_FREQ=0)。
        self._jtop = None
        self._gpu_ring = []                  # 最近 ~6s 的瞬时 load（0.5s 一采样）
        self._stop_gpu_sampler = False
        if HAS_JTOP:
            threading.Thread(target=self._gpu_sampler, daemon=True,
                             name="gpu-sampler").start()
        # 产线相机（2026-08-24 图片模拟 / 2026-08-28 真实相机）：独立检测 → 主动推送
        # 传入 gpu_sampler 以便在 stream_frame 中实时推送 GPU 利用率
        self._camera_sim = CameraSimulator(
            self._store, self.mm, self._executor, self._engine_lock,
            loop=None, fps=10, shuffle=False, gpu_sampler=self,
            source=source if source in ("sim", "camera") else "sim")
        # 2026-09-01 v2：相机侧检测引擎自愈时同步替换本服务共享 executor
        # （单张检测/批量路径同样受益，避免挂死线程池持续占用）
        self._camera_sim._on_executor_replaced = self._replace_detect_executor

    def _replace_detect_executor(self, new_executor):
        """camera_sim 引擎自愈回调：替换共享推理线程池。"""
        self._executor = new_executor
        log.warning("推理线程池已替换（检测引擎自愈）")

    def _gpu_sampler(self):
        """后台线程：启动 jtop 客户端并 0.5s 采样一次真实 GPU load 写入滚动窗口。

        若 jtop.service 尚未就绪导致初始化失败，持续重试（成功后恢复真实读数），
        而不是永久回退 tegrastats。"""
        _fail_logged = False
        while True:
            if self._stop_gpu_sampler:
                break
            try:
                if self._jtop is None:
                    try:
                        j = _jtop_mod.jtop(interval=0.5)
                        j.start()
                        self._jtop = j
                        _fail_logged = False
                        log.info("jtop GPU 采样已启用")
                    except Exception as e:
                        if not _fail_logged:
                            log.warning(f"jtop 初始化失败，将后台重试: {e}")
                            _fail_logged = True
                # ok(spin=True)：非阻塞检查客户端存活（ok() 默认会阻塞等数据，不能用于采样线程）
                if self._jtop is not None and self._jtop.ok(spin=True):
                    load = float(self._jtop.gpu["gpu"]["status"]["load"])
                    self._gpu_ring.append(load)
                    if len(self._gpu_ring) > 12:
                        self._gpu_ring.pop(0)
            except Exception:
                pass
            time.sleep(0.5)

    def get_current_load(self) -> float:
        """获取当前 GPU 负载（供 CameraSimulator 实时推送使用）。
        
        返回最近一次采样的瞬时值，如果不可用返回 0.0。
        """
        try:
            if self._jtop is not None and self._jtop.ok(spin=True):
                return float(self._jtop.gpu["gpu"]["status"]["load"])
        except Exception:
            pass
        return 0.0

    def close(self):
        """停止后台采样并释放 jtop/线程池（进程退出时调用）"""
        self._stop_gpu_sampler = True
        # 取消事件循环活络心跳（2026-09-01 16:2x）
        hb = getattr(self, "_loop_heartbeat_task", None)
        if hb is not None and self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(hb.cancel)
            except Exception:
                pass
        # 停止产线相机任务（进程退出前清理）
        try:
            if self._camera_sim.running:
                self._camera_sim.running = False
                task = self._camera_sim._task
                if task is not None and self._loop is not None:
                    self._loop.call_soon_threadsafe(task.cancel)
            self._camera_sim.close()   # 释放相机句柄 + 读帧线程池（幂等）
        except Exception:
            pass
        if getattr(self, "_batch", None) is not None:
            s = self._batch
            s["stop"] = True
            self._batch = None
            if s.get("task") is not None:
                try:
                    s["task"].cancel()
                except Exception:
                    pass
        if self._jtop is not None:
            try:
                self._jtop.close()
            except Exception:
                pass
            self._jtop = None
        if getattr(self, "_executor", None) is not None:
            try:
                self._executor.shutdown(wait=False)
            except Exception:
                pass
        if getattr(self, "_status_exec", None) is not None:
            try:
                self._status_exec.shutdown(wait=False)
            except Exception:
                pass

    def _loop_watchdog(self):
        """事件循环健康看门狗（2026-09-01 修复，独立线程，不依赖事件循环）。

        事故复盘：asyncio 循环被同步阻塞调用（MVS SDK / 缩略图 / subprocess 等）卡死后，
        进程内所有协程与看门狗一并失效，服务"活着但全断"（8888 accept 队列堆满 →
        客户端 timed out），最长无人值守 2 小时。本线程每 5s 检查一次心跳，
        冻结超过 LOOP_FREEZE_EXIT_S 秒直接 os._exit(1)，systemd Restart=always 5s 后拉起，
        GUI 自动重连，把"永久静默故障"降级为"最多 1 分钟自动恢复"。"""
        while True:
            time.sleep(5.0)
            try:
                if time.monotonic() - self._loop_beat > LOOP_FREEZE_EXIT_S:
                    print(f"[loop-watchdog] 事件循环冻结 >{LOOP_FREEZE_EXIT_S}s，"
                          f"强制退出触发 systemd 重启", file=sys.stderr, flush=True)
                    os._exit(1)
            except Exception:
                pass

    async def _loop_heartbeat(self):
        """事件循环活络心跳（2026-09-01 16:2x 修复伪阳性）：
        只在事件循环真正推进时才会每 1s 刷新 _loop_beat。
        - 循环健康但无任何客户端消息时，心跳仍刷新 → 不被 watchdog 误杀；
        - 循环真被同步调用卡死时，心跳协程同样被饿死 → loop_beat 停更 → watchdog 照常兜底。
        这样 _loop_beat 反映的是「事件循环是否活着」而非「有没有客户端发包」。"""
        try:
            while True:
                await asyncio.sleep(1)
                self._loop_beat = time.monotonic()
        except asyncio.CancelledError:
            pass

    async def start(self):
        # 捕获主事件循环引用，供编译子线程安全回调（子线程无 event loop）
        self._start_ts = time.time()
        self._last_timing = None
        self._loop = asyncio.get_running_loop()
        self._camera_sim.loop = self._loop   # 产线模拟相机绑定事件循环
        self._loop_beat = time.monotonic()
        threading.Thread(target=self._loop_watchdog, daemon=True,
                         name="loop-watchdog").start()
        self._loop_heartbeat_task = asyncio.ensure_future(self._loop_heartbeat())
        server = await asyncio.start_server(self._handle_client, self.host, self.port)
        addrs = ", ".join(str(s.getsockname()) for s in server.sockets)
        log.info(f"TCP Server 监听 {addrs}")
        async with server:
            await server.serve_forever()

    # ---------- 消息路由 ----------
    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        peer = writer.get_extra_info("peername")
        log.info(f"客户端连接: {peer}")
        try:
            while True:
                msg = await FrameProtocol.read(reader)
                msg_type = msg.get("type", "")
                self._loop_beat = time.monotonic()   # 事件循环心跳（loop-watchdog 判据）

                if msg_type == "heartbeat":
                    await self._send(writer, {"type": "heartbeat_ack"})
                    continue

                if msg_type == "control":
                    self.mm.engine.update_params(
                        conf_thres=msg.get("conf_thres"),
                        iou_thres=msg.get("iou_thres"),
                    )
                    if msg.get("rois") is not None:
                        self.mm.set_rois(msg.get("rois"))   # ROI 检测门控（持久化，重启恢复）
                    log.info(f"参数更新: conf={self.mm.engine.conf_thres}, iou={self.mm.engine.iou_thres}, roi={len(self.mm.engine.rois)}")
                    await self._send(writer, {"type": "control_ack", "ok": True,
                                              "roi_count": len(self.mm.engine.rois)})
                    continue

                if msg_type == "detect_request":
                    await self._handle_detect(writer, msg)
                    continue

                if msg_type == "model_list_request":
                    data = self.mm.list_models()
                    await self._send(writer, {"type": "model_list_response", "ok": True, **data})
                    continue

                if msg_type == "model_load_request":
                    await self._handle_model_load(writer, msg)
                    continue

                if msg_type == "model_upload_start":
                    await self._handle_upload_start(writer, msg)
                    continue

                if msg_type == "model_upload_chunk":
                    await self._handle_upload_chunk(writer, msg)
                    continue

                if msg_type == "model_upload_end":
                    await self._handle_upload_end(writer, msg)
                    continue

                if msg_type == "model_delete_request":
                    r = self.mm.delete(msg.get("model", ""))
                    await self._send(writer, {"type": "model_delete_response", **r, "model": msg.get("model", "")})
                    continue

                if msg_type == "status_request":
                    # 2026-09-01：客户端写缓冲过大（未及时读取，可能卡死）→ 跳过本次状态推送，
                    # 避免 drain 阻塞/断连风暴（原实现此处出现 12 连发 Connection lost）。
                    # 2026-09-01 修复：状态采集（相机温度=同步 MVS SDK 调用 / tegrastats /
                    # top 子进程）移入独立线程池执行，带 3s 超时——SDK 卡死不再冻结事件循环。
                    if self._client_write_ok(writer):
                        try:
                            status = await asyncio.wait_for(
                                asyncio.get_event_loop().run_in_executor(
                                    self._status_exec, self._get_system_status),
                                timeout=3.0)
                            await self._send(writer, status)
                        except asyncio.TimeoutError:
                            log.warning("状态采集超时(3s)，跳过本次推送")
                        except Exception as e:
                            log.warning(f"状态采集异常: {e}")
                    continue

                if msg_type == "nano_image_dir_request":
                    await self._handle_nano_image_dir(writer, msg)
                    continue

                if msg_type == "nano_images_request":
                    await self._handle_nano_images(writer, msg)
                    continue

                if msg_type == "nano_image_request":
                    await self._handle_nano_image(writer, msg)
                    continue

                if msg_type == "nano_detect_request":
                    await self._handle_nano_detect(writer, msg)
                    continue

                if msg_type == "nano_batch_request":
                    await self._handle_nano_batch(writer, msg)
                    continue

                if msg_type == "nano_batch_stop_request":
                    await self._handle_nano_batch_stop(writer, msg)
                    continue

                if msg_type == "nano_batch_status_request":
                    await self._handle_nano_batch_status(writer, msg)
                    continue

                if msg_type == "stream_subscribe_request":
                    await self._handle_stream_subscribe(writer, msg)
                    continue

                if msg_type == "stream_control_request":
                    await self._handle_stream_control(writer, msg)
                    continue

                log.warning(f"未知消息类型: {msg_type}")

        except asyncio.IncompleteReadError:
            log.info(f"客户端断开: {peer}")
        except Exception as e:
            log.error(f"处理异常: {e}", exc_info=True)
        finally:
            # 产线流订阅者断连：移除订阅（自动停止向该连接推送）
            self._camera_sim.unsubscribe(writer)
            if self._status_writer is writer:
                self._status_writer = None
                # 上传会话中断（断连/异常）：清理半成品临时文件，防残留累积
                try:
                    self.mm.upload_abort()
                except Exception:
                    pass
            # 批量检测会话绑定本连接：断连即中止任务（防任务悬挂在已死连接上）
            if self._batch is not None and self._batch.get("writer") is writer:
                s = self._batch
                s["stop"] = True
                self._batch = None
                task = s.get("task")
                if task is not None:
                    try:
                        task.cancel()
                    except Exception:
                        pass
                log.info(f"客户端断开，批量检测会话已中止: {s.get('session_id')}")
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    # ---------- handler ----------
    async def _handle_detect(self, writer, msg):
        if self.mm.compiling:
            await self._send(writer, {"type": "detect_response", "ok": False,
                                      "error": "busy: 模型编译中，请稍候"})
            return
        t0 = time.perf_counter()
        img_b64 = msg.get("image_base64", "")
        if not img_b64:
            await self._send(writer, {"type": "detect_response", "ok": False, "error": "missing image"})
            return
        # N3 帧大小上限：base64 长度 > 40MB（原图约 30MB）直接拒绝，防脏数据拖垮服务
        if len(img_b64) > MAX_IMAGE_B64:
            await self._send(writer, {"type": "detect_response", "ok": False,
                                      "error": "image too large"})
            return
        img_bytes = base64.b64decode(img_b64)
        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if img is None:
            await self._send(writer, {"type": "detect_response", "ok": False, "error": "decode failed"})
            return

        # N4 单帧超时保护 + 同步推理丢单 worker 线程池（不阻塞事件循环，
        # 超时只跳过本次响应；旧任务继续排队执行，与后续 detect/load 天然串行，无并发竞争）
        async with self._engine_lock:
            try:
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(self._executor, self.mm.engine.detect, img),
                    timeout=DETECT_TIMEOUT_S)
            except asyncio.TimeoutError:
                log.error(f"推理超时({DETECT_TIMEOUT_S}s)，帧已跳过")
                await self._send(writer, {"type": "detect_response", "ok": False,
                                          "error": "detect timeout"})
                return
            except Exception as e:
                log.error(f"推理异常: {e}")
                await self._send(writer, {"type": "detect_response", "ok": False,
                                          "error": f"detect error: {e}"})
                return

        t_total = (time.perf_counter() - t0) * 1000
        self._last_timing = result.get("timing", {})
        resp = {
            "type": "detect_response",
            "ok": True,
            "detections": result["detections"],
            "timing": {**result["timing"], "total_with_decode": round(t_total, 2)},
            "image_size": {"w": img.shape[1], "h": img.shape[0]},
            "model": os.path.basename(self.mm.current),
        }
        await self._send(writer, resp)
        n = len(result["detections"])
        log.info(f"检测完成: {n} 目标, {t_total:.1f}ms, peer={writer.get_extra_info('peername')}")

    def _get_system_status(self) -> dict:
        """System status collector (status_request response).
        jtop 常驻采样 GPU 真实利用率/温度/功耗/CPU，mem 读 /proc/meminfo，
        last detect timing if available；jtop 不可用时回退 tegrastats one-shot。"""
        status = {
            "type": "status_response",
            "model": os.path.basename(self.mm.current),
            "gpu_util": None,
            "gpu_temp": None,
            "power_mw": None,
            "cpu_util": None,
            "mem_used_gb": None,
            "mem_total_gb": None,
            "last_detect_ms": None,
            "camera_temp_c": self._camera_sim.camera_temperature(),
            "camera_online": self._camera_sim.camera_online(),
            "uptime_s": int(time.time() - self._start_ts) if getattr(self, "_start_ts", 0) else None,
        }
        if getattr(self, "_jtop", None) is not None and self._jtop.ok(spin=True):
            try:
                # GPU 利用率：2026-09-04 延迟修复——原来取全窗口(6s/12点)最大值，
                # GPU 降载后状态栏仍显示旧峰值，叠加 GUI 5s 轮询最多延迟 ~11s；
                # 改为最近 1.5s（3 个采样点）峰值，既保留捕捉 10~30ms 推理突发
                # 的抗抖能力，又把状态延迟降到 ~3s。
                load = float(self._jtop.gpu["gpu"]["status"]["load"])
                ring = list(self._gpu_ring)[-3:]
                status["gpu_util"] = round(max([load] + ring), 1)
            except Exception:
                pass
            try:
                status["gpu_temp"] = round(float(self._jtop.temperature["gpu"]["temp"]), 1)
            except Exception:
                pass
            try:
                status["power_mw"] = int(self._jtop.power["tot"]["power"])
            except Exception:
                pass
            try:
                total = self._jtop.cpu["total"]
                idle = float(total.get("idle", 0.0))
                status["cpu_util"] = round(100.0 - idle, 1)
            except Exception:
                pass
        # jtop 不可用 → 回退 tegrastats one-shot（旧逻辑）
        if status["gpu_util"] is None:
            try:
                out = subprocess.run(
                    ["timeout", "3", "tegrastats", "--interval", "1000"],
                    capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
                line = out[-1] if out else ""
                m = re.search(r"GR3D_FREQ (\d+)%", line)
                if m:
                    status["gpu_util"] = int(m.group(1))
                m = re.search(r"gpu@([\d.]+)C", line)
                if m:
                    status["gpu_temp"] = float(m.group(1))
                m = re.search(r"VDD_IN ([\d]+)mW", line)
                if m:
                    status["power_mw"] = int(m.group(1))
            except Exception as e:
                log.warning(f"tegrastats sample failed: {e}")
        try:
            with open("/proc/meminfo") as f:
                info = {}
                for line in f:
                    k, _, v = line.partition(":")
                    info[k.strip()] = int(v.strip().split()[0]) // 1024  # kB -> MB
            total = info.get("MemTotal", 0)
            avail = info.get("MemAvailable", 0)
            status["mem_total_gb"] = round(total / 1024, 1)
            status["mem_used_gb"] = round((total - avail) / 1024, 1)
        except Exception:
            pass
        if status["cpu_util"] is None:
            try:
                cpu = subprocess.run(
                    ["sh", "-c", "top -bn1 | head -3 | tail -1"],
                    capture_output=True, text=True, timeout=3).stdout
                m = re.search(r"([\d.]+)\s+id", cpu)
                if m:
                    status["cpu_util"] = round(100.0 - float(m.group(1)), 1)
            except Exception:
                pass
        if getattr(self, "_last_timing", None):
            status["last_detect_ms"] = round(
                self._last_timing.get("total_ms", 0), 1)
        return status

    async def _handle_model_load(self, writer, msg):
        name = msg.get("model", "")
        # 2026-09-01 修复：engine 反序列化+预热可能耗时数十秒甚至挂死，
        # 移入推理线程池执行 + 30s 超时（不再同步阻塞事件循环）。
        async with self._engine_lock:
            try:
                r = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        self._executor, self.mm.load, name),
                    timeout=30.0)
            except asyncio.TimeoutError:
                log.error(f"模型加载超时(30s): {name}")
                await self._send(writer, {"type": "model_load_response", "model": name,
                                          "ok": False, "error": "模型加载超时(30s)"})
                return
        if r["ok"]:
            log.info(f"切换模型成功: {name} ({r['load_ms']}ms)")
        else:
            log.warning(f"切换模型失败: {name} -> {r['error']}")
        await self._send(writer, {"type": "model_load_response", "model": name, **r})

    async def _handle_upload_start(self, writer, msg):
        filename = msg.get("filename", "")
        size = int(msg.get("size", 0))
        tmp, err = self.mm.upload_begin(filename, size)
        if err:
            await self._send(writer, {"type": "model_upload_response", "ok": False, "error": err})
            return
        self._status_writer = writer
        log.info(f"上传开始: {filename} ({size} bytes)")
        await self._send(writer, {"type": "model_upload_response", "ok": True,
                                  "state": "started", "filename": filename})

    async def _handle_upload_chunk(self, writer, msg):
        ok, info = self.mm.upload_chunk(msg.get("filename", ""), msg.get("seq", -1), msg.get("data", ""))
        if not ok:
            self.mm.upload_abort()
            await self._send(writer, {"type": "model_upload_response", "ok": False, "error": info})
            return
        await self._send(writer, {"type": "model_upload_response", "ok": True,
                                  "state": "chunk_ack", "seq": msg.get("seq"), "received": info})

    async def _handle_upload_end(self, writer, msg):
        filename = msg.get("filename", "")
        final, err = self.mm.upload_commit(filename, msg.get("checksum", ""))
        if err:
            self._status_writer = None
            await self._send(writer, {"type": "model_upload_response", "ok": False, "error": err})
            return
        ext = os.path.splitext(final)[1].lower()
        if ext in (".onnx", ".pt"):
            await self._send(writer, {"type": "model_upload_response", "ok": True,
                                      "state": "saved", "filename": filename,
                                      "message": "文件已保存，开始编译（期间不可推理）"})
            self.mm.compile_async(final, self._on_compile_progress)
        else:
            # .engine：直接尝试加载（平台不兼容则报错，文件保留可删）
            async with self._engine_lock:
                r = self.mm.load(_rel_name(final))
            if r["ok"]:
                self._status_writer = None
                await self._send(writer, {"type": "model_upload_response", "ok": True,
                                          "state": "ready", "filename": filename,
                                          "message": "engine 已加载", "load_ms": r["load_ms"]})
            else:
                self._status_writer = None
                await self._send(writer, {"type": "model_upload_response", "ok": False,
                                          "error": r["error"], "filename": filename})

    # ---------- 编译进度 ----------
    def _on_compile_progress(self, stage, progress, message):
        """子线程回调 -> 事件循环推送（用主线程捕获的 loop，子线程无 event loop）"""
        loop = getattr(self, "_loop", None)
        if loop is None:
            log.warning("事件循环未就绪，进度推送失败")
            return
        try:
            loop.call_soon_threadsafe(self._push_status, stage, progress, message)
        except RuntimeError:
            log.warning("事件循环已关闭，进度推送失败")

    def _push_status(self, stage, progress, message):
        writer = self._status_writer
        if writer is None or writer.is_closing():
            return
        payload = {"type": "model_status", "stage": stage, "progress": progress, "message": message}
        asyncio.ensure_future(self._send(writer, payload))
        if stage == "compiled":
            # 编译完成：持锁自动加载新 engine
            name = message.split(": ")[-1]
            asyncio.ensure_future(self._auto_load_after_compile(name))

    async def _auto_load_after_compile(self, name):
        async with self._engine_lock:
            r = self.mm.load(name)
        writer = self._status_writer
        if writer is None or writer.is_closing():
            return
        if r["ok"]:
            await self._send(writer, {"type": "model_upload_response", "ok": True,
                                      "state": "ready", "filename": name,
                                      "message": "编译完成并已加载", "load_ms": r["load_ms"]})
            log.info(f"编译完成并加载: {name} ({r['load_ms']}ms)")
        else:
            await self._send(writer, {"type": "model_upload_response", "ok": False,
                                      "error": r["error"], "filename": name})
            log.error(f"编译后加载失败: {name} -> {r['error']}")
        self._status_writer = None

    # ---------- 下位机图片检测（2026-08-20 新增） ----------
    async def _handle_nano_image_dir(self, writer, msg):
        """图片目录 查询/设置（nano_image_dir_request）"""
        action = msg.get("action", "get")
        if action == "set":
            r = self._store.set_dir(msg.get("dir", ""))
            await self._send(writer, {"type": "nano_image_dir_response", **r})
        else:
            info = self._store.info()
            await self._send(writer, {"type": "nano_image_dir_response", "ok": True, **info})

    async def _handle_nano_images(self, writer, msg):
        """图片列表 + 小缩略图（nano_images_request）
        2026-09-01 修复：缩略图批量生成（读图+解码+缩放+编码）是 CPU 重活，
        移入独立线程池 + 15s 超时，避免卡死事件循环（大图库时曾拖垮整个服务）。"""
        try:
            r = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    self._status_exec, self._store.list_images,
                    msg.get("offset", 0), msg.get("limit", 100),
                    msg.get("sort", "name"), msg.get("thumb", True),
                    msg.get("thumb_size", 256)),
                timeout=15.0)
        except asyncio.TimeoutError:
            log.warning("图片列表缩略图生成超时(15s)，返回空列表")
            await self._send(writer, {"type": "nano_images_response", "ok": False,
                                      "dir": self._store.dir,
                                      "error": "图片列表生成超时", "images": []})
            return
        await self._send(writer, {"type": "nano_images_response", "ok": True,
                                  "dir": self._store.dir, **r})

    async def _handle_nano_image(self, writer, msg):
        """单张预览大图，按需拉取（nano_image_request）
        2026-09-01 修复：预览缩略图生成移出事件循环 + 10s 超时。"""
        name = msg.get("name", "")
        try:
            b64, err, w, h = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    self._status_exec, self._store.thumb_b64,
                    name, msg.get("size", 640)),
                timeout=10.0)
        except asyncio.TimeoutError:
            log.warning(f"预览图生成超时(10s): {name}")
            await self._send(writer, {"type": "nano_image_response", "ok": False,
                                      "name": name, "error": "预览图生成超时"})
            return
        if err is not None:
            await self._send(writer, {"type": "nano_image_response", "ok": False,
                                      "name": name, "error": err})
            return
        await self._send(writer, {"type": "nano_image_response", "ok": True,
                                  "name": name, "w": w, "h": h, "image_b64": b64})

    async def _handle_nano_detect(self, writer, msg):
        """单张检测：下位机本地图片 + 当前激活模型（nano_detect_request）"""
        if self.mm.compiling:
            await self._send(writer, {"type": "nano_detect_response", "ok": False,
                                      "error": "busy: 模型编译中，请稍候"})
            return
        name = msg.get("name", "")
        img, err = self._store.read(name)
        if err is not None:
            await self._send(writer, {"type": "nano_detect_response", "ok": False,
                                      "name": name, "error": err})
            return
        t0 = time.perf_counter()
        async with self._engine_lock:
            try:
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(self._executor, self.mm.engine.detect, img),
                    timeout=DETECT_TIMEOUT_S)
            except asyncio.TimeoutError:
                log.error(f"推理超时({DETECT_TIMEOUT_S}s): {name}")
                await self._send(writer, {"type": "nano_detect_response", "ok": False,
                                          "name": name, "error": "detect timeout"})
                return
            except Exception as e:
                log.error(f"推理异常: {e}")
                await self._send(writer, {"type": "nano_detect_response", "ok": False,
                                          "name": name, "error": f"detect error: {e}"})
                return
        t_total = (time.perf_counter() - t0) * 1000
        resp = {
            "type": "nano_detect_response", "ok": True, "name": name,
            "detections": result["detections"],
            "timing": {**result["timing"], "total_with_read": round(t_total, 2)},
            "image_size": {"w": img.shape[1], "h": img.shape[0]},
            "model": os.path.basename(self.mm.current),
        }
        if msg.get("annotate", True):
            # 2026-09-01 修复：标注图生成（画框+编码）移出事件循环 + 10s 超时
            try:
                b64, aerr = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        self._status_exec, self._store.annotate_b64,
                        img, result["detections"], self.mm.engine.labels,
                        msg.get("thumb_size", 640)),
                    timeout=10.0)
            except asyncio.TimeoutError:
                b64, aerr = None, "标注图生成超时"
            if aerr is None:
                resp["annot_b64"] = b64
        await self._send(writer, resp)
        log.info(f"下位机图片单张检测完成: {name} {len(result['detections'])} 目标, {t_total:.1f}ms")

    async def _handle_nano_batch(self, writer, msg):
        """批量检测会话启动（nano_batch_request）：Nano 端本地驱动 + 逐张推送进度"""
        if self.mm.compiling:
            await self._send(writer, {"type": "nano_batch_response", "ok": False,
                                      "error": "busy: 模型编译中，请稍候"})
            return
        if self._batch is not None:
            await self._send(writer, {"type": "nano_batch_response", "ok": False,
                                      "error": "busy: 已有批量检测进行中"})
            return
        names = msg.get("names") or []
        if msg.get("all"):
            names = [i["name"] for i in
                     self._store.list_images(limit=self._store.max_images)["images"]]
        # 安全校验：只保留合法文件名（非法名在执行阶段会逐张报错，这里直接过滤）
        names = [n for n in names if _safe_basename(n) is not None]
        if not names:
            await self._send(writer, {"type": "nano_batch_response", "ok": False,
                                      "error": "未选择有效图片（目录为空或文件名非法）"})
            return
        rounds = max(1, min(int(msg.get("rounds", 1) or 1), BATCH_MAX_ROUNDS))
        sid = uuid.uuid4().hex[:12]
        s = {
            "session_id": sid, "names": list(names), "rounds": rounds,
            "annotate": bool(msg.get("annotate", True)),
            "thumb_size": int(msg.get("thumb_size", 640) or 640),
            "writer": writer, "stop": False, "task": None,
            "done": 0, "total": len(names) * rounds,
            "ok_count": 0, "ng_count": 0, "failed": [], "times": [],
            "start_ts": time.time(),
        }
        self._batch = s
        s["task"] = asyncio.ensure_future(self._batch_worker(s))
        await self._send(writer, {"type": "nano_batch_response", "ok": True,
                                  "session_id": sid, "total": s["total"],
                                  "rounds": rounds, "count": len(names)})
        log.info(f"批量检测开始: {len(names)} 张 × {rounds} 轮, session={sid}")

    async def _handle_nano_batch_stop(self, writer, msg):
        """中止批量检测（nano_batch_stop_request）：当前张完成后停止"""
        s = self._batch
        if s is None:
            await self._send(writer, {"type": "nano_batch_stop_response", "ok": False,
                                      "stopped": False, "error": "无进行中的批量任务"})
            return
        s["stop"] = True
        await self._send(writer, {"type": "nano_batch_stop_response", "ok": True,
                                  "session_id": s["session_id"], "stopped": True})
        log.info(f"批量检测停止请求: session={s['session_id']}")

    async def _handle_nano_batch_status(self, writer, msg):
        """批量检测状态查询（nano_batch_status_request）"""
        s = self._batch
        if s is None:
            await self._send(writer, {"type": "nano_batch_status_response",
                                      "ok": True, "running": False})
            return
        await self._send(writer, {"type": "nano_batch_status_response", "ok": True,
                                  "running": True, "session_id": s["session_id"],
                                  "done": s["done"], "total": s["total"]})

    # ---------- 批量检测执行 ----------
    async def _batch_worker(self, s: dict):
        """批量会话主协程：轮次 × 图片 逐张检测，推送 progress/item，结束推送 done。
        单张失败不中断整批；stop/断连/总时长护栏均可终止。"""
        loop = asyncio.get_event_loop()
        try:
            for r in range(1, s["rounds"] + 1):
                for idx, name in enumerate(s["names"]):
                    if s["stop"]:
                        await self._push_batch_done(s, stopped=True)
                        return
                    # 总时长护栏：超时强制中止（防异常挂死占住引擎）
                    if time.time() - s["start_ts"] > BATCH_MAX_MINUTES * 60:
                        log.warning(f"批量检测超时(>{BATCH_MAX_MINUTES}min)，强制中止")
                        await self._push_batch_done(s, stopped=True,
                                                    error=f"超时(>{BATCH_MAX_MINUTES}min)强制中止")
                        return
                    item = await self._batch_detect_one(s, r, idx, name, loop)
                    s["done"] += 1
                    if item.get("ok"):
                        if item.get("det_count", 0) > 0:
                            s["ng_count"] += 1
                        else:
                            s["ok_count"] += 1
                        if item.get("timing_ms"):
                            s["times"].append(item["timing_ms"])
                    else:
                        s["failed"].append(name)
                    await self._push_batch_item(s, item)
                    await self._push_batch_progress(s, current=name, round_no=r)
            await self._push_batch_done(s, stopped=s["stop"])
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.error(f"批量任务异常: {e}", exc_info=True)
            try:
                await self._push_batch_done(s, stopped=True, error=f"批量任务异常: {e}")
            except Exception:
                pass
        finally:
            if self._batch is s:
                self._batch = None

    async def _batch_detect_one(self, s: dict, round_no: int, idx: int,
                                name: str, loop) -> dict:
        """单张检测（批量内）：读图 → 引擎检测（与实时 detect 共用锁 + 单 worker）"""
        img, err = self._store.read(name)
        if err is not None:
            return {"ok": False, "name": name, "round": round_no, "index": idx,
                    "error": err}
        t0 = time.perf_counter()
        async with self._engine_lock:
            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(self._executor, self.mm.engine.detect, img),
                    timeout=DETECT_TIMEOUT_S)
            except asyncio.TimeoutError:
                log.error(f"批量推理超时({DETECT_TIMEOUT_S}s): {name}")
                return {"ok": False, "name": name, "round": round_no, "index": idx,
                        "error": "detect timeout"}
            except Exception as e:
                log.error(f"批量推理异常: {name} -> {e}")
                return {"ok": False, "name": name, "round": round_no, "index": idx,
                        "error": f"detect error: {e}"}
        total_ms = (time.perf_counter() - t0) * 1000
        dets = result["detections"]
        item = {
            "ok": True, "name": name, "round": round_no, "index": idx,
            "detections": dets, "det_count": len(dets),
            "timing_ms": round(total_ms, 1),
            "timing": {**result.get("timing", {}), "total_with_read": round(total_ms, 1)},
            "image_size": {"w": img.shape[1], "h": img.shape[0]},
            "model": os.path.basename(self.mm.current),
        }
        if s["annotate"]:
            # 2026-09-01 修复：标注图生成移出事件循环 + 10s 超时
            try:
                b64, aerr = await asyncio.wait_for(
                    loop.run_in_executor(self._status_exec, self._store.annotate_b64,
                                         img, dets, self.mm.engine.labels, s["thumb_size"]),
                    timeout=10.0)
            except asyncio.TimeoutError:
                b64, aerr = None, "标注图生成超时"
            if aerr is None:
                item["annot_b64"] = b64
        return item

    async def _push_batch_item(self, s: dict, item: dict):
        await self._send(s["writer"], {"type": "nano_batch_item",
                                       "session_id": s["session_id"], **item})

    async def _push_batch_progress(self, s: dict, current: str = "", round_no: int = 1):
        avg = (sum(s["times"]) / len(s["times"])) if s["times"] else 0
        await self._send(s["writer"], {
            "type": "nano_batch_progress", "session_id": s["session_id"],
            "done": s["done"], "total": s["total"], "current_name": current,
            "round": round_no, "ok_count": s["ok_count"], "ng_count": s["ng_count"],
            "avg_ms": round(avg, 1)})

    async def _push_batch_done(self, s: dict, stopped: bool = False, error: str = None):
        avg = (sum(s["times"]) / len(s["times"])) if s["times"] else 0
        await self._send(s["writer"], {
            "type": "nano_batch_done", "session_id": s["session_id"],
            "total": s["total"], "done": s["done"], "rounds": s["rounds"],
            "ok_count": s["ok_count"], "ng_count": s["ng_count"],
            "failed": s["failed"], "avg_ms": round(avg, 1),
            "stopped": stopped, "error": error})
        log.info(f"批量检测结束: session={s['session_id']} done={s['done']}/{s['total']} "
                 f"ok={s['ok_count']} ng={s['ng_count']} failed={len(s['failed'])}")

    # ---------- 产线模拟相机（2026-08-24 新增） ----------
    async def _handle_stream_subscribe(self, writer, msg):
        """订阅/退订产线检测流（stream_subscribe_request）"""
        if msg.get("subscribe", True):
            self._camera_sim.subscribe(writer)
        else:
            self._camera_sim.unsubscribe(writer)
        st = self._camera_sim.state()
        await self._send(writer, {"type": "stream_subscribe_response", "ok": True,
                                  "subscribed": bool(msg.get("subscribe", True)), **st})

    async def _handle_stream_control(self, writer, msg):
        """控制产线相机（stream_control_request：start/stop/set_fps/set_source）"""
        action = msg.get("action", "")
        sim = self._camera_sim
        if action == "start":
            # 可选帧源：start 附带 source 时先切换（sim/camera），运行中切源先停
            src = msg.get("source", "")
            if src in ("sim", "camera") and src != sim.source:
                if sim.running:
                    await sim.stop()
                sim.set_source(src)
                log.info(f"stream_control start 附带帧源切换 -> {src}")
            ok, err = await sim.start()
            await self._send(writer, {"type": "stream_control_response", "ok": ok,
                                      "action": action, "error": err,
                                      **sim.state()})
        elif action == "stop":
            ok, err = await sim.stop()
            await self._send(writer, {"type": "stream_control_response", "ok": ok,
                                      "action": action, "error": err,
                                      **sim.state()})
        elif action == "set_fps":
            try:
                fps = int(msg.get("fps", 10))
            except (TypeError, ValueError):
                fps = 10
            sim.set_fps(fps)
            await self._send(writer, {"type": "stream_control_response", "ok": True,
                                      "action": action, "fps": sim.fps, **sim.state()})
        elif action == "set_source":
            src = msg.get("source", "")
            ok, err = sim.set_source(src)
            await self._send(writer, {"type": "stream_control_response", "ok": ok,
                                      "action": action, "error": err, **sim.state()})
        elif action == "set_camera":
            # 相机参数下发（曝光 us / 增益 dB；仅 camera 帧源生效）
            exp = msg.get("exposure_us")
            gain = msg.get("gain")
            try:
                exp = float(exp) if exp is not None else None
            except (TypeError, ValueError):
                exp = None
            try:
                gain = float(gain) if gain is not None else None
            except (TypeError, ValueError):
                gain = None
            # 2026-09-01 修复：相机参数下发=同步 MVS SDK 调用，移出事件循环 + 5s 超时
            try:
                ok, err, applied = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        self._status_exec, sim.set_camera_params, exp, gain),
                    timeout=5.0)
            except asyncio.TimeoutError:
                ok, err, applied = False, "相机参数下发超时(5s)", None
            await self._send(writer, {"type": "stream_control_response", "ok": ok,
                                      "action": action, "error": err,
                                      "camera": applied or {"exposure_us": exp, "gain": gain},
                                      **sim.state()})
        elif action == "auto_exposure":
            # 一键自动曝光（仅真实相机帧源；sim 帧源返回明确错误）
            try:
                target = float(msg.get("target_brightness", 120.0))
            except (TypeError, ValueError):
                target = 120.0
            ok, err, exp, gain, analysis = await sim.auto_exposure(target_brightness=target)
            resp = {"type": "stream_control_response", "ok": ok,
                    "action": action, "error": err,
                    "camera": ({"exposure_us": exp, "gain": gain} if exp is not None else None),
                    "analysis": analysis, **sim.state()}
            await self._send(writer, resp)
            log.info(f"一键自动曝光完成: ok={ok} exp={exp} gain={gain} err={err}")
        else:
            await self._send(writer, {"type": "stream_control_response", "ok": False,
                                      "action": action,
                                      "error": f"未知动作: {action}", **sim.state()})

    @staticmethod
    def _client_write_ok(writer) -> bool:
        """客户端写缓冲 < 2MB 视为健康（可正常接收）；超过判定客户端未及时读取（可能卡死），
        跳过本次推送（2026-09-01 稳定性加固）。"""
        try:
            tr = writer.transport
            if tr is not None and tr.get_write_buffer_size() > 2_000_000:
                return False
        except Exception:
            pass
        return True

    async def _send(self, writer, obj: dict):
        writer.write(FrameProtocol.pack(obj))
        # 2026-09-01：drain 套 5s 超时——客户端缓冲满（卡死/未读取）时不再无限阻塞，
        # 超时后按连接异常处理（断开）；客户端自动重连并重新订阅产线流。
        await asyncio.wait_for(writer.drain(), timeout=5.0)


def main():
    parser = argparse.ArgumentParser(description="缺陷检测 TCP 推理服务")
    parser.add_argument("--engine", default="/home/nvidia/defect_detection/models/baseline/yolov8s_fp16.engine")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--image-dir", default="",
                        help=f"下位机图片检测文件夹（默认 {DEFAULT_IMAGE_DIR}）")
    parser.add_argument("--source", choices=["sim", "camera"], default="sim",
                        help="产线流帧源：sim=图片模拟（默认）/ camera=真实相机（MVS USB）")
    args = parser.parse_args()

    mm = ModelManager(args.engine, conf_thres=args.conf, iou_thres=args.iou)
    server = DetectServer(mm, args.host, args.port, image_dir=args.image_dir,
                          source=args.source)

    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        log.info("服务停止")
    finally:
        server.close()


if __name__ == "__main__":
    main()