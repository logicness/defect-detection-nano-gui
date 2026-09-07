#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下位机产线模拟相机（2026-08-24 新增，方案 v2）
=================================================
- 用下位机 images/input 的图片模拟相机：按 FPS 节拍循环取帧（顺序/随机）
- 用当前激活模型独立检测（与 TCP 服务共用锁/worker，天然串行）
- 检测结果 stream_frame 主动推送给所有订阅连接（产线自主检测，上位机只接收查看）
- CLI 独立运行：板端直跑（标注图+CSV），--push 可推送到指定上位机

协议（走 4 字节长度头 + JSON，由 infer_server 分发）：
  PC→Nano stream_subscribe_request {type, subscribe: bool}
  PC→Nano stream_control_request   {type, action: start|stop|set_fps, fps?}
  Nano→PC stream_frame             {type, seq, ts, name, image_size, detections,
                                    annot_b64, timing, model}
  Nano→PC stream_state             {type, running, fps, total_frames, ok_count,
                                    ng_count, avg_ms}
"""
import asyncio
import base64
import csv
import json
import os
import random
import socket
import struct
import sys
import time
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor

# 真实相机帧源（相机接入 P2，2026-08-28）：SDK/MVS 不可用时安全降级为图片模拟
try:
    from camera_source import NanoCameraSource
    _HAS_CAMERA_MODULE = True
except Exception:
    _HAS_CAMERA_MODULE = False

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "inference"))
from image_store import ImageStore, DEFAULT_IMAGE_DIR
from model_manager import ModelManager, _rel_name
from log_config import log_metric, log_performance, log_health, PerformanceTimer

log = logging.getLogger("camera_sim")

DETECT_TIMEOUT_S = 10.0
MAX_TIMES_WINDOW = 30          # avg_ms 滚动窗口
HEARTBEAT_INTERVAL = 100       # 每 100 帧推送一次健康心跳
STALL_THRESHOLD_MS = 5000      # 帧间隔超过 5 秒视为卡顿

# ---- 2026-09-01 稳定性加固（T0/T1）----
BROADCAST_DRAIN_TIMEOUT_S = 1.0   # 单订阅者 drain 超时：超时只跳帧，不阻塞产线主循环
SLOW_BUF_BYTES = 1_500_000        # 写缓冲阈值：超过即视为慢订阅者，跳过本帧（≈5帧×300KB）
WATCHDOG_SILENCE_S = 15.0         # 产线看门狗：连续 N 秒无新帧 → 自动重启产线流
CAM_REOPEN_BASE_S = 1.0           # 相机重连冷却基数（指数退避）
CAM_REOPEN_MAX_S = 16.0           # 相机重连冷却上限


class CameraSimulator:
    """产线模拟相机：按 FPS 节拍取图 → 检测 → 广播 stream_frame。

    与 detect/batch/load 共用 engine_lock + 单 worker executor，天然串行。
    独立于 infer_server 运行（CLI 模式），也可内嵌服务使用。
    """

    def __init__(self, store: ImageStore, mm: ModelManager,
                 executor: ThreadPoolExecutor, engine_lock: asyncio.Lock,
                 loop: asyncio.AbstractEventLoop, fps: int = 10,
                 shuffle: bool = False, gpu_sampler=None, source: str = "sim"):
        self.store = store
        self.mm = mm
        self.executor = executor
        self.engine_lock = engine_lock
        self.loop = loop
        self.fps = max(1, min(60, int(fps)))
        self.shuffle = bool(shuffle)
        # 帧源：sim=图片模拟（默认，无相机回退）/ camera=真实相机（MVS SDK）
        self.source = source if source in ("sim", "camera") else "sim"
        self.running = False
        self._task = None
        self._seq = 0            # 全局帧序号（跨 start/stop 累计）
        self._ok = 0
        self._ng = 0
        self._times = []
        self._names = []         # 帧来源快照
        self._idx = 0
        self._subscribers = set()  # asyncio.StreamWriter 集合（广播目标）
        self._gpu_sampler = gpu_sampler  # GPU 采样器引用（可选）
        # 真实相机：懒加载 + 独立读帧线程池（不占用推理 executor）
        self._camera = None              # NanoCameraSource 实例（camera 模式）
        self._cam_executor = None        # 相机读帧专用线程池
        self._cam_fail_seq = 0           # 连续取帧失败计数（触发重连阈值）
        self._cam_reopen_at = 0.0        # 下次允许重连的时间戳（指数退避冷却）
        self._cam_reopen_delay = CAM_REOPEN_BASE_S  # 当前重连冷却（指数退避）
        self._build_executor = None      # 标注帧构建专用线程池（独立于相机读帧）
        self._slow_skips = 0             # 慢订阅者跳帧计数（健康统计）
        self._watchdog_restarts = 0      # 看门狗自动重启次数
        self._recovering = False         # 检测引擎自愈进行中（防并发重复自愈）
        self._detect_recoveries = 0      # 检测引擎自愈次数（健康统计）
        self._last_output_ts = 0.0       # 最近一次产线输出（含启动时刻，看门狗判据）
        self._suspend_read = False       # 一键自动曝光期间暂停取帧（优化器独占相机读，防并发）
        self._last_quality_ts = 0.0      # 图像质量评分节流时间戳（~1Hz）
        self._last_quality = None        # 最近一次质量评分（随 stream_frame 推送）
        self._watchdog_task = None       # 外部看门狗 task（循环冻结时兜底 cancel+重启）
        # 健康检查相关
        self._error_count = 0          # 累计错误数
        self._last_frame_ts = 0.0      # 最后一帧时间戳
        self._stall_count = 0          # 卡顿次数
        self._health_stats = {         # 健康统计
            "timeouts": 0,
            "read_errors": 0,
            "detect_errors": 0,
            "stalls": 0,
        }

    # ---------- 控制 ----------
    async def start(self) -> tuple:
        """启动产线任务。返回 (ok, error)。sim 模式图片目录为空返回失败；
        camera 模式相机不可用也返回失败。"""
        if self.running:
            return False, "产线模拟已在运行"
        if self.source == "camera":
            # 真实相机：校验 SDK/设备（懒加载打开），失败给出明确原因
            if not _HAS_CAMERA_MODULE:
                return False, "camera_source 模块不可用（缺少 MVS Python 绑定）"
            if self._cam_executor is None:
                self._cam_executor = ThreadPoolExecutor(
                    max_workers=2, thread_name_prefix="cam-read")
            # 2026-09-01 修复：相机打开=同步 MVS SDK 调用（CreateHandle/OpenDevice/
            # StartGrabbing），移入相机线程池 + 5s 超时，避免卡死事件循环。
            loop = self.loop or asyncio.get_running_loop()
            try:
                cam = await asyncio.wait_for(
                    loop.run_in_executor(self._cam_executor, self._ensure_camera),
                    timeout=5.0)
            except asyncio.TimeoutError:
                log.error("相机打开超时(5s)，产线启动失败")
                return False, "相机打开超时(5s)（检查 USB 连接 / MVS SDK）"
            if cam is None:
                return False, "相机打开失败（检查 USB 连接 / MVS SDK）"
            self.running = True
            self._idx = 0
            self._task = asyncio.ensure_future(self._produce())
            self._watchdog_task = asyncio.ensure_future(self._external_watchdog())
            log.info(f"产线真实相机启动: {self.fps} FPS")
            return True, None
        r = self.store.list_images(limit=self.store.max_images)
        self._names = [i["name"] for i in r["images"]]
        if not self._names:
            return False, "图片目录为空，无法启动产线模拟"
        if self.shuffle:
            random.shuffle(self._names)
        self.running = True
        self._idx = 0
        self._task = asyncio.ensure_future(self._produce())
        self._watchdog_task = asyncio.ensure_future(self._external_watchdog())
        log.info(f"产线模拟相机启动: {len(self._names)} 张 × {self.fps} FPS"
                 f"{'（随机序）' if self.shuffle else ''}")
        return True, None

    async def stop(self) -> tuple:
        """停止产线任务（当前帧检测完成后停）。"""
        if not self.running:
            return False, "产线模拟未在运行"
        self.running = False
        task = self._task
        self._task = None
        if task is not None:
            try:
                task.cancel()
            except Exception:
                pass
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        wt = self._watchdog_task
        self._watchdog_task = None
        if wt is not None:
            try:
                wt.cancel()
            except Exception:
                pass
            try:
                await wt
            except (asyncio.CancelledError, Exception):
                pass
        log.info(f"产线{'真实相机' if self.source == 'camera' else '模拟相机'}停止: "
                 f"累计 {self._seq} 帧, OK={self._ok}, NG={self._ng}")
        return True, None

    def set_source(self, source: str) -> tuple:
        """运行时切换帧源（sim/camera）。运行中拒绝切换，需先 stop。"""
        if source not in ("sim", "camera"):
            return False, f"未知帧源: {source}"
        if self.running:
            return False, "产线运行中，请先停止再切换帧源"
        self.source = source
        log.info(f"产线帧源切换: {source}")
        return True, None

    def close(self):
        """释放相机资源（进程退出 / 服务关闭时调用；幂等）。"""
        if self._camera is not None:
            try:
                self._camera.close()
            except Exception:
                pass
            self._camera = None
        if self._cam_executor is not None:
            try:
                self._cam_executor.shutdown(wait=False)
            except Exception:
                pass
            self._cam_executor = None
        if self._build_executor is not None:
            try:
                self._build_executor.shutdown(wait=False)
            except Exception:
                pass
            self._build_executor = None

    def set_fps(self, fps: int):
        self.fps = max(1, min(60, int(fps)))

    def set_camera_params(self, exposure_us=None, gain=None):
        """运行时下发相机参数（仅 camera 帧源；sim 模式忽略）。
        返回 (ok, error, applied)。
        2026-09-01 完善：相机未打开时尝试自动打开；applied=读回的实际生效值。"""
        if self.source != "camera":
            return False, "当前为图片模拟帧源，相机参数不适用", None
        cam = self._ensure_camera()
        if cam is None:
            return False, "相机未打开（检查 USB 连接；可先启动产线流再下发参数）", None
        try:
            return cam.set_params(exposure_us=exposure_us, gain=gain)
        except Exception as e:
            log.warning(f"相机参数下发失败: {e}")
            return False, str(e), None

    def camera_temperature(self):
        """当前相机温度（°C）；无相机/失败返回 None。2026-09-01 状态栏展示用。"""
        try:
            if self.source == "camera" and self._camera is not None:
                return self._camera.get_temperature()
        except Exception as e:
            log.debug(f"camera_temperature 失败: {e}")
        return None

    def camera_online(self) -> bool:
        """camera 帧源且相机已打开（status_response 展示用；sim/无相机=False）。"""
        try:
            return (self.source == "camera" and self._camera is not None
                    and bool(getattr(self._camera, "_opened", False)))
        except Exception:
            return False

    async def auto_exposure(self, target_brightness: float = 120.0):
        """一键自动曝光（仅真实相机帧源）：算法迭代搜索曝光/增益到目标亮度。
        运行期间暂停产线取帧（优化器独占相机读，防并发读同一相机句柄）。
        返回 (ok, error, exposure_us, gain, analysis)。"""
        if self.source != "camera":
            return False, "仅真实相机帧源支持自动曝光（当前为图片模拟）", None, None, None
        # 2026-09-01 修复：相机打开移出事件循环（同 start()）
        loop = self.loop or asyncio.get_running_loop()
        try:
            cam = await asyncio.wait_for(
                loop.run_in_executor(self._cam_executor or self.executor,
                                     self._ensure_camera),
                timeout=5.0)
        except asyncio.TimeoutError:
            return False, "相机打开超时(5s)", None, None, None
        if cam is None:
            return False, "相机未打开（检查 USB 连接）", None, None, None
        if not hasattr(cam, "auto_optimize_exposure"):
            return False, "相机不支持自动曝光（缺少 auto_optimize_exposure 实现）", None, None, None
        self._suspend_read = True
        try:
            loop = self.loop or asyncio.get_running_loop()
            ex, gn, analysis = await asyncio.wait_for(
                loop.run_in_executor(
                    self._cam_executor or self.executor,
                    cam.auto_optimize_exposure,
                    float(target_brightness), 20.0, 5),
                timeout=20.0)
            return True, None, ex, gn, analysis
        except asyncio.TimeoutError:
            log.error("自动曝光优化超时(20s)")
            return False, "自动曝光优化超时", None, None, None
        except Exception as e:
            log.warning(f"自动曝光优化失败: {e}")
            return False, str(e), None, None, None
        finally:
            self._suspend_read = False

    # ---------- 真实相机（camera 模式） ----------
    def _ensure_camera(self):
        """返回已打开的 NanoCameraSource；失败返回 None（含指数退避冷却）。

        2026-09-01 优化：冷却从固定 1s 改为指数退避（1→2→4→8→16s 封顶），
        避免 USB 抖动时高频重连风暴；打开成功后复位退避。
        """
        now = time.time()
        if now < self._cam_reopen_at:
            return None  # 冷却期内不重试（指数退避）
        if self._camera is None:
            self._camera = NanoCameraSource()
        if not getattr(self._camera, "_opened", False):
            try:
                ok = self._camera.open()
            except Exception as e:
                log.warning(f"相机打开异常: {e}")
                ok = False
            if not ok:
                self._cam_fail_seq += 1
                self._cam_reopen_delay = min(
                    self._cam_reopen_delay * 2, CAM_REOPEN_MAX_S)
                self._cam_reopen_at = time.time() + self._cam_reopen_delay
                self._health_stats["read_errors"] += 1
                self._error_count += 1
                log.warning(f"相机打开失败(退避{self._cam_reopen_delay:.0f}s): "
                            f"{getattr(self._camera, '_last_error', '')}")
                return None
            self._cam_fail_seq = 0
            self._cam_reopen_delay = CAM_REOPEN_BASE_S
            log.info("相机已打开（camera 帧源）")
        return self._camera

    def _ensure_and_read(self):
        """相机确保打开 + 读一帧（同步，供 executor 调用）。
        2026-09-01 修复：原 _read_camera_frame 先同步调 _ensure_camera()
        （MVS 打开调用无超时）再 executor 读帧——相机打开挂死同样会冻结事件循环；
        现在 ensure 与 read 一起放进线程池执行。"""
        cam = self._ensure_camera()
        if cam is None:
            return False, None
        return cam.read()

    async def _read_camera_frame(self):
        """真实相机读帧：独立线程池执行（不阻塞推理 executor / 事件循环）。
        2026-09-01 v3：run_in_executor 套 2.5s 超时——MVS 内部偶发挂死时不再卡死产线循环；
        v4(修复)：相机 open 与 read 一并在线程池执行。"""
        if self.source != "camera":
            return None, "非相机帧源"
        loop = self.loop or asyncio.get_running_loop()
        try:
            ok, frame = await asyncio.wait_for(
                loop.run_in_executor(self._cam_executor, self._ensure_and_read),
                timeout=2.5)
        except asyncio.TimeoutError:
            log.warning("相机读帧超时(2.5s)，按取帧失败处理")
            ok, frame = False, None
        except Exception as e:
            log.warning(f"相机读帧异常: {e}")
            ok, frame = False, None
        cam = self._camera
        if not ok or frame is None:
            self._cam_fail_seq += 1
            self._cam_reopen_delay = min(
                self._cam_reopen_delay * 2, CAM_REOPEN_MAX_S)
            self._cam_reopen_at = time.time() + self._cam_reopen_delay
            err = getattr(cam, "_last_error", "") or "取帧失败"
            log.warning(f"相机取帧失败({self._cam_fail_seq}, "
                        f"退避{self._cam_reopen_delay:.0f}s): {err}")
            if self._cam_fail_seq >= 5:
                # 连续失败 → 关掉句柄，下次 _ensure_camera 重开（USB 掉线自愈）
                try:
                    cam.close()
                except Exception:
                    pass
            return None, err
        self._cam_fail_seq = 0
        self._cam_reopen_delay = CAM_REOPEN_BASE_S
        return frame, None

    async def _watchdog_restart(self):
        """看门狗重启产线流：先引擎自愈 → 强制重开相机 → 重新 start（最多 5 次尝试）。"""
        try:
            # 先做检测引擎自愈（TRT 挂死占死线程池是卡死主因，相机重开只是兜底）；
            # 若超时路径已在自愈中则跳过，避免并发双重载
            if not self._recovering:
                self._recovering = True
                await self._recover_detect_engine()
            else:
                # 等已有自愈完成（产线循环 _recovering 检查会暂停检测）
                for _ in range(60):
                    if not self._recovering:
                        break
                    await asyncio.sleep(0.5)
            for attempt in range(5):
                await asyncio.sleep(1.0)
                # 强制清除重连冷却并重建相机实例（旧句柄在 USB 复位后可能失效）
                self._cam_reopen_at = 0.0
                self._cam_reopen_delay = CAM_REOPEN_BASE_S
                if self.source == "camera":
                    self._cam_fail_seq = 0
                    if self._camera is not None:
                        try:
                            self._camera.close()
                        except Exception:
                            pass
                        self._camera = None
                ok, err = await self.start()
                if ok:
                    log.info(f"看门狗已重启产线流（尝试 {attempt + 1}）")
                    return
                log.warning(f"看门狗重启失败({attempt + 1}/5): {err}")
            log.error("看门狗重启产线流失败 5 次，停止尝试（可在 GUI 手动重新开始）")
        except Exception as e:
            log.error(f"看门狗重启异常: {e}", exc_info=True)

    async def _recover_detect_engine(self):
        """检测引擎自愈（2026-09-01 v2）：TRT 单次推理挂死后换线程池 + 重载引擎。
        - 旧线程池含挂死线程，弃用（Python 3.9+ 线程池线程为 daemon，进程退出不阻塞）
        - 新引擎 = 新 TRTInfer + 新 CUDA context，与挂死线程完全隔离
        - 通知服务端替换共享 executor（单张检测/批量路径同样受益）"""
        try:
            log.warning("检测引擎自愈开始：替换 executor + 重载引擎")
            new_exec = ThreadPoolExecutor(max_workers=1, thread_name_prefix="trt")
            old_exec = self.executor
            self.executor = new_exec
            cb = getattr(self, "_on_executor_replaced", None)
            if cb is not None:
                try:
                    cb(new_exec)
                except Exception as e:
                    log.warning(f"通知服务端替换 executor 失败: {e}")
            name = _rel_name(self.mm.current) or os.path.basename(self.mm.current)
            try:
                loop = self.loop or asyncio.get_running_loop()
                async with self.engine_lock:
                    r = await asyncio.wait_for(
                        loop.run_in_executor(new_exec, self.mm.load, name),
                        timeout=30.0)
                if r and r.get("ok"):
                    log.info(f"检测引擎自愈成功: {name} ({r.get('load_ms', 0)}ms)")
                else:
                    log.error(f"检测引擎自愈加载失败: {(r or {}).get('error', 'unknown')}")
            except Exception as e:
                log.error(f"检测引擎自愈异常: {e}", exc_info=True)
            self._detect_recoveries += 1
            try:
                old_exec.shutdown(wait=False)
            except Exception:
                pass
        finally:
            self._recovering = False

    # ---------- 订阅 ----------
    def subscribe(self, writer):
        self._subscribers.add(writer)

    def unsubscribe(self, writer):
        self._subscribers.discard(writer)

    def state(self) -> dict:
        avg = (sum(self._times) / len(self._times)) if self._times else 0.0
        return {
            "running": self.running, "fps": self.fps,
            "source": self.source,   # sim / camera
            "total_frames": self._seq, "ok_count": self._ok,
            "ng_count": self._ng, "avg_ms": round(avg, 1),
            "error_count": self._error_count,
            "stall_count": self._stall_count,
            "slow_skips": self._slow_skips,
            "watchdog_restarts": self._watchdog_restarts,
            "detect_recoveries": self._detect_recoveries,
            "health": self._health_stats.copy(),
        }

    # ---------- 外部看门狗（2026-09-01 v4） ----------
    async def _external_watchdog(self):
        """独立看门狗 task：周期性检查输出时间戳。

        循环内看门狗在产线循环冻结（某个 await 永久挂起，如 engine_lock 竞争、
        detect/读图/构建卡死）时同样失效——循环不迭代，检查不执行。
        本 task 不依赖循环迭代：超时后 cancel 冻结的产线任务（asyncio 可打断
        挂起的 await），自动重建产线任务，预览不永久卡死。
        """
        try:
            while self.running:
                await asyncio.sleep(WATCHDOG_SILENCE_S)
                if not self.running:
                    break
                now = time.perf_counter()
                if now - self._last_output_ts <= WATCHDOG_SILENCE_S + 2.0:
                    continue
                gap = now - self._last_output_ts
                log.error(f"外部看门狗: {gap:.0f}s 无新帧，取消冻结的产线任务并自动重启")
                self._watchdog_restarts += 1
                self._error_count += 1
                self._health_stats.setdefault("watchdog_restarts", 0)
                self._health_stats["watchdog_restarts"] += 1
                task = self._task
                self._task = None
                if task is not None:
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass
                self._last_output_ts = time.perf_counter()
                if self.running:
                    self._task = asyncio.ensure_future(self._produce())
                    log.info("外部看门狗: 产线任务已重建")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error(f"外部看门狗异常: {e}")

    # ---------- 产线主循环 ----------
    def _next_name(self) -> str:
        if not self._names:
            return ""
        name = self._names[self._idx % len(self._names)]
        self._idx += 1
        return name

    # ---------- 帧源取帧（sim / camera 统一入口） ----------
    async def _get_frame(self):
        """按当前帧源取一帧，返回 (name, img, err)。
        sim 模式：图片循环（store.read）；camera 模式：真实相机（MVS SDK）。"""
        if self.source == "camera":
            frame, err = await self._read_camera_frame()
            return f"camera_{self._seq:06d}", frame, err
        name = self._next_name()
        if not name:
            return "", None, "图片目录为空"
        # 2026-09-01 v4：同步读图移 executor + 3s 超时——图片损坏/IO 卡死不再冻结事件循环
        try:
            loop = self.loop or asyncio.get_running_loop()
            img, err = await asyncio.wait_for(
                loop.run_in_executor(self._build_executor or self.executor,
                                     self.store.read, name),
                timeout=3.0)
        except asyncio.TimeoutError:
            return "", None, f"读图超时(3s): {name}"
        except Exception as e:
            return "", None, f"读图异常: {e}"
        return name, img, err

    async def _produce(self):
        """按 1/fps 节拍：取帧→检测→广播。检测耗时超节拍时跳过 sleep（保节奏不堆积）。"""
        interval = 1.0 / self.fps
        self._last_output_ts = time.perf_counter()  # 看门狗判据：含启动时刻
        log_health("camera_sim", "started",
                   {"fps": self.fps, "source": self.source,
                    "images": len(self._names) if self.source == "sim" else "camera"})
        
        try:
            while self.running:
                # 一键自动曝光期间暂停取帧（优化器独占相机读，防并发读同一相机句柄）
                if self._suspend_read:
                    await asyncio.sleep(0.05)
                    continue
                t_frame = time.perf_counter()
                
                # 卡顿检测：帧间隔异常
                if self._last_frame_ts > 0:
                    gap_ms = (t_frame - self._last_frame_ts) * 1000
                    if gap_ms > STALL_THRESHOLD_MS:
                        self._stall_count += 1
                        self._health_stats["stalls"] += 1
                        log.warning(f"产线模拟卡顿检测: 帧间隔 {gap_ms:.0f}ms > {STALL_THRESHOLD_MS}ms")
                        log_metric("stall_detected", gap_ms, {"threshold": STALL_THRESHOLD_MS})

                # 看门狗（2026-09-01 v2）：连续 WATCHDOG_SILENCE_S 秒无新帧 → 自动重启产线流。
                # 用 _last_output_ts（含产线启动时刻）：首帧即失败也能触发，不再依赖首帧成功。
                if self.running and (t_frame - self._last_output_ts) > WATCHDOG_SILENCE_S:
                    gap_s = t_frame - self._last_output_ts
                    log.error(f"产线看门狗: {gap_s:.0f}s 无新帧，"
                              f"自动重启产线流（第 {self._watchdog_restarts + 1} 次）")
                    self._watchdog_restarts += 1
                    self._error_count += 1
                    self._health_stats.setdefault("watchdog_restarts", 0)
                    self._health_stats["watchdog_restarts"] += 1
                    self._last_frame_ts = 0.0  # 防止 finally 后立即再触发
                    self._last_output_ts = time.perf_counter()
                    asyncio.ensure_future(self._watchdog_restart())
                    break

                if self.mm.compiling:
                    log.warning("模型编译中，产线模拟跳帧")
                    self._last_frame_ts = time.perf_counter()  # 编译期不计入卡死
                    await asyncio.sleep(0.5)
                    continue
                
                # 取帧（sim=图片循环 / camera=真实相机）
                t_read_start = time.perf_counter()
                name, img, err = await self._get_frame()
                t_read = (time.perf_counter() - t_read_start) * 1000
                if err is not None or img is None:
                    log.warning(f"产线取帧失败(跳过): {err}")
                    self._health_stats["read_errors"] += 1
                    self._error_count += 1
                    await asyncio.sleep(interval)
                    continue
                
                t0 = time.perf_counter()
                loop = self.loop or asyncio.get_running_loop()
                # 检测前缩小图像到模型输入尺寸，减少 TRT 内部缩放开销
                import cv2 as _cv22
                _h, _w = img.shape[:2]
                _det_side = 640
                _ds = min(1.0, _det_side / max(_h, _w))
                det_img = _cv22.resize(img, (int(_w * _ds), int(_h * _ds)),
                                       interpolation=_cv22.INTER_AREA) if _ds < 1.0 else img
                t_resize = (time.perf_counter() - t0) * 1000
                # 传递缩放后的图像给 _build_frame，避免二次缩放
                small_img = det_img

                # 引擎自愈中：跳过检测帧（避免排队到旧挂死线程池上）
                if self._recovering:
                    await asyncio.sleep(0.2)
                    continue

                t_detect_start = time.perf_counter()
                # 2026-09-01 v4：engine_lock 获取加 5s 超时——锁被其他协程挂死持有时
                # 不再永久等待（永久等待=循环冻结=内部看门狗失效=预览卡死需人工重启）
                try:
                    await asyncio.wait_for(self.engine_lock.acquire(), timeout=5.0)
                except asyncio.TimeoutError:
                    log.error("engine_lock 获取超时(5s)，跳过本帧")
                    self._health_stats.setdefault("lock_timeouts", 0)
                    self._health_stats["lock_timeouts"] += 1
                    await asyncio.sleep(0.2)
                    continue
                try:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(self.executor, self.mm.engine.detect, det_img),
                        timeout=DETECT_TIMEOUT_S)
                except asyncio.TimeoutError:
                    log.error(f"产线模拟推理超时: {name} → 触发检测引擎自愈")
                    self._health_stats["timeouts"] += 1
                    self._error_count += 1
                    log_metric("detect_timeout", 1, {"name": name})
                    # 2026-09-01 v2：TRT 单次推理可能偶发挂死并永久占住单 worker 线程池，
                    # 必须立即自愈（换 executor + 重载引擎），否则产线永不产帧（预览卡死）。
                    if not self._recovering:
                        self._recovering = True
                        asyncio.ensure_future(self._recover_detect_engine())
                    await asyncio.sleep(interval)
                    continue
                except Exception as e:
                    log.error(f"产线模拟推理异常: {name} -> {e}")
                    self._health_stats["detect_errors"] += 1
                    self._error_count += 1
                    log_metric("detect_error", 1, {"name": name, "error": str(e)})
                    await asyncio.sleep(interval)
                    continue
                finally:
                    self.engine_lock.release()
                t_detect = (time.perf_counter() - t_detect_start) * 1000
                
                total_ms = (time.perf_counter() - t0) * 1000
                self._seq += 1
                self._last_frame_ts = time.perf_counter()
                self._last_output_ts = self._last_frame_ts
                dets = result["detections"]
                
                if dets:
                    self._ng += 1
                else:
                    self._ok += 1
                
                self._times.append(total_ms)
                if len(self._times) > MAX_TIMES_WINDOW:
                    self._times.pop(0)
                
                # 记录检测性能指标
                log_metric("frame_detected", total_ms, {
                    "seq": self._seq,
                    "name": name,
                    "detections": len(dets),
                    "result": "NG" if dets else "OK"
                })
                
                t_build_start = time.perf_counter()
                if self._build_executor is None:
                    self._build_executor = ThreadPoolExecutor(
                        max_workers=1, thread_name_prefix="build")
                try:
                    frame = await asyncio.wait_for(
                        loop.run_in_executor(
                            self._build_executor, self._build_frame,
                            name, small_img, dets, result, total_ms),
                        timeout=5.0)
                except asyncio.TimeoutError:
                    # 2026-09-01 v3：标注帧构建挂死（OpenCV 编码偶发）不再卡死产线循环
                    log.error(f"标注帧构建超时(5s)，跳过该帧: {name}")
                    self._health_stats.setdefault("build_timeouts", 0)
                    self._health_stats["build_timeouts"] += 1
                    await asyncio.sleep(interval)
                    continue
                t_build = (time.perf_counter() - t_build_start) * 1000
                
                t_broadcast_start = time.perf_counter()
                await self._broadcast(frame)
                t_broadcast = (time.perf_counter() - t_broadcast_start) * 1000
                
                t_total_frame = (time.perf_counter() - t_frame) * 1000
                # 详细计时日志（每10帧打印一次）
                if self._seq % 10 == 0:
                    log.info(f"[TIMING] seq={self._seq} read={t_read:.0f}ms resize={t_resize:.0f}ms "
                             f"detect={t_detect:.0f}ms build={t_build:.0f}ms broadcast={t_broadcast:.0f}ms "
                             f"total={t_total_frame:.0f}ms")
                
                # 健康心跳：每 HEARTBEAT_INTERVAL 帧推送一次
                if self._seq % HEARTBEAT_INTERVAL == 0:
                    await self._broadcast_health()
                    log_health("camera_sim", "running", {
                        "seq": self._seq,
                        "ok": self._ok,
                        "ng": self._ng,
                        "avg_ms": round(sum(self._times) / len(self._times), 1) if self._times else 0,
                        "errors": self._error_count,
                        "stalls": self._stall_count
                    })
                
                # CLI 本地回调（标注图落盘/CSV/推送），TCP 服务内不设置
                cb = getattr(self, "_on_frame_cb", None)
                if cb is not None:
                    try:
                        await cb(frame)
                    except Exception as e:
                        log.error(f"产线模拟帧回调异常: {e}")
                
                # 节拍：检测+推送耗时超过 1/fps 则不 sleep（尽力保帧率）
                elapsed = time.perf_counter() - t_frame
                wait = interval - elapsed
                if wait > 0:
                    await asyncio.sleep(wait)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error(f"产线模拟任务异常: {e}", exc_info=True)
        finally:
            self.running = False
            log_health("camera_sim", "stopped", {
                "total_frames": self._seq,
                "ok_count": self._ok,
                "ng_count": self._ng,
                "error_count": self._error_count,
                "stall_count": self._stall_count
            })

    def _build_frame(self, name, img, dets, result, total_ms) -> dict:
        frame = {
            "type": "stream_frame", "seq": self._seq,
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "name": name,
            "image_size": {"w": img.shape[1], "h": img.shape[0]},
            "detections": dets, "det_count": len(dets),
            "timing": {**result.get("timing", {}),
                       "total_with_read": round(total_ms, 1)},
            "model": os.path.basename(self.mm.current),
        }
        # GPU 利用率实时推送（从采样器获取当前值）
        if self._gpu_sampler is not None:
            try:
                frame["gpu_util"] = self._gpu_sampler.get_current_load()
            except Exception:
                pass
        # 图像质量评分（2026-09-01：对焦/曝光辅助，节流 ~1Hz；在构建线程内执行不阻塞循环）
        try:
            now_q = time.perf_counter()
            if now_q - self._last_quality_ts >= 1.0:
                from auto_focus import ImageQualityEvaluator
                _m = ImageQualityEvaluator().evaluate(img)
                self._last_quality_ts = now_q
                self._last_quality = {
                    "sharpness": round(float(_m["sharpness"]), 1),
                    "brightness_mean": round(float(_m["brightness_mean"]), 1),
                    "overexposed_pct": round(float(_m["overexposed_pct"]), 2),
                    "underexposed_pct": round(float(_m["underexposed_pct"]), 2),
                    "contrast": round(float(_m["contrast"]), 1),
                }
            if self._last_quality is not None:
                frame["quality"] = self._last_quality
        except Exception as e:
            log.debug(f"图像质量评分失败: {e}")
        # 直接在已缩放的图像上画框编码（避免二次缩放）
        # 2026-09-03：预览分辨率 640→1280（2448×2048 相机画面缩到 640 细节损失大，
        # 1280 在 GUI 预览/ROI 区观感明显更清晰；带宽/CPU 仍可控）
        b64, aerr = self.store.annotate_b64(img, dets, getattr(self.mm.engine, "labels", None), max_side=1280)
        if aerr is None:
            frame["annot_b64"] = b64
        # 2026-09-02：原始无框预览帧（「打开相机画面」纯预览用，不含检测框）
        rb, rerr = self.store.annotate_b64(img, [], max_side=1280)
        if rerr is None:
            frame["raw_b64"] = rb
        return frame

    async def _broadcast(self, frame: dict):
        """广播一帧给所有订阅者。

        慢订阅者保护（2026-09-01 优化，根治"GUI 卡 → 相机卡死"）：
        - 发送前检查写缓冲：超过 SLOW_BUF_BYTES 视为慢订阅者，跳过本帧（latest-wins）
        - drain 套 BROADCAST_DRAIN_TIMEOUT_S 超时，超时只跳帧不阻塞产线主循环
        - 仅真正断开（写异常）才移除订阅者
        单个慢/死客户端不再拖死整条产线流。
        """
        dead = []
        for w in list(self._subscribers):
            try:
                tr = w.transport
                if tr is not None and tr.get_write_buffer_size() > SLOW_BUF_BYTES:
                    self._slow_skips += 1
                    continue  # 慢订阅者：跳过本帧，不写不阻塞
                await asyncio.wait_for(self._send(w, frame),
                                       timeout=BROADCAST_DRAIN_TIMEOUT_S)
            except asyncio.TimeoutError:
                self._slow_skips += 1
                continue  # 超时：跳过本帧（写缓冲已满，不追加数据避免半帧）
            except Exception:
                dead.append(w)
        for w in dead:
            self._subscribers.discard(w)

    async def _broadcast_health(self):
        """推送健康心跳（每 HEARTBEAT_INTERVAL 帧调用一次）"""
        health = {
            "type": "stream_health",
            "seq": self._seq,
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "running": self.running,
            "fps": self.fps,
            "total_frames": self._seq,
            "ok_count": self._ok,
            "ng_count": self._ng,
            "avg_ms": round(sum(self._times) / len(self._times), 1) if self._times else 0.0,
            "error_count": self._error_count,
            "stall_count": self._stall_count,
            "slow_skips": self._slow_skips,
            "watchdog_restarts": self._watchdog_restarts,
            "detect_recoveries": self._detect_recoveries,
            "health": self._health_stats.copy(),
        }
        # GPU 利用率
        if self._gpu_sampler is not None:
            try:
                health["gpu_util"] = self._gpu_sampler.get_current_load()
            except Exception:
                pass
        await self._broadcast(health)
        log.debug(f"产线模拟健康心跳: seq={self._seq}, errors={self._error_count}, stalls={self._stall_count}")

    @staticmethod
    async def _send(writer, obj: dict):
        payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        writer.write(struct.pack(">I", len(payload)) + payload)
        await writer.drain()


# ================= CLI 独立运行 =================

def _pack(obj: dict) -> bytes:
    payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    return struct.pack(">I", len(payload)) + payload


def cli_main():
    parser = argparse.ArgumentParser(description="下位机产线模拟相机（独立运行）")
    parser.add_argument("--dir", default="", help=f"图片文件夹（默认 {DEFAULT_IMAGE_DIR}）")
    parser.add_argument("--fps", type=int, default=10, help="检测节拍（帧/秒）")
    parser.add_argument("--rounds", type=int, default=20, help="检测帧数")
    parser.add_argument("--shuffle", action="store_true", help="随机取帧")
    parser.add_argument("--output", default="outputs/camera_sim", help="标注图输出目录")
    parser.add_argument("--csv", default="", help="汇总 CSV（默认 output/records.csv）")
    parser.add_argument("--push", default="", help="推送到指定 IP:PORT（上位机查看）")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    args = parser.parse_args()

    store = ImageStore(default_dir=args.dir or DEFAULT_IMAGE_DIR)
    mm = ModelManager("/home/nvidia/defect_detection/models/baseline/yolov8s_fp16.engine",
                      conf_thres=args.conf, iou_thres=args.iou)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="trt")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    sim = CameraSimulator(store, mm, executor, asyncio.Lock(), loop,
                          fps=args.fps, shuffle=args.shuffle)

    push_sock = None
    if args.push:
        host, _, port = args.push.partition(":")
        push_sock = socket.create_connection((host, int(port)), timeout=10)
        push_sock.sendall(_pack({"type": "stream_subscribe_request", "subscribe": True}))
        log.info(f"已推送订阅: {args.push}")

    os.makedirs(args.output, exist_ok=True)
    csv_path = args.csv or os.path.join(args.output, "records.csv")
    csv_rows = []

    async def _run():
        ok, err = await sim.start()
        if not ok:
            print(f"启动失败: {err}")
            return 1
        # 收帧：监听自身广播（内联订阅为回调）
        async def _on_frame(frame):
            seq = frame["seq"]
            name = frame["name"]
            dets = frame["detections"]
            # 标注图落盘
            if frame.get("annot_b64"):
                raw = base64.b64decode(frame["annot_b64"])
                with open(os.path.join(args.output, f"{seq:04d}_{name}"), "wb") as f:
                    f.write(raw)
            csv_rows.append({"seq": seq, "name": name,
                             "result": "NG" if dets else "OK",
                             "count": len(dets),
                             "ms": round(frame.get("timing", {}).get("total_with_read", 0), 1)})
            print(f"[{seq}] {name}: {'NG ×' + str(len(dets)) if dets else 'OK'}"
                  f"  {frame.get('timing', {}).get('total_with_read', 0):.0f}ms")
            # 推送到指定上位机
            if push_sock is not None:
                try:
                    push_sock.sendall(_pack(frame))
                except OSError as e:
                    print(f"推送失败: {e}")
            if seq >= args.rounds:
                await sim.stop()

        sim._on_frame_cb = _on_frame
        while sim.running:
            await asyncio.sleep(0.01)
        st = sim.state()
        print("=" * 50)
        print(f"完成: {st['total_frames']} 帧, OK={st['ok_count']}, NG={st['ng_count']}, "
              f"avg={st['avg_ms']}ms")
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["seq", "name", "result", "count", "ms"])
            w.writeheader()
            w.writerows(csv_rows)
        print(f"标注图: {args.output}\nCSV: {csv_path}")
        if push_sock is not None:
            push_sock.close()
        return 0

    rc = loop.run_until_complete(_run())
    mm.engine.close()
    sys.exit(rc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    cli_main()
