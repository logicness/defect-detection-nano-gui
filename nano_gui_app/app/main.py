import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
"""
工业缺陷检测工作站（Nano单屏版）
无边框主窗口 + 自定义标题栏 + 5 Tab 导航 + 状态栏
接线：控制器(TCP/数据库) ↔ 5 页面 ↔ 实时流引擎 ↔ NG 归档
（Nano裁剪：无 PLC 直连、无串口、无 Nano推理源）
"""
import os
import sys
import time
import traceback

import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QTabWidget,
    QLabel, QStatusBar, QSizeGrip, QMessageBox, QFileDialog
)
from PyQt5.QtCore import Qt, QRect, QTimer
from PyQt5.QtGui import QFont, QImage

from components.title_bar import TitleBar
from pages import (
    RealtimeDetectPage, ParamSettingPage, HistoryRecordPage,
    CommSettingPage, RunLogPage
)
from core.controller import AppController
from core.stream_engine import StreamEngine
from core.model_manager_ctl import ModelManagerCtl
from core.ng_saver import NGSaver
from core import config as cfg_mod

_EDGE_MARGIN = 6


def _popup_info(parent, title: str, text: str):
    """信息弹窗（offscreen 无头环境跳过，避免崩溃）"""
    if QApplication.platformName() != "offscreen":
        QMessageBox.information(parent, title, text)


def _usm(frame, strength: float = 0.5):
    """预览锐化（USM，2026-09-01）：仅增强显示清晰度，不影响检测/存档。"""
    try:
        blur = cv2.GaussianBlur(frame, (0, 0), 3.0)
        return cv2.addWeighted(frame, 1.0 + strength, blur, -strength, 0)
    except Exception:
        return frame


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint)
        # 分辨率自适应（2026-08-28）：按屏幕可用区域动态调整，
        # 兼容常见外接屏（1280×720 ~ 1920×1080+），避免最小尺寸锁死超出屏幕。
        self._adapt_to_screen()
        # 最小窗口（防止缩得太小界面错乱；随屏幕缩放）
        self.setMinimumSize(self._min_w, self._min_h)

        central = QWidget()
        central.setMouseTracking(True)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 标题栏
        self.title_bar = TitleBar("缺陷检测工作站")
        self.title_bar.window_minimized.connect(self.showMinimized)
        self.title_bar.window_maximized.connect(self._toggle_maximize)
        self.title_bar.window_closed.connect(self.close)
        self.title_bar.menu_requested.connect(self._on_menu)
        layout.addWidget(self.title_bar)
        self.title_bar.set_model_tag("Product_A_v1")
        self.title_bar.set_camera_tag("Camera_01")

        # 核心对象
        self.cfg = cfg_mod.load_config()
        self.controller = AppController()
        # 目录改名自愈（RK3568 -> RK3588）：迁移 state 中失效的旧根路径并持久化
        self._migrate_state_paths()
        # 同步当前模型名，用于 class_id → 类别名 映射
        self.controller.nano_model_name = self.cfg.get(
            "state", {}).get("last_nano_model", "")
        self.model_ctl = ModelManagerCtl(self.controller.tcp)
        # 默认 infer_mode 用 tcp，禁止未配置时误入模拟演示
        self.stream_engine = StreamEngine(infer_mode="tcp")
        self.ng_saver = NGSaver(self.cfg.get("storage", {}).get(
            "save_path", os.path.join(os.path.expanduser("~"), "Inspect", "Images")))
        self._last_frame = None
        self._plc_running = False
        # 本地图片模式：加载图片后直接显示在预览区，点「开始检测」做单帧推理
        self._local_image = None          # numpy BGR 帧
        self._local_image_path = ""       # 图片路径
        self._local_image_active = False  # 是否正处于本地图片模式
        self._nano_active = False         # 是否推理服务已连接（自动切换Nano推理服务模型）
        # 多图批量检测状态
        self._multi_image_paths = []      # 多次模式选中的图片列表
        self._multi_image_idx = 0         # 当前浏览下标
        self._multi_results = {}          # path -> {"dets": [...], "ms": float}
        self._batch_running = False
        self._batch_queue = []            # 待检测图片路径
        self._current_batch_path = ""     # 当前正在检测的图片
        self._nano_local_pending = ""     # 正在等待 Nano 本地图片结果的路径
        # 检测图库（2026-08-24）：与本地图片同交互，图片驻留 Nano，走文件名指令
        self._nano_image_active = False       # 是否处于检测图库模式
        self._nano_image_names = []           # 选中的检测图库文件名（顺序同 _multi_image_paths）
        self._nano_image_dir = ""             # 检测图库文件夹
        self._nano_file_pending = ""          # 正在等待检测图库结果的路径
        self._nano_wait_preview_detect = ""   # 预览就绪后要自动检测的文件名
        self._nano_start_pending = False      # 预览未就绪时点开始检测 → 就绪后自动触发
        self._nano_preview_name = ""          # 当前已显示预览的Nano文件名
        self._stream_camera_active = False    # 是否处于产线流模式
        # 从配置恢复上次本地模型（必须是存在的 .onnx/.pt 模型文件）
        self._local_model = self.cfg.get("state", {}).get("last_local_model", "")
        if self._local_model and not (
                os.path.isfile(self._local_model) and
                self._local_model.lower().endswith((".onnx", ".pt"))):
            self._local_model = ""
        # 没有本地模型时，自动探测 nano 模型训练目录中的默认缺陷检测模型
        if not self._local_model:
            self._local_model = self._find_default_local_model()
            if self._local_model:
                self._save_state(last_local_model=self._local_model)

        # 5 个 Tab
        self.tab = QTabWidget()
        self.tab.setObjectName("mainTabs")
        self.page_realtime = RealtimeDetectPage()
        self.page_param = ParamSettingPage()
        self.page_history = HistoryRecordPage()
        self.page_comm = CommSettingPage()
        self.page_log = RunLogPage()
        self.tab.addTab(self.page_realtime, "实时检测")
        self.tab.addTab(self.page_param, "参数设置")
        self.tab.addTab(self.page_history, "历史记录")
        self.tab.addTab(self.page_comm, "通信设置")
        self.tab.addTab(self.page_log, "运行日志")
        layout.addWidget(self.tab, 1)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_left = QLabel("就绪　|　检测帧率: -- FPS")
        self.status_time = QLabel("")
        self.status_bar.addWidget(self.status_left)
        self.status_bar.addPermanentWidget(self.status_time)
        self.grip = QSizeGrip(self)
        self.status_bar.addPermanentWidget(self.grip)
        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._tick_clock)
        self._clock_timer.start(1000)
        self._tick_clock()

        # 配置回填 + ROI 共享
        self.page_param.apply_config(self.cfg)
        self.page_comm.apply_config(self.cfg)
        self._sync_cfg_to_realtime()
        self._rois = self.cfg.get("rois", [])
        self.page_realtime.set_rois(self._rois)
        self.page_param.set_rois(self._rois)

        # 恢复上次本地模型到界面
        if self._local_model:
            self.title_bar.set_model_tag(os.path.basename(self._local_model))
            self.page_param.set_cur_model(os.path.basename(self._local_model))
            self.page_realtime.set_cur_model(os.path.basename(self._local_model))

        self._wire()
        self._emit_startup_logs()
        self._auto_connect()

        # 边缘缩放状态
        self._maximized = False
        self._resize_dir = None

    def _adapt_to_screen(self):
        """分辨率自适应：读取屏幕可用区域，动态设置窗口尺寸与最小尺寸。
        - 屏幕 ≥1920×1080：保持 1920×1080 设计尺寸（最大化时填满）
        - 屏幕 1280×720 ~ 1919×1079：按可用区域缩放（最小尺寸=可用区域的 80%）
        - 屏幕 <1280×720：仍按 1280×720 下限（工业屏普遍 ≥720p）
        """
        from PyQt5.QtWidgets import QApplication as _QA
        try:
            geo = _QA.desktop().availableGeometry()
            sw, sh = geo.width(), geo.height()
        except Exception:
            sw, sh = 1920, 1080
        # 设计基准 1920×1080；按比例缩放
        if sw >= 1920 and sh >= 1080:
            self._init_w, self._init_h = 1920, 1080
        else:
            self._init_w, self._init_h = sw, sh
        # 最小尺寸 = 可用区域的 80%（下限 1280×720），保证界面可读且不超出屏幕
        self._min_w = max(1280, int(sw * 0.8))
        self._min_h = max(720, int(sh * 0.8))
        self.resize(self._init_w, self._init_h)
        # 记录可用屏幕尺寸供 showMaximized 使用（最大化即填满屏幕）
        self._screen_geo = geo

    # ================= 接线 =================
    def _wire(self):
        c = self.controller

        # 日志 → 运行日志页 + 实时页 mini 日志
        c.log_message.connect(self.page_log.append_log)
        c.log_message.connect(
            lambda lv, mod, msg: self.page_realtime.append_mini_log(lv, msg))

        # FPS → 状态栏 + 日志页
        self.stream_engine.fps_updated.connect(self._on_fps)
        c.fps_updated.connect(self.page_log.set_fps)

        # 状态灯
        c.status_changed.connect(self._on_status_changed)

        # TCP 调试日志（写文件便于诊断 pythonw 无控制台场景）
        c.tcp.log_message.connect(self._log_tcp_debug)
        c.tcp.connected.connect(lambda: self._log_tcp_debug("INFO", "SIGNAL connected"))
        c.tcp.disconnected.connect(lambda: self._log_tcp_debug("INFO", "SIGNAL disconnected"))

        # 推理服务识别 → 自动切换模型（连接切Nano推理服务模型 / 断开切回本地模型）
        c.tcp.connected.connect(self._on_nano_connected)
        c.tcp.disconnected.connect(self._on_nano_disconnected)

        # Nano 系统状态轮询（连接时启动，断开停止）
        c.tcp.status_received.connect(self._on_nano_status)
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(5000)
        self._status_timer.timeout.connect(self._poll_nano_status)

        # 模型管理（Nano清单/切换 → 标题栏与页面联动）
        self.model_ctl.models_updated.connect(self._on_models_updated)
        self.model_ctl.load_result.connect(self._on_model_load_result)

        # 实时流
        self.stream_engine.frame_ready.connect(self._on_stream_frame_local)
        self.stream_engine.result_received.connect(
            lambda r: c.ingest_result(r.get("detections", []), r.get("frame")))
        self.stream_engine.log_message.connect(
            lambda lv, m: c.log_message.emit(lv, "检测", m))
        self.stream_engine.state_changed.connect(self.page_realtime.set_running)

        # 产线流帧合并节流（2026-09-01）：latest-wins + QTimer(80ms) 统一处理，
        # 杜绝每帧 base64/解码/写库全压在 GUI 线程导致的队列堆积与卡死
        self._pending_stream_frame = None
        self._stream_flush_timer = QTimer(self)
        self._stream_flush_timer.setInterval(80)
        self._stream_flush_timer.timeout.connect(self._flush_stream_frame)

        # 产线流断流看门狗（2026-09-01）：TCP 在线但 >5s 无帧 → 自动恢复（start 幂等）
        self._last_stream_frame_ts = 0.0
        self._stream_watchdog_last_ts = 0.0   # 自动恢复冷却（≥10s 一次，防抖）
        self._stream_watchdog_timer = QTimer(self)
        self._stream_watchdog_timer.setInterval(2000)
        self._stream_watchdog_timer.timeout.connect(self._on_stream_watchdog)
        self._stream_watchdog_timer.start()

        # 检测图库预览超时保护（2026-09-01）：服务端事件循环冻结/连接异常时
        # 预览请求可能永不回包 → 界面卡在「加载预览中...」；10s 无回包则
        # 状态栏提示并复位，避免永久黑屏观感。收到回包（成功或失败）即取消。
        self._preview_timeout_timer = QTimer(self)
        self._preview_timeout_timer.setSingleShot(True)
        self._preview_timeout_timer.setInterval(10000)
        self._preview_timeout_timer.timeout.connect(self._on_preview_timeout)

        # 断线重连后自动重新订阅产线流（2026-09-01，服务端订阅按连接计数）
        c.tcp.connected.connect(self._on_tcp_reconnected)

        # 推理结果 → 实时页/历史页/通信页
        c.detection_result.connect(self._on_detection_result)
        c.ng_alarm.connect(self._on_ng_alarm)

        # 实时页
        self.page_realtime.start_requested.connect(self._on_start)
        self.page_realtime.stop_requested.connect(self._on_stop)
        self.page_realtime.save_image_requested.connect(self._on_save_image)
        self.page_realtime.local_image_requested.connect(self._on_local_image)
        self.page_realtime.nano_image_detect_requested.connect(self._on_nano_image_detect)
        self.page_realtime.roi_changed.connect(self._on_roi_changed)
        self.page_realtime.load_model_requested.connect(self._on_load_model)
        self.page_realtime.model_mgr_requested.connect(self._on_model_mgr)
        self.page_realtime.save_path_changed.connect(self._on_save_path_changed)
        self.page_realtime.reconnect_requested.connect(self._on_tcp_reconnect)
        self.page_realtime.camera_params_requested.connect(self._on_camera_params)
        self.page_realtime.auto_exposure_requested.connect(self._on_auto_exposure)
        self.page_realtime.conf_changed.connect(self._on_conf_changed)
        self.page_realtime.prev_image_requested.connect(self._on_nav_prev)
        self.page_realtime.next_image_requested.connect(self._on_nav_next)

        # 检测图库：预览大图 + 单张检测结果回传（2026-08-24）
        c.tcp.nano_image_received.connect(self._on_nano_preview_image)
        c.tcp.nano_detect_received.connect(self._on_nano_detect_result)

        # 产线模拟流（2026-08-25）
        c.tcp.stream_frame_received.connect(self._on_stream_frame)
        c.tcp.stream_control_received.connect(self._on_stream_control)

        # 参数页
        self.page_param.params_apply_requested.connect(self._on_params_apply)
        self.page_param.save_config_requested.connect(self._on_save_config)
        self.page_param.reset_requested.connect(self._on_reset_config)
        self.page_param.roi_changed.connect(self._on_roi_changed)
        self.page_param.load_model_requested.connect(self._on_load_model)
        self.page_param.model_mgr_requested.connect(self._on_model_mgr)

        # 历史页
        self.page_history.query_requested.connect(self._on_history_query)
        self.page_history.export_requested.connect(self._on_history_export)
        self.page_history.report_requested.connect(self._on_history_report)
        self.page_history.clear_history_requested.connect(self._on_history_clear)

        # 通信页
        self.page_comm.reconnect_requested.connect(self._on_tcp_reconnect)
        self.page_comm.plc_connect_requested.connect(self._noop)
        self.page_comm.plc_disconnect_requested.connect(self._noop)
        self.page_comm.test_comm_requested.connect(self._noop)

    @staticmethod
    def _noop(*_a, **_k):
        """Nano版占位：PLC/串口等工作站功能已裁剪，信号保留兼容。"""
        pass

    # ================= 默认本地模型 =================
    def _find_default_local_model(self) -> str:
        """在 nano 模型训练目录中探测默认本地缺陷检测模型（NEU-DET 类别）"""
        candidates = [
            # NANO-GUI：本地模型目录 ~/models（onnx 优先；部署时如有模型文件可放入）
            os.path.join(os.path.expanduser("~"), "models", "best.onnx"),
            os.path.join(os.path.expanduser("~"), "models", "best.pt"),
        ]
        for p in candidates:
            if os.path.isfile(p) and p.lower().endswith((".onnx", ".pt")):
                return p
        # 兜底：递归扫描 runs 目录下最近的 best 模型
        base = os.path.join(os.path.expanduser("~"), "models")
        best_found = None
        latest = 0.0
        if os.path.isdir(base):
            for root, _dirs, files in os.walk(base):
                for f in files:
                    if f in ("best.onnx", "best.pt"):
                        full = os.path.join(root, f)
                        if "_trash_" in full:
                            continue
                        mtime = os.path.getmtime(full)
                        if mtime > latest:
                            latest, best_found = mtime, full
        return best_found or ""

    def _migrate_state_paths(self):
        """目录改名自愈（2026-08 RK3568 -> RK3588）：state 中指向旧根的历史路径，
        若新根下实际存在则迁移并持久化；不存在则保留原值（不臆造）。"""
        try:
            from components.model_library import migrate_legacy_root
        except Exception:
            return
        st = self.cfg.setdefault("state", {})
        changed = False
        for key in ("last_local_model", "last_model_dir", "last_image_dir"):
            v = st.get(key, "")
            if v:
                nv = migrate_legacy_root(v)
                if nv != v:
                    st[key] = nv
                    changed = True
        if changed:
            cfg_mod.save_config(self.cfg)
            self.controller.log_message.emit(
                "INFO", "系统", "检测到历史路径目录改名，已自动迁移本地模型/目录配置")

    # ================= 启动 =================
    def _emit_startup_logs(self):
        msgs = [
            ("INFO", "系统", "软件启动"),
            ("INFO", "系统", "配置文件加载完成"),
        ]
        if self._local_model and os.path.isfile(self._local_model):
            msgs.append(
                ("INFO", "模型",
                 f"默认本地模型: {os.path.basename(self._local_model)}（选择本地图片即可检测）"))
            msgs.append(("INFO", "模型", "连接推理服务后将自动切换为Nano推理服务模型"))
        else:
            msgs.append(("INFO", "系统", "等待连接设备或加载本地图片；未连接前不会启动模拟检测"))
        msgs.append(("INFO", "通信", "TCP 未自动连接，请点击「重新连接」手动连接Nano"))
        for lv, mod, msg in msgs:
            self.controller.log_message.emit(lv, mod, msg)

    def _auto_connect(self):
        """Nano单屏版：启动时自动连接Nano回环推理服务（127.0.0.1:8888）。
        推理服务由 systemd(defect-infer.service) 守护，开机即在线；
        若失败仅记录日志，用户可点「重新连接」重试。"""
        t = self.cfg.get("tcp", {})
        host = t.get("host", "127.0.0.1")
        port = t.get("port", 8888)
        self.controller.configure_tcp(
            host, port,
            heartbeat=t.get("heartbeat", 5), retries=t.get("retries", 1),
            timeout=t.get("timeout", 10))
        self.controller.connect_tcp()
        self.controller.log_message.emit(
            "INFO", "通信", f"已自动连接Nano推理服务 {host}:{port}（Nano回环模式）")

    def _on_plc_connect(self):
        """PLC 直连已裁剪（Nano版由 infer_server 服务层对接产线 PLC）。"""
        pass

    # ================= 检测流 =================
    def _on_start(self):
        # 新一次检测开始时，允许接收推理结果
        self._detection_paused = False

        # 产线模拟流模式：订阅 + 启动产线（Nano自主检测，工作站只接收）
        if "产线流" in self.page_realtime.get_infer_source():
            self._on_start_stream_camera()
            return

        # 检测图库模式：预览未就绪时挂起，预览回传后自动重入本方法
        if getattr(self, "_nano_image_active", False) and self._local_image is None:
            self._nano_start_pending = True
            self.status_left.setText("检测图库: 预览加载中，就绪后自动检测...")
            return

        # 已加载本地图片 → 推理（单次/多次）
        if self._local_image is not None:
            if self.stream_engine.is_running:
                self._on_stop()
            self._local_image_active = True
            self.page_realtime.set_running(True)

            # 多次模式 + 多张图片 → 批量检测
            multi = self.page_realtime.get_detect_mode() == "多次检测"
            if multi and len(self._multi_image_paths) > 1:
                self._start_batch()
                return
            # 单次模式 或 多次模式只有一张 → 单张检测
            self._detect_local_image()
            return

        # 没有本地图片时的实时流启停控制
        if self.stream_engine.is_running:
            self._on_stop()
            return

        # 没有选择图片、也没有真实相机输入源 → 明确提示，绝不启动模拟流
        self._local_image_active = False
        self.controller.log_message.emit(
            "WARN", "检测", "没有可用的检测输入源，未启动检测")
        if QApplication.platformName() != "offscreen":
            QMessageBox.information(
                self, "开始检测",
                "当前没有可用的检测输入源，无法开始检测。\n\n"
                "请先点击「本地图片」选择图片进行检测。")

    @staticmethod
    def _is_torchscript_model(path: str) -> bool:
        """检测 .pt 文件是否为 TorchScript 格式（无法直接用 YOLO 推理）"""
        # 文件名是最快的判断方式
        if "torchscript" in os.path.basename(path).lower():
            return True
        try:
            import torch
            # TorchScript 模型用 torch.jit.load 能直接加载
            # 普通 PyTorch 权重用 torch.load 会加载为 dict/OrderedDict
            m = torch.jit.load(path, map_location="cpu")
            # 如果能加载且类型是 ScriptModule / RecursiveScriptModule，即为 TorchScript
            return "ScriptModule" in type(m).__name__
        except Exception:
            return False

    def _detect_local_image(self):
        """对本地图片做单帧推理（单张/多次只有一张时）。

        推理源下拉选择「Nano 模型」→ 走 TCP 下发位机推理（与批量一致）；
        否则走 Nano推理 core.local_infer.LocalInferEngine 后台线程
        （异步执行、信号回传、不阻塞 UI）。
        """
        # 确保不跟实时流同时跑
        if self.stream_engine.is_running:
            self.stream_engine.stop()
        # 检测图库模式：板端读自己的图片 + 板端当前模型，走 nano_detect_request
        if getattr(self, "_nano_image_active", False):
            self._nano_start_detect()
            return
        path = self._local_image_path
        basename = os.path.basename(path)

        # ⭐ 推理源 = Nano 模型（非产线流）→ 单张也走Nano回环推理（对齐批量逻辑）
        source = self.page_realtime.get_infer_source()
        if "产线流" not in source:
            if not self.controller.tcp.is_connected:
                self.controller.log_message.emit(
                    "WARN", "检测", "推理源为Nano 模型但推理服务未连接，请先重新连接")
                self.status_left.setText(f"本地图片: {basename}　| 推理服务未连接")
                return
            self._nano_local_pending = path
            self.status_left.setText(f"本地图片: {basename}　| Nano推理中...")
            self.controller.log_message.emit(
                "INFO", "检测", f"开始 Nano推理: {basename}")
            ok, buf = cv2.imencode(".jpg", self._local_image)
            if ok:
                self.controller.send_image(buf.tobytes(), path)
                del buf  # 释放编码缓冲
            else:
                self._nano_local_pending = ""
                self.controller.log_message.emit(
                    "ERROR", "检测", f"图片编码失败: {basename}")
            return

        # 没有本地模型
        if not self._local_model:
            self.controller.log_message.emit(
                "WARN", "检测", "未设置本地模型，无法检测本地图片")
            if QApplication.platformName() != "offscreen" and not self._batch_running:
                QMessageBox.information(
                    self, "本地图片检测",
                    "当前未设置 Nano 模型。\n\n"
                    "请先到「模型管理」或点击「加载模型」选择 .onnx / .pt 模型，\n"
                    "然后再进行本地图片检测。")
            return

        # 模型文件不存在
        if not os.path.isfile(self._local_model):
            self.controller.log_message.emit(
                "WARN", "检测", f"本地模型文件不存在: {self._local_model}")
            if QApplication.platformName() != "offscreen" and not self._batch_running:
                QMessageBox.warning(
                    self, "本地图片检测",
                    f"模型文件不存在或已被删除:\n{self._local_model}\n\n"
                    "请重新选择本地模型。")
            return

        ext = os.path.splitext(self._local_model)[1].lower()
        # 格式不支持
        if ext not in (".onnx", ".pt"):
            self.controller.log_message.emit(
                "WARN", "检测",
                f"本地模型格式 {ext} 暂不支持，请使用 .onnx / .pt 模型")
            if QApplication.platformName() != "offscreen" and not self._batch_running:
                QMessageBox.information(
                    self, "本地图片检测",
                    f"当前本地模型:\n{self._local_model}\n\n"
                    f"格式 {ext} 暂不支持 Nano推理。\n"
                    "请加载 .onnx 或 .pt 模型后再检测。")
            return

        # .pt 先排除 TorchScript 模型（文件名含 torchscript 或加载后类型不符）
        if self._is_torchscript_model(self._local_model):
            self.controller.log_message.emit(
                "WARN", "检测",
                f"TorchScript 模型暂不支持本地推理: {os.path.basename(self._local_model)}")
            if QApplication.platformName() != "offscreen" and not self._batch_running:
                QMessageBox.warning(
                    self, "本地图片检测",
                    f"当前模型是 TorchScript 格式：\n{self._local_model}\n\n"
                    "该格式无法直接用于 Nano推理。\n\n"
                    "请使用以下方式之一解决：\n"
                    "1. 换成标准的 PyTorch 训练权重 (.pt)\n"
                    "2. 导出为 ONNX 格式 (.onnx) 后加载\n\n"
                    "例如：python export.py --weights yolov5s.pt --include onnx")
            return

        # 复用同一个推理引擎（懒加载），结果/错误经信号回主线程
        if getattr(self, "_local_infer_engine", None) is None:
            from core.local_infer import LocalInferEngine
            eng = LocalInferEngine(self)
            eng.result_ready.connect(self._on_local_infer_result)
            eng.error_ready.connect(self._on_local_infer_error)
            self._local_infer_engine = eng

        if self._local_infer_engine.busy:
            self.controller.log_message.emit(
                "WARN", "检测", "本地推理正在进行，请稍候")
            return

        # 启动后台推理（不阻塞 UI），状态先行提示
        self.status_left.setText(f"本地图片: {basename}　|　推理中...")
        self.controller.log_message.emit(
            "INFO", "检测", f"开始本地推理: {basename}（模型 {os.path.basename(self._local_model)}）")
        self._local_infer_engine.detect(path, self._local_model,
                                        conf_thres=0.25, iou_thres=0.45)

    def _on_local_infer_result(self, result: dict):
        """本地推理完成（后台线程回传）→ 转 UI 格式 → 统一入管线（KPI/写库/历史/预览画框）"""
        # 用户已点击停止，忽略延迟到达的推理结果
        if getattr(self, "_detection_paused", False):
            return
        from core.class_names import resolve_class_names, class_name_of
        class_names = resolve_class_names(self._local_model)
        raw = result.get("detections", [])
        dets = []
        for d in raw:
            box = d.get("box", [0, 0, 0, 0])
            cid = d.get("class_id", 0)
            cls = class_name_of(class_names, cid)
            dets.append((cls, d.get("confidence", 0), *box[:4]))
        ms = result.get("timing", {}).get("total_ms", 0) or 0
        path = self._local_image_path or ""
        basename = os.path.basename(path) or "本地图片"

        # 再次检查：如果停止按钮已触发，则只记录不渲染
        if getattr(self, "_detection_paused", False):
            self.controller.log_message.emit(
                "INFO", "检测", f"本地推理完成但已停止，忽略结果: {basename}")
            self.status_left.setText(
                f"本地图片: {basename}　| 已停止")
            return

        self.controller.ingest_result(dets, self._local_image, path)
        self._multi_results[path] = {"dets": list(dets), "ms": ms}
        self.controller.log_message.emit(
            "INFO", "检测",
            f"本地图片检测完成: {basename} "
            f"({'NG 缺陷×' + str(len(dets)) if dets else 'OK'})  耗时 {ms:.0f} ms")
        self.status_left.setText(
            f"本地图片: {basename}　| 检测完成 {ms:.0f} ms")
        if self._batch_running:
            self.page_realtime.update_batch_progress(
                len(self._multi_results), len(self._multi_image_paths))
            self._batch_next()

    def _on_local_infer_error(self, msg: str):
        """本地推理失败（后台线程回传）"""
        self.controller.log_message.emit("ERROR", "检测", f"本地推理失败: {msg}")
        self.page_realtime.set_running(False)
        self.status_left.setText(
            f"本地图片: {os.path.basename(self._local_image_path) or '--'}　| 推理失败")
        if QApplication.platformName() != "offscreen":
            QMessageBox.warning(
                self, "本地图片检测",
                f"模型推理时出错:\n{msg}\n\n"
                "请检查模型文件是否完整，或尝试其他 .onnx / .pt 模型。")

    def _on_stop(self, cancel_batch: bool = True):
        """停止实时流；保留本地图片模式，方便用户再次点击开始检测同一图片。
        cancel_batch=False 仅供内部加载图片时清框使用（不清空批量队列）。"""
        # 标记停止：后续延迟到达的推理结果不再渲染，避免停止后框又出现
        self._detection_paused = True
        # 用户主动停止 → 取消未完成的批量检测：清队列/清 Nano 挂起请求，
        # 避免延迟结果继续翻页或画框
        if cancel_batch:
            was_batch = (getattr(self, "_batch_running", False)
                         or bool(self._batch_queue) or bool(self._nano_local_pending)
                         or bool(getattr(self, "_nano_file_pending", "")))
            self._batch_running = False
            self._batch_queue = []
            self._nano_local_pending = ""
            self._nano_file_pending = ""
            self._nano_wait_preview_detect = ""
            self._nano_start_pending = False
            if was_batch:
                done = len(self._multi_results)
                total = len(self._multi_image_paths)
                self.page_realtime.lbl_batch_prog.setVisible(True)
                self.page_realtime.lbl_batch_prog.setText(
                    f"批量检测已停止（{done}/{total} 完成）")
                self.page_realtime.lbl_batch_prog.setStyleSheet(
                    "color:#f59e0b; font-size:13px; background:transparent;")
                self.page_realtime.bar_batch.setVisible(False)
        # 产线流：发 stop 指令（保留订阅，可重启）
        if getattr(self, "_stream_camera_active", False):
            try:
                self.controller.tcp.control_stream("stop")
            except Exception:
                pass
            self._stream_camera_active = False
            self.controller.log_message.emit("INFO", "检测", "产线检测已停止")

        if self.stream_engine.is_running:
            self.stream_engine.stop()
        eng = getattr(self, "_local_infer_engine", None)
        if eng is not None and getattr(eng, "busy", False):
            try:
                eng.cancel()
            except Exception:
                pass
        self.page_realtime.set_running(False)

        # 内部加载新图时清框（cancel_batch=False）；用户点停止时保留最后一帧结果（cancel_batch=True）
        if not cancel_batch:
            try:
                self.page_realtime.preview.clear_detections()
                self.page_realtime.preview.repaint()
            except Exception:
                pass
            self.page_realtime.update_detections([])
            try:
                self.page_realtime.preview.clear_detections()
                self.page_realtime.preview.repaint()
            except Exception:
                pass
            self.page_realtime.update_detail(
                "Product_A_v1", "--", "--", 0, "--",
                time.strftime("%Y-%m-%d %H:%M:%S"),
                self._local_image_path or "--")
            if (self._local_image_active or getattr(self, "_nano_image_active", False)) \
                    and self._local_image is not None:
                rgb = cv2.cvtColor(self._local_image, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                self.page_realtime.update_image(
                    QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy())

        # 用户点停止：保留检测结果（框+详情+KPI），更新状态栏提示
        if cancel_batch:
            self.status_left.setText("检测已停止　|　结果保留，可再次检测")

        self.controller.log_message.emit("INFO", "检测", "检测已停止")

    def _safe_clear_detections(self):
        """（已废弃）停止检测后不再清框，保留最后一帧结果。保留方法避免外部调用崩溃。"""
        pass

    def _on_stream_frame_local(self, frame):
        """本地帧源预览（stream_engine.frame_ready）——与产线流 TCP 帧处理区分。"""
        self._last_frame = frame

    def _on_fps(self, fps):
        self.status_left.setText(f"运行中　|　检测帧率: {fps:.0f} FPS")
        self.page_log.set_fps(fps)

    def _on_detection_result(self, result: dict):
        # 用户已点击停止，忽略延迟到达的结果（保留最后一帧的检测框不被覆盖）
        if getattr(self, "_detection_paused", False):
            return

        dets = result.get("detections", [])
        frame = result.get("frame")
        if frame is None:
            frame = self._last_frame
        path = result.get("image_path", "")
        is_stream = str(path or "").startswith("stream://")
        verdict = "NG" if dets else "OK"
        ts = time.strftime("%Y-%m-%d %H:%M:%S")

        # 预览 + 画框（产线流帧已在 _flush_stream_frame 显示标注图，跳过避免双框/重复绘制）
        if frame is not None and not is_stream:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            self.page_realtime.update_image(
                QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy())
        if not is_stream:
            self.page_realtime.update_detections(dets)

        # 右侧详情 + KPI
        first = dets[0] if dets else None
        if dets:
            # 汇总缺陷类型及数量：inclusion ×1, scratches ×2
            from collections import Counter
            cnt = Counter(d[0] for d in dets)
            type_str = ", ".join(f"{cls} ×{n}" for cls, n in cnt.items())
            # 取面积最大的缺陷框作为代表（面积/置信度）
            biggest = max(dets, key=lambda d: abs((d[4] - d[2]) * (d[5] - d[3])))
            area = int(abs((biggest[4] - biggest[2]) * (biggest[5] - biggest[3])))
            conf = f"{biggest[1]:.2f}"
        else:
            type_str, area, conf = "-", 0, "--"
        self.page_realtime.update_detail(
            "Product_A_v1", verdict, type_str,
            area, conf, ts,
            path or os.path.join(os.path.expanduser("~"), "Inspect", "Images", "sample.png"))
        st = self.controller.get_stats()
        yld = st["pass"] / st["total"] * 100 if st["total"] else 0
        self.page_realtime.update_kpi(st["total"], st["pass"], st["defect"], yld)

        # 底部历史
        self.page_realtime.add_history([
            time.strftime("%H:%M:%S"), "Product_A_v1", verdict,
            first[0] if first else "-", os.path.basename(path) if path else "--"])

        # NG 归档
        if dets and frame is not None:
            fid = int(time.time() * 1000) % 100000
            self.ng_saver.enqueue_save(frame, dets, fid)
            self.ng_saver.enqueue_low_conf(frame, dets, fid)

        # 日志
        if dets:
            self.controller.log_message.emit(
                "ERROR", "检测",
                f"检测到缺陷：{first[0]}，置信度 {first[1]:.2f}")
        else:
            self.controller.log_message.emit("INFO", "检测", "检测通过")

        # Nano 本地图片检测结果：存结果并继续下一张（批量）或更新状态（单张）
        if self._nano_local_pending:
            path = self._nano_local_pending
            self._nano_local_pending = ""
            ms = 0
            self._multi_results[path] = {"dets": list(dets), "ms": ms}
            if self._batch_running:
                self.page_realtime.update_batch_progress(
                    len(self._multi_results), len(self._multi_image_paths))
                self._batch_next()
            else:
                self.status_left.setText(
                    f"本地图片: {os.path.basename(path)}　| Nano检测完成")
        # 检测图库结果：存结果并继续下一张（批量）或更新状态（单张）
        if self._nano_file_pending:
            path = self._nano_file_pending
            self._nano_file_pending = ""
            self._multi_results[path] = {"dets": list(dets), "ms": 0}
            if self._batch_running:
                self.page_realtime.update_batch_progress(
                    len(self._multi_results), len(self._multi_image_paths))
                self._batch_next()
            else:
                self.status_left.setText(
                    f"检测图库: {os.path.basename(path)}　| Nano检测完成")

    def _on_ng_alarm(self, count):
        self.controller.log_message.emit(
            "ERROR", "检测", f"连续 {count} 次检出缺陷，请检查产线状态！")
        # 微信告警推送（企业微信/Server酱，配置 data/alarm_config.json）
        try:
            from core.alarm import get_pusher
            st = self.controller.get_stats()
            msg = (f"⚠️ 缺陷检测告警\n"
                   f"连续 {count} 次检出缺陷\n"
                   f"当前累计: 总 {st['total']} / 缺陷 {st['defect']} / 良率 "
                   f"{st['pass']/st['total']*100:.1f}%\n"
                   f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            if get_pusher().push(msg):
                self.controller.log_message.emit("INFO", "告警", "微信告警推送已受理（后台发送中）")
        except Exception as e:
            self.controller.log_message.emit("WARN", "告警", f"微信告警失败: {e}")
        if QApplication.platformName() == "offscreen":
            return  # 无头模式不弹窗

        # 避免弹窗堆叠：复用同一个非模态告警框，5 分钟内不重复新建
        now = time.time()
        cooldown = getattr(self, "_ng_alarm_cooldown", 0)
        if now < cooldown:
            return

        box = getattr(self, "_ng_alarm_box", None)
        if box is None:
            box = QMessageBox(self)
            box.setWindowTitle("连续 NG 告警")
            box.setIcon(QMessageBox.Warning)
            box.setWindowModality(Qt.NonModal)
            box.setStandardButtons(QMessageBox.Ok)
            box.finished.connect(lambda _: setattr(self, "_ng_alarm_box", None))
            self._ng_alarm_box = box

        box.setText(f"连续检出 {count} 个缺陷！")
        box.setInformativeText("可能原因：产线异常 / 相机失焦 / 光源异常，请立即检查。")
        box.show()
        box.raise_()
        box.activateWindow()
        self._ng_alarm_cooldown = now + 300  # 5 分钟冷却

    # ================= 配置/ROI =================
    def _sync_cfg_to_realtime(self):
        """配置回填实时页（2026-09-01 修复：移除已删除的 combo_camera/spin_bright，
        曝光/增益与实时页新控件（μs/dB）对齐，默认 22000μs / 15dB）。"""
        cam = self.cfg.get("camera", {})
        det = self.cfg.get("detect", {})
        st = self.cfg.get("storage", {})
        p = self.page_realtime
        p.spin_exposure.setValue(float(cam.get("exposure", 22000.0)))
        p.spin_gain.setValue(float(cam.get("gain", 15.0)))
        p.spin_conf.setValue(float(det.get("conf", 0.85)))
        p.spin_area.setValue(int(det.get("min_area", 50)))
        p.edit_save_path.setText(st.get("save_path", os.path.join(os.path.expanduser("~"), "Inspect", "Images")))
        p.combo_clean.setCurrentText(st.get("auto_clean", "磁盘空间 < 10% 时删除"))

    def _on_roi_changed(self, rois):
        self._rois = rois
        # 两页互相同步（避免回环：仅更新不同步的一方）
        sender = self.sender()
        if sender is not self.page_realtime:
            self.page_realtime.set_rois(rois)
        if sender is not self.page_param:
            self.page_param.set_rois(rois)
        # 持久化 ROI 变更
        self.cfg["rois"] = rois
        cfg_mod.save_config(self.cfg)

    def _on_save_config(self, cfg: dict):
        merged = dict(self.cfg)
        merged.update(cfg)
        self.cfg = merged
        cfg_mod.save_config(merged)
        self.ng_saver.set_save_root(
            merged.get("storage", {}).get("save_path", os.path.join(os.path.expanduser("~"), "Inspect", "Images")))
        self.controller.log_message.emit("INFO", "系统", "配置已保存")
        _popup_info(self, "保存成功", "参数配置已保存，重启后仍会生效。")

    def _on_reset_config(self):
        self.cfg = _deep_default()
        self.page_param.apply_config(self.cfg)
        self._sync_cfg_to_realtime()
        self.page_realtime.set_rois(self.cfg["rois"])
        self.page_param.set_rois(self.cfg["rois"])
        self.controller.log_message.emit("INFO", "系统", "已恢复默认配置")
        _popup_info(self, "已恢复默认", "所有参数已恢复为默认值。")

    def _on_save_path_changed(self, path: str):
        self.cfg.setdefault("storage", {})["save_path"] = path
        cfg_mod.save_config(self.cfg)
        self.ng_saver.set_save_root(path)
        self.page_param.apply_config(self.cfg)
        self.controller.log_message.emit("INFO", "系统", f"保存路径已更新: {path}")
        _popup_info(self, "已设置", f"图像保存路径已更新：\n{path}")

    def _on_params_apply(self, conf, iou):
        rois = [{"x": r["x"], "y": r["y"], "w": r["w"], "h": r["h"]}
                for r in self._rois if r.get("enabled", True)]
        # 相机参数（参数设置页）一并下发（仅真实相机帧源生效，sim 帧源服务端忽略）
        exp = getattr(self.page_param, "spin_exposure", None)
        gain = getattr(self.page_param, "spin_gain", None)
        cam_exp = float(exp.value()) if exp is not None else None
        cam_gain = float(gain.value()) if gain is not None else None
        if self.controller.tcp.is_connected:
            self.controller.tcp.send_control(conf_thres=conf, iou_thres=iou, rois=rois)
            if cam_exp is not None and cam_gain is not None:
                self.controller.tcp.control_camera(exposure_us=cam_exp, gain=cam_gain)
                self.controller.log_message.emit(
                    "INFO", "参数", f"已下发相机参数: 曝光={cam_exp:.1f}us 增益={cam_gain:.1f}dB")
            self.controller.log_message.emit(
                "INFO", "参数", f"已下发参数: conf={conf}, roi×{len(rois)}")
            _popup_info(self, "应用成功",
                        f"检测参数已下发到Nano：\n置信度 {conf}，ROI {len(rois)} 个"
                        + (f"\n相机: 曝光 {cam_exp:.1f}us / 增益 {cam_gain:.1f}dB"
                           if cam_exp is not None else ""))
        else:
            self.controller.log_message.emit("INFO", "参数",
                                             f"参数已应用（本地）: conf={conf}")
            _popup_info(self, "应用成功",
                        f"检测参数已应用（未连接Nano，仅本地生效）：\n置信度 {conf}")

    def _on_conf_changed(self, conf: float):
        """实时页置信度数值变化 → 即时下发（无需点应用，检测中直接生效）"""
        try:
            if self.controller.tcp.is_connected:
                rois = [{"x": r["x"], "y": r["y"], "w": r["w"], "h": r["h"]}
                        for r in self._rois if r.get("enabled", True)]
                self.controller.tcp.send_control(conf_thres=float(conf), rois=rois)
            else:
                self.stream_engine.set_local_conf(float(conf))
            self.cfg.setdefault("detect", {})["conf"] = float(conf)
            self.controller.log_message.emit(
                "DEBUG", "参数", f"置信度实时更新: {conf:.2f}")
        except Exception as e:
            self._log_tcp_debug("WARN", f"实时置信度下发失败: {e}")

    def _on_camera_params(self, exposure_us: float, gain: float):
        """实时页「应用相机参数」→ stream_control set_camera 下发到产线相机。
        2026-09-01 完善：改为异步回显——实际生效值/错误由服务端
        stream_control_response(set_camera) 回传后显示（见 _on_stream_control）。"""
        try:
            if not self.controller.tcp.is_connected:
                self.controller.log_message.emit(
                    "WARN", "相机", "推理服务未连接，无法下发相机参数")
                self.page_realtime.lbl_cam_state.setText("推理服务未连接")
                return
            self.controller.tcp.control_camera(exposure_us=exposure_us, gain=gain)
            self.controller.log_message.emit(
                "INFO", "相机",
                f"相机参数下发中: 曝光={exposure_us:.0f}μs 增益={gain:.1f}dB（等待回显确认）")
        except Exception as e:
            self._log_tcp_debug("WARN", f"相机参数下发失败: {e}")
            self.page_realtime.lbl_cam_state.setText("下发失败")

    def _on_auto_exposure(self, target_brightness: float = 120.0):
        """一键自动曝光（实时页按钮）→ stream_control auto_exposure。
        结果经 stream_control_response(auto_exposure) 回显并回填（见 _on_stream_control）。"""
        try:
            if not self.controller.tcp.is_connected:
                self.controller.log_message.emit(
                    "WARN", "相机", "推理服务未连接，无法自动曝光")
                self.page_realtime.lbl_cam_state.setText("推理服务未连接")
                return
            self.page_realtime.lbl_cam_state.setText("自动曝光中...")
            self.controller.tcp.control_auto_exposure(target_brightness=float(target_brightness))
            self.controller.log_message.emit(
                "INFO", "相机", f"一键自动曝光下发中（目标亮度 {target_brightness:.0f}）")
        except Exception as e:
            self._log_tcp_debug("WARN", f"自动曝光下发失败: {e}")
            self.page_realtime.lbl_cam_state.setText("下发失败")

    def _on_stream_watchdog(self):
        """产线流断流看门狗（2026-09-01）：TCP 在线但 >5s 无帧 → 自动恢复。
        start 幂等（服务端已在运行会返回"已在运行"），带 10s 冷却防抖。"""
        if not getattr(self, "_stream_camera_active", False):
            return
        if not self.controller.tcp.is_connected:
            return
        now = time.time()
        if now - self._last_stream_frame_ts <= 5.0:
            return
        if now - getattr(self, "_stream_watchdog_last_ts", 0.0) < 10.0:
            return
        self._stream_watchdog_last_ts = now
        self.controller.log_message.emit(
            "WARN", "检测", "产线流中断（>5s 无帧），自动恢复中...")
        try:
            self.controller.tcp.subscribe_stream(True)
            fps = self.page_realtime.spin_stream_fps.value()
            src = self.page_realtime.get_stream_source()
            self.controller.tcp.control_stream("start", fps=fps, source=src)
        except Exception as e:
            self.controller.log_message.emit(
                "WARN", "检测", f"产线流自动恢复失败: {e}")

    # ================= 模型 =================
    def _on_load_model(self, name=""):
        """「加载模型」入口：打开模型选择对话框，默认显示「选择模型」Tab"""
        from components.model_manager_dialog import ModelManagerDialog
        dlg = ModelManagerDialog(
            self.controller.tcp, self,
            default_model=self._local_model,
            default_model_dir=self.cfg.get("state", {}).get("last_model_dir", ""),
            open_tab="select"
        )
        dlg.local_model_selected.connect(self._on_local_model_selected)
        dlg.local_model_deleted.connect(self._on_local_model_deleted)
        dlg.reconnect_requested.connect(self._on_model_mgr_reconnect)
        dlg.exec_()

    def _on_model_mgr(self):
        """「模型管理」入口：打开模型管理对话框，默认显示「模型管理」Tab"""
        from components.model_manager_dialog import ModelManagerDialog
        dlg = ModelManagerDialog(
            self.controller.tcp, self,
            default_model=self._local_model,
            default_model_dir=self.cfg.get("state", {}).get("last_model_dir", ""),
            open_tab="manage"
        )
        dlg.local_model_selected.connect(self._on_local_model_selected)
        dlg.local_model_deleted.connect(self._on_local_model_deleted)
        dlg.reconnect_requested.connect(self._on_model_mgr_reconnect)
        dlg.exec_()

    def _on_local_model_deleted(self, path: str):
        """模型管理对话框中删除了本地模型文件/条目：若正是当前本地模型则清空"""
        if path and self._local_model == path:
            self._local_model = ""
            self._save_state(last_local_model="")
            self.title_bar.set_model_tag("--")
            self.controller.log_message.emit(
                "WARN", "模型", "当前本地模型已被删除，请重新选择模型")
        else:
            self.controller.log_message.emit(
                "INFO", "模型", f"已从模型库移除: {os.path.basename(path)}")

    def _on_models_updated(self, data):
        """Nano 模型清单变化 → 同步标题栏/页面当前模型"""
        active = data.get("active", "")
        if not active:
            return
        base = os.path.basename(active)
        self.title_bar.set_model_tag(base)
        self.page_realtime.set_cur_model(base)
        self.page_param.set_cur_model(base)
        self.controller.log_message.emit("INFO", "模型", f"当前模型: {base}")
        # 记忆 Nano 当前模型 + 同步类别映射
        self._save_state(last_nano_model=active)
        self.controller.nano_model_name = active

    def _on_model_load_result(self, payload):
        """Nano 模型切换结果 → 同步标题栏/页面"""
        if not payload.get("ok"):
            self.controller.log_message.emit(
                "WARN", "模型", f"切换失败: {payload.get('error', '')}")
            return
        model = payload.get("model", "")
        base = os.path.basename(model)
        self.title_bar.set_model_tag(base)
        self.page_realtime.set_cur_model(base)
        self.page_param.set_cur_model(base)
        self.controller.log_message.emit("INFO", "模型", f"模型切换成功: {base}")
        # 记忆 Nano 当前模型 + 同步类别映射
        self._save_state(last_nano_model=model)
        self.controller.nano_model_name = model

    def _on_nano_connected(self):
        """推理服务已连接 → 自动切换为Nano推理服务模型"""
        self._nano_active = True
        nano_model = self.cfg.get("state", {}).get("last_nano_model", "")
        base = os.path.basename(nano_model) if nano_model else "Nano 模型"
        self.title_bar.set_model_tag(base)
        self.page_realtime.set_cur_model(base)
        self.page_param.set_cur_model(base)
        self.controller.log_message.emit(
            "INFO", "模型", f"推理服务已连接，已切换为Nano推理服务模型: {base}")
        self.status_left.setText(f"推理服务已连接　|　模型: {base}")
        # 启动 Nano 状态轮询
        if not getattr(self, "_status_timer", None):
            self._status_timer = QTimer(self)
            self._status_timer.setInterval(5000)
            self._status_timer.timeout.connect(self._poll_nano_status)
        self._status_timer.start()
        self._poll_nano_status()
        # 连接推理服务后默认切换为产线流检测（真实相机）
        self.page_realtime.set_infer_source("实时产线流")
        self.model_ctl.refresh()
        self.page_log.set_nano_online(True)
        # 连接成功弹窗已禁用（避免自动重连时频繁弹窗）
        # 如需恢复弹窗，取消下面注释
        # now = time.time()
        # if now - getattr(self, "_last_conn_popup_ts", 0) >= 10:
        #     self._last_conn_popup_ts = now
        #     if QApplication.platformName() != "offscreen":
        #         tcp = self.controller.tcp
        #         QMessageBox.information(
        #             self, "连接成功",
        #             f"已成功连接推理服务（{getattr(tcp, 'host', '')}:{getattr(tcp, 'port', '')}），"
        #             f"当前模型：{base}")

    def _on_nano_disconnected(self):
        """Nano推理服务断开 → 自动切回本地模型"""
        self._nano_active = False
        if self._local_model and os.path.isfile(self._local_model):
            base = os.path.basename(self._local_model)
            self.title_bar.set_model_tag(base)
            self.page_realtime.set_cur_model(base)
            self.page_param.set_cur_model(base)
            self.controller.log_message.emit(
                "INFO", "模型", f"推理服务已断开，切回本地模型: {base}")
        else:
            self.title_bar.set_model_tag("--")
            self.controller.log_message.emit("INFO", "模型", "推理服务已断开，无本地模型")
        if not self.stream_engine.is_running:
            self.status_left.setText("就绪　|　检测帧率: -- FPS")
        # 停止 Nano 状态轮询
        if getattr(self, "_status_timer", None):
            self._status_timer.stop()
        # 断开后保持Nano推理源（Nano版始终走回环推理，断开时仅提示重连）
        self.page_realtime.set_infer_source("Nano 模型")
        self.page_log.set_nano_online(False)

    # ---------------- Nano 状态轮询 ----------------
    def _poll_nano_status(self):
        """周期请求Nano系统状态（仅已连接时）"""
        if self.controller.tcp.is_connected:
            self.controller.tcp.request_status()

    def _on_nano_status(self, status: dict):
        """status_response → 状态栏显示Nano GPU/CPU/内存/温度/推理耗时 + 相机温度"""
        try:
            gpu = status.get("gpu_util")
            cpu = status.get("cpu_util")
            mem = status.get("mem_used_gb")
            temp = status.get("gpu_temp")
            ms = status.get("last_detect_ms")
            cam_temp = status.get("camera_temp_c")
            parts = []
            parts.append(f"GPU {gpu}%" if gpu is not None else "GPU --")
            parts.append(f"CPU {cpu}%" if cpu is not None else "CPU --")
            if mem is not None:
                parts.append(f"内存 {mem:.1f}G")
            if temp is not None:
                parts.append(f"{temp:.0f}°C")
            # 相机状态（2026-09-01）：在线/离线优先（camera_online），温度兜底，sim 显示 --
            cam_online = status.get("camera_online")
            if cam_online is not None:
                parts.append("相机 在线" if cam_online else "相机 离线")
            elif cam_temp is not None:
                parts.append(f"相机 {cam_temp:.0f}°C")
            else:
                parts.append("相机 --")
            if ms is not None:
                parts.append(f"推理 {ms:.0f}ms")
            model = status.get("model")
            suffix = f" | 模型: {model}" if model else ""
            self.status_left.setText(" | ".join(parts) + suffix)
            self.page_log.update_nano_status(status)
        except Exception as e:
            self._log_tcp_debug("WARN", f"状态显示异常: {e}")

    def _on_local_model_selected(self, path):
        import os
        self._local_model = path
        self.controller.log_message.emit(
            "INFO", "模型", f"已选择本地模型 {os.path.basename(path)}，开始检测时启用本地推理")
        self.title_bar.set_model_tag(os.path.basename(path))
        self.page_param.set_cur_model(os.path.basename(path))
        self.page_realtime.set_cur_model(os.path.basename(path))
        # 持久化
        self._save_state(last_local_model=path,
                         last_model_dir=os.path.dirname(path) or "")

    def _save_state(self, **kwargs):
        """更新并保存 ui_state 到配置文件"""
        state = dict(self.cfg.get("state", {}))
        state.update(kwargs)
        self.cfg["state"] = state
        cfg_mod.save_config(self.cfg)

    # ================= 历史 =================
    def _on_history_query(self, filters, limit, offset):
        db = self.controller.db
        rows = db.query_records(limit=limit, offset=offset, **filters)
        total = db.count_records(**filters)
        kpi = db.get_kpi(**filters)
        self.page_history.set_defect_types(db.get_defect_types())
        self.page_history.render(rows, total, kpi)
        # A4 统计图表刷新
        try:
            daily = db.get_daily_stats(days=14)
            types = db.get_defect_type_stats()
            self.page_history.update_charts(daily, types)
        except Exception as e:
            self.controller.log_message.emit("WARN", "历史", f"图表刷新失败: {e}")

    def _on_history_export(self, filters):
        path, _ = QFileDialog.getSaveFileName(
            self, "导出记录", "defect_records.csv", "CSV (*.csv)")
        if not path:
            return
        n = self.controller.db.export_csv(path, **filters)
        self.controller.log_message.emit("INFO", "历史", f"已导出 {n} 条到 {path}")
        _popup_info(self, "导出成功", f"已导出 {n} 条检测记录到：\n{path}")

    def _on_history_report(self, filters):
        """U9 检测报告导出（NG 拼图 + 统计，HTML）"""
        default_name = f"defect_report_{time.strftime('%Y%m%d_%H%M%S')}.html"
        path, _ = QFileDialog.getSaveFileName(
            self, "导出检测报告", default_name, "HTML 报告 (*.html)")
        if not path:
            return
        try:
            report = self.controller.db.export_report(path, **filters)
            if report["count"] == 0:
                _popup_info(self, "无数据", "当前筛选条件下没有检测记录")
                return
            self.controller.log_message.emit(
                "INFO", "历史",
                f"已导出报告: {report['count']} 条 (NG {report['ng']}, "
                f"良率 {report['yield']}%)")
            _popup_info(self, "导出成功",
                        f"检测报告已导出：\n{path}\n\n"
                        f"记录 {report['count']} 条 · NG {report['ng']} · "
                        f"良率 {report['yield']}%")
        except Exception as e:
            self.controller.log_message.emit("ERROR", "历史", f"报告导出失败: {e}")
            _popup_info(self, "导出失败", f"报告导出失败：\n{e}")

    def _on_history_clear(self):
        """清空所有检测记录与生产统计"""
        try:
            n = self.controller.db.clear_all_records()
            self.controller.log_message.emit("INFO", "历史", f"已清空 {n} 条记录")
            _popup_info(self, "清空成功", "所有检测记录已清空。")
            self.page_history._query(1)
        except Exception as e:
            self.controller.log_message.emit("ERROR", "历史", f"清空失败: {e}")
            _popup_info(self, "清空失败", f"清空记录时出错：\n{e}")

    # ================= 通信（Nano回环） =================
    def _on_plc_status(self, status):
        """PLC 已裁剪（工作站功能）。保留占位，防止旧信号意外触发崩溃。"""
        pass

    def _on_test_comm(self):
        """PLC 通信测试已裁剪。保留占位。"""
        pass

    def _on_tcp_reconnect(self):
        """手动重新连接Nano推理服务（断开旧线程后立即重连）"""
        self.controller.tcp.disconnect()
        t = self.cfg.get("tcp", {})
        host = t.get("host", "192.168.1.101")
        port = t.get("port", 8888)
        self.controller.configure_tcp(
            host, port,
            heartbeat=t.get("heartbeat", 5), retries=t.get("retries", 3),
            timeout=t.get("timeout", 10))
        self.controller.connect_tcp()
        self.controller.log_message.emit(
            "INFO", "通信", f"正在重新连接Nano {host}:{port}，结果请关注右下角运行日志")

    def _on_model_mgr_reconnect(self):
        """模型管理对话框里的重新连接按钮"""
        self._on_tcp_reconnect()

    def _log_tcp_debug(self, level, msg):
        """把 TCP 相关日志追加到文件，便于 pythonw 无控制台时排查"""
        import os, time
        try:
            path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "data", "tcp_debug.log")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%H:%M:%S')} [{level}] {msg}\n")
        except Exception:
            pass

    def _on_status_changed(self, name, status):
        bar = {"camera": self.title_bar.light_camera,
               "model": self.title_bar.light_model}.get(name)
        if bar:
            bar.set_status(status)
        if name == "camera":
            self.page_realtime.set_camera_light(status == 1)
        if name == "model":
            self.page_realtime.light_nano_mini.set_status(
                status, "已连接" if status == 1 else "未连接")
            # Nano回环：推理服务(8888)连接状态联动通信页
            self.page_comm.set_plc_conn_ui(status == 1)
            if status == 1:
                self.controller.tcp.request_model_list()

    # ================= 杂项 =================
    def _on_save_image(self, path):
        if self._last_frame is not None:
            cv2.imwrite(path, self._last_frame)
            self.controller.log_message.emit("INFO", "系统", f"图像已保存: {path}")
            _popup_info(self, "保存成功", f"图像已保存到：\n{path}")
        else:
            self.controller.log_message.emit("WARN", "系统", "无可用帧，保存失败")
            _popup_info(self, "保存失败", "当前没有可保存的图像帧。\n\n"
                        "请先加载本地图片或开始实时检测后再保存。")

    def _tick_clock(self):
        self.status_time.setText(
            f"当前时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    def _on_menu(self):
        from PyQt5.QtWidgets import QMenu
        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu { background-color: #1e293b; color: #e2e8f0; border: 1px solid #334155;"
            " font-size: 15px; padding: 4px; }"
            "QMenu::item { padding: 8px 24px; border-radius: 4px; }"
            "QMenu::item:selected { background-color: #2563eb; color: #ffffff; }")
        act_nano = menu.addAction("检测图库")
        act_model = menu.addAction("模型管理")
        menu.addSeparator()
        act_about = menu.addAction("关于")
        act = menu.exec_(self.title_bar.mapToGlobal(self._menu_pos()))
        if act == act_nano:
            self._on_nano_image_detect()
        elif act == act_model:
            self._on_model_mgr()
        elif act == act_about:
            QMessageBox.about(self, "关于",
                              "缺陷检测工作站（Nano单屏版）v1.3\n"
                              "运行平台: Jetson Orin Nano Super 8G\n"
                              "推理: Nano回环 127.0.0.1:8888 (infer_server, TensorRT)\n"
                              "相机: MV-CS050-10UC USB3.0 (MVS SDK 5.0.2)\n"
                              "产线流: 真实相机 / 图片模拟 双帧源")

    def _menu_pos(self):
        """菜单弹出位置：标题栏菜单按钮附近"""
        return self.title_bar.rect().topRight() - self.title_bar.rect().topLeft() \
            + self.title_bar.pos()

    # ================= 检测图库（2026-08-24，与本地图片同交互） =================
    def _nano_path_of(self, name: str) -> str:
        return f"nano://{self._nano_image_dir}/{name}"

    @staticmethod
    def _nano_name_of(path: str) -> str:
        return os.path.basename(str(path or "").replace("\\", "/"))

    def _nano_load_preview_by_name(self, name: str):
        """拉取检测图库预览大图（640px）显示到主界面预览区"""
        self.status_left.setText(f"检测图库: {name}　| 加载预览中...")
        self._preview_timeout_timer.start()  # 10s 无回包 → 超时提示（2026-09-01）
        self.controller.tcp.request_nano_image(name, 640)

    def _nano_load_preview(self, path: str):
        """按 nano:// 路径加载预览（对齐 _load_image_file 的入口语义）"""
        name = self._nano_name_of(path)
        if path in self._multi_image_paths:
            self._nano_image_idx = self._multi_image_paths.index(path)
        self._nano_load_preview_by_name(name)

    def _nano_start_detect(self):
        """单帧检测：对当前浏览的检测图库发起检测"""
        if not self._nano_image_names:
            return
        idx = max(0, min(self._nano_image_idx, len(self._nano_image_names) - 1))
        self._nano_start_detect_for(self._nano_image_names[idx])

    def _nano_start_detect_for(self, name: str):
        """发起检测图库：预览帧已在手直接发；否则先拉预览，就绪后自动发"""
        if self._nano_preview_name == name and self._local_image is not None:
            self.controller.tcp.request_nano_detect(name, annotate=False)
            return
        self._nano_wait_preview_detect = name
        self._nano_load_preview_by_name(name)

    def _batch_next_nano(self, path: str):
        """批量流程：检测图库走 预览→检测 链路（不读本地文件）"""
        self._detection_paused = False
        self.page_realtime.update_nav(self._multi_image_idx, len(self._multi_image_paths))
        self.page_realtime.set_running(True)
        self.page_realtime.update_batch_progress(
            len(self._multi_results), len(self._multi_image_paths))
        name = self._nano_name_of(path)
        if not self.controller.tcp.is_connected:
            self._multi_results[path] = {"dets": [], "ms": 0}
            self._batch_next()
            return
        self.controller.log_message.emit(
            "INFO", "检测",
            f"检测图库 [{self._multi_image_idx+1}/{len(self._multi_image_paths)}]: {name}")
        self.status_left.setText(
            f"批量检测 [{self._multi_image_idx+1}/{len(self._multi_image_paths)}] 检测图库推理中...")
        # 先拉预览显示新图，预览就绪后自动续发检测（批量中逐张动态切换）
        self._nano_wait_preview_detect = name
        self._nano_load_preview_by_name(name)

    def _update_detail_for_path(self, path: str):
        """用 _multi_results 更新右侧详情表：缺陷类型汇总（同类显示 ×数量）+ 最大框信息"""
        r = self._multi_results.get(path, {})
        dets = r.get("dets", [])
        verdict = "NG" if dets else "OK"
        if dets:
            from collections import Counter
            cnt = Counter(d[0] for d in dets)
            type_str = ", ".join(f"{cls} ×{n}" for cls, n in cnt.items())
            biggest = max(dets, key=lambda d: abs((d[4] - d[2]) * (d[5] - d[3])))
            area = int(abs((biggest[4] - biggest[2]) * (biggest[5] - biggest[3])))
            conf = f"{biggest[1]:.2f}"
        else:
            type_str, area, conf = "-", 0, "--"
        self.page_realtime.update_detail(
            "Product_A_v1", verdict, type_str, area, conf,
            time.strftime("%Y-%m-%d %H:%M:%S"), path or "--")

    def _on_nano_preview_image(self, msg: dict):
        """检测图库预览回传：显示原图（对齐本地图片加载）；等待检测时自动续发"""
        if not getattr(self, "_nano_image_active", False):
            return
        self._preview_timeout_timer.stop()  # 收到回包（成功或失败）即取消超时（2026-09-01）
        name = msg.get("name", "")
        if not msg.get("ok"):
            if self._nano_wait_preview_detect == name:
                self._nano_wait_preview_detect = ""
                path = self._nano_path_of(name)
                self._multi_results[path] = {"dets": [], "ms": 0,
                                             "error": msg.get("error", "")}
                self.status_left.setText(
                    f"检测图库: {name}　| 预览失败 {msg.get('error', '')}")
                if self._batch_running:
                    self.page_realtime.update_batch_progress(
                        len(self._multi_results), len(self._multi_image_paths))
                    self._batch_next()
            else:
                self.status_left.setText(
                    f"检测图库预览失败: {msg.get('error', '')}")
            return
        import base64
        import numpy as np
        raw = base64.b64decode(msg["image_b64"])
        img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        del raw  # 释放 base64 解码数据
        if img is None:
            return
        self._nano_preview_name = name
        self._local_image = img  # 保持引用，后续检测用
        self._local_image_path = self._nano_path_of(name)
        self._local_image_active = True
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        del rgb  # 释放 RGB 数据
        self.page_realtime.update_image(qimg)
        path = self._local_image_path
        if path in self._multi_results:
            r = self._multi_results[path]
            self.page_realtime.update_detections(r.get("dets", []))
            self._update_detail_for_path(path)
            self.status_left.setText(
                f"检测图库: {name}　| 检测完成 {r.get('ms', 0):.0f} ms")
        else:
            self.page_realtime.update_detections([])
            self.status_left.setText(f"检测图库: {name}　| 点击开始检测")
        self.page_realtime.update_nav(self._nano_image_idx, len(self._multi_image_paths))
        # 预览就绪 → 自动续发等待中的检测
        if self._nano_wait_preview_detect == name:
            self._nano_wait_preview_detect = ""
            self.controller.tcp.request_nano_detect(name, annotate=False)

    def _on_preview_timeout(self):
        """检测图库预览 10s 无回包（服务端事件循环冻结/连接异常）→ 状态栏提示并复位。
        2026-09-01 新增：配合 _preview_timeout_timer，避免界面永久停在
        「加载预览中...」的黑屏观感；用户可点「重新连接」恢复。"""
        self._nano_wait_preview_detect = ""
        self._nano_start_pending = False
        self.status_left.setText(
            "检测图库预览超时（10s 无回包）| 推理服务可能已异常，请点「重新连接」")
        self.controller.log_message.emit(
            "WARN", "检测",
            "检测图库预览 10s 无回包（推理服务可能已冻结），请检查服务状态并重新连接")
        # 开始检测时预览未就绪 → 就绪后自动重入 _on_start
        if getattr(self, "_nano_start_pending", False):
            self._nano_start_pending = False
            self._on_start()

    def _on_nano_detect_result(self, msg: dict):
        """检测图库单张检测结果（nano_detect_response）：入管线 + 批量续接"""
        if getattr(self, "_detection_paused", False):
            return
        if not getattr(self, "_nano_image_active", False):
            return
        name = msg.get("name", "")
        path = self._nano_path_of(name)
        if not msg.get("ok"):
            self._multi_results[path] = {"dets": [], "ms": 0,
                                         "error": msg.get("error", "")}
            self.status_left.setText(f"检测图库: {name}　| 检测失败 {msg.get('error', '')}")
            if self._batch_running:
                self.page_realtime.update_batch_progress(
                    len(self._multi_results), len(self._multi_image_paths))
                self._batch_next()
            return
        from core.class_names import resolve_class_names, class_name_of
        class_names = resolve_class_names(self.controller.nano_model_name)
        dets = []
        for d in msg.get("detections", []):
            box = d.get("box", [0, 0, 0, 0]) or [0, 0, 0, 0]
            b = []
            for v in box[:4]:
                try:
                    b.append(float(v))
                except (TypeError, ValueError):
                    b.append(0.0)
            while len(b) < 4:
                b.append(0.0)
            cid = d.get("class_id", 0)
            cls = class_name_of(class_names, cid)
            dets.append((cls, float(d.get("confidence", 0) or 0), *b))
        try:
            ms = float(msg.get("timing", {}).get("total_with_read", 0) or 0)
        except (TypeError, ValueError):
            ms = 0.0
        # 统一结果管线：预览/KPI/写库/历史/批量续接 由 _on_detection_result 完成
        self._nano_file_pending = path
        self.controller.ingest_result(dets, self._local_image, path)
        if not self._batch_running:
            self.status_left.setText(f"检测图库: {name}　| 检测完成 {ms:.0f} ms")

    def _on_local_image(self):
        """本地图片：单次模式选一张，多次模式可选多张。
        选好后直接显示在实时预览区，不弹窗。"""
        # 切换到本地图片模式：退出检测图库模式
        self._nano_image_active = False
        self._nano_image_names = []
        last_dir = self.cfg.get("state", {}).get("last_image_dir", "")
        start_dir = ""
        if self._local_model:
            from components.model_info import resolve_dataset_image_dir
            start_dir = resolve_dataset_image_dir(self._local_model)
        if not start_dir:
            start_dir = last_dir

        multi = self.page_realtime.get_detect_mode() == "多次检测"
        if multi:
            paths, _ = QFileDialog.getOpenFileNames(
                self, "选择检测图片（可多选）", start_dir,
                "图片文件 (*.png *.jpg *.jpeg *.bmp)")
            if not paths:
                return
            self._multi_image_paths = paths
            self._multi_image_idx = 0
            self._multi_results.clear()
            self._save_state(last_image_dir=os.path.dirname(paths[0]) or "")
            self._on_stop()
            self._local_image_active = True
            self._load_image_file(paths[0])
            self.page_realtime.update_nav(0, len(paths))
            source = self.page_realtime.get_infer_source()
            self.page_realtime.set_source(
                f"{'实时产线流' if '产线流' in source else 'Nano 模型'}（多图 {len(paths)} 张）", "#22d3ee")
            self.controller.log_message.emit("INFO", "检测", f"已选择 {len(paths)} 张图片，点击「开始检测」批量检测")
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "选择检测图片", start_dir,
                "图片文件 (*.png *.jpg *.jpeg *.bmp)")
            if not path:
                if self._local_image_active:
                    self._local_image = None
                    self._local_image_path = ""
                    self._local_image_active = False
                    self.page_realtime.set_source("--", "#94a3b8")
                    self.status_left.setText("就绪　|　检测帧率: -- FPS")
                return
            self._multi_image_paths = [path]
            self._multi_image_idx = 0
            self._multi_results.clear()
            self._save_state(last_image_dir=os.path.dirname(path) or "")
            self._on_stop()
            self._local_image_active = True
            self._load_image_file(path)
            source = self.page_realtime.get_infer_source()
            self.page_realtime.set_source(
                f"{'实时产线流' if '产线流' in source else 'Nano 模型'}（单图）", "#22d3ee")
            self.controller.log_message.emit("INFO", "检测", f"已加载本地图片: {os.path.basename(path)}")

    def _load_image_file(self, path: str):
        """加载图片文件到预览区（单图/多图共用）"""
        import numpy as np
        try:
            _raw = np.fromfile(path, dtype=np.uint8)
            _img = cv2.imdecode(_raw, cv2.IMREAD_COLOR)
        except Exception:
            _img = None
        if _img is None:
            self.controller.log_message.emit("ERROR", "检测", f"无法读取图片: {path}")
            return
        # 内部清框不取消批量（批量流程中 _batch_next 会调用本方法逐张加载）
        self._on_stop(cancel_batch=False)
        self._local_image = _img
        self._local_image_path = path
        self._local_image_active = True
        rgb = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        self.page_realtime.update_image(
            QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy())
        self.page_realtime.update_detections([])
        basename = os.path.basename(path)
        self.status_left.setText(f"本地图片: {basename}　|　点击开始检测")

    def _start_batch(self):
        """启动批量检测：依次检测所有未检测的图片"""
        # Nano 模型推理源但推理服务未连接 → 明确中止，绝不把图片静默记为 OK
        source = self.page_realtime.get_infer_source()
        if "产线流" not in source and not self.controller.tcp.is_connected:
            self.controller.log_message.emit(
                "WARN", "检测",
                "推理源为Nano 模型但推理服务未连接，批量检测未启动。请先重新连接推理服务")
            self.status_left.setText("批量检测未启动　|　推理服务未连接")
            self.page_realtime.set_running(False)
            return
        # Nano版：无 Nano推理（统一走Nano回环），本段校验移除
        self._batch_running = True
        self._batch_queue = [p for p in self._multi_image_paths if p not in self._multi_results]
        total = len(self._multi_image_paths)
        done = len(self._multi_results)
        self.page_realtime.update_batch_progress(done, total)
        if not self._batch_queue:
            self._finish_batch()
            return
        self.controller.log_message.emit("INFO", "检测", f"开始批量检测 {len(self._batch_queue)} 张图片")
        self._batch_next()

    def _batch_next(self):
        """检测队列中下一张图片"""
        if not self._batch_queue:
            self._finish_batch()
            return
        path = self._batch_queue.pop(0)
        self._current_batch_path = path
        if path in self._multi_image_paths:
            self._multi_image_idx = self._multi_image_paths.index(path)
        # 检测图库模式：不读本地文件，走 预览→检测 链路
        if getattr(self, "_nano_image_active", False):
            self._batch_next_nano(path)
            return
        self._load_image_file(path)
        # ⚠️ 关键：_load_image_file 内部调了 _on_stop() 会设 _detection_paused=True，
        # 必须在这里重置为 False，否则推理结果回来时会被忽略，批量检测卡住
        self._detection_paused = False
        self.page_realtime.update_nav(self._multi_image_idx, len(self._multi_image_paths))
        self.page_realtime.set_running(True)
        self.page_realtime.update_batch_progress(
            len(self._multi_results), len(self._multi_image_paths))

        source = self.page_realtime.get_infer_source()
        if "产线流" not in source and self.controller.tcp.is_connected:
            # Nano推理（日志明确记录推理源，便于排查）
            self._nano_local_pending = path
            self.controller.log_message.emit(
                "INFO", "检测",
                f"Nano推理 [{self._multi_image_idx+1}/{len(self._multi_image_paths)}]: "
                f"{os.path.basename(path)}")
            self.status_left.setText(f"批量检测 [{self._multi_image_idx+1}/{len(self._multi_image_paths)}] Nano推理中...")
            ok, buf = cv2.imencode(".jpg", self._local_image)
            if ok:
                self.controller.send_image(buf.tobytes(), path)
            else:
                self._multi_results[path] = {"dets": [], "ms": 0}
                self._batch_next()
        else:
            # Nano版：无 Nano推理，Nano 模型走回环；此处仅处理异常路径
            if "产线流" in source:
                # 产线流不参与图片批量（由实时流独立处理）
                self._multi_results[path] = {"dets": [], "ms": 0}
                self._batch_next()
            else:
                self._detect_local_image()
                eng = getattr(self, "_local_infer_engine", None)
                if eng is None or not eng.busy:
                    if not self.controller.tcp.is_connected:
                        self.controller.log_message.emit(
                            "WARN", "检测", "推理服务未连接，跳过此图片")
                    self._multi_results[path] = {"dets": [], "ms": 0}
                    self._batch_next()

    def _finish_batch(self):
        """批量检测完成：回到第一张，显示结果"""
        self._batch_running = False
        self._batch_queue = []
        self.page_realtime.set_running(False)
        done = len(self._multi_results)
        total = len(self._multi_image_paths)
        self.page_realtime.update_batch_progress(done, total)
        self.controller.log_message.emit("INFO", "检测", f"批量检测完成 {done}/{total} 张")
        if self._multi_image_paths:
            self._multi_image_idx = 0
            self._show_multi_image(0)
        self.status_left.setText(f"批量检测完成 {done}/{total} 张　|　显示第一张结果")

    def _show_multi_image(self, idx: int):
        """加载多图列表中的第 idx 张并显示检测结果（不清框、不触发停止）"""
        paths = self._multi_image_paths
        if idx < 0 or idx >= len(paths):
            return
        path = paths[idx]
        self._multi_image_idx = idx
        # 检测图库模式：异步拉预览，回传时自动套用已存结果
        if getattr(self, "_nano_image_active", False):
            self._nano_image_idx = idx
            self._nano_load_preview(path)
            self.page_realtime.update_nav(idx, len(paths))
            return
        # 直接加载图片显示，不调 _load_image_file（它会触发 _on_stop 清框）
        import numpy as np
        try:
            _raw = np.fromfile(path, dtype=np.uint8)
            _img = cv2.imdecode(_raw, cv2.IMREAD_COLOR)
        except Exception:
            _img = None
        if _img is None:
            self.controller.log_message.emit("ERROR", "检测", f"无法读取图片: {path}")
            return
        self._local_image = _img
        self._local_image_path = path
        self._local_image_active = True
        # 显示原图
        rgb = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        self.page_realtime.update_image(
            QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy())
        # 如果已有检测结果，显示框
        if path in self._multi_results:
            r = self._multi_results[path]
            dets = r.get("dets", [])
            self.page_realtime.update_detections(dets)
            self._update_detail_for_path(path)
            ms = r.get("ms", 0)
            self.status_left.setText(f"多图检测 [{idx+1}/{len(paths)}] {os.path.basename(path)}　| {ms:.0f}ms")
        else:
            self.page_realtime.update_detections([])
            self.status_left.setText(f"多图检测 [{idx+1}/{len(paths)}] {os.path.basename(path)}　| 未检测")
        self.page_realtime.update_nav(idx, len(paths))

    def _on_nav_prev(self):
        """多图导航：上一张（批量检测进行中不响应，避免与自动翻页竞争）"""
        if self._batch_running:
            return
        self._show_multi_image(self._multi_image_idx - 1)

    def _on_nav_next(self):
        """多图导航：下一张（批量检测进行中不响应，避免与自动翻页竞争）"""
        if self._batch_running:
            return
        self._show_multi_image(self._multi_image_idx + 1)

    def _on_batch_detect(self):
        """实时页「批量检测」：打开多图检测对话框，支持 Nano / Nano"""
        from components.local_image_detect import LocalImageDetectDialog
        dlg = LocalImageDetectDialog(
            local_model=self._local_model,
            rois=[{"x": r["x"], "y": r["y"], "w": r["w"], "h": r["h"]}
                  for r in self._rois if r.get("enabled", True)],
            parent=self,
            default_image_dir=self.cfg.get("state", {}).get("last_image_dir", ""),
            tcp_client=self.controller.tcp,
        )
        dlg.result_committed.connect(self._on_batch_result)
        dlg.exec_()

    # ================= 产线流（2026-08-25；Nano版真实相机/图片模拟双帧源） =================
    def _on_start_stream_camera(self):
        """开始产线流：订阅 + start + FPS + 帧源（camera=真实相机 / sim=图片模拟）"""
        if not self.controller.tcp.is_connected:
            self.controller.log_message.emit(
                "WARN", "检测", "产线流需要连接推理服务，请先「重新连接」")
            if QApplication.platformName() != "offscreen":
                QMessageBox.information(
                    self, "产线检测", "推理服务未连接，请先「重新连接」。")
            return
        fps = self.page_realtime.spin_stream_fps.value()
        src = self.page_realtime.get_stream_source()   # camera / sim
        self._stream_camera_active = True
        self._detection_paused = False
        self.page_realtime.set_running(True)
        self.controller.tcp.subscribe_stream(True)
        self.controller.tcp.control_stream("start", fps=fps, source=src)
        # 2026-09-01：camera 帧源自动重放上次保存的曝光/增益（保持检测条件一致）
        if src == "camera":
            cam = self.cfg.get("camera", {})
            if cam.get("exposure") and cam.get("gain"):
                try:
                    self.controller.tcp.control_camera(
                        exposure_us=float(cam["exposure"]), gain=float(cam["gain"]))
                except Exception:
                    pass
        src_cn = "真实相机" if src == "camera" else "图片模拟"
        self.controller.log_message.emit(
            "INFO", "检测", f"产线检测启动: {fps} FPS，帧源={src_cn}，Nano自主检测中...")
        self.status_left.setText(f"产线运行中　|　{src_cn} {fps} FPS　|　等待Nano推帧...")

    def _on_stream_frame(self, msg: dict):
        """产线流帧回传（轻量入口，2026-09-01 改合并节流）：
        只保存最新一帧，由 QTimer(80ms) 统一处理，防止 Qt 队列无限堆积导致主线程过载。"""
        if getattr(self, "_detection_paused", False):
            return
        if not getattr(self, "_stream_camera_active", False):
            return
        self._last_stream_frame_ts = time.time()  # 断流看门狗判据
        self._pending_stream_frame = msg
        if not self._stream_flush_timer.isActive():
            self._stream_flush_timer.start()

    def _flush_stream_frame(self):
        """合并节流处理：取最新一帧做解码/显示/入管线（最多 ~12 帧/s，中间帧丢弃）。"""
        msg = self._pending_stream_frame
        self._pending_stream_frame = None
        if msg is None:
            self._stream_flush_timer.stop()
            return
        # 停止/暂停后到达的残留帧：清空即返回
        if getattr(self, "_detection_paused", False) or \
                not getattr(self, "_stream_camera_active", False):
            self._stream_flush_timer.stop()
            return
        try:
            import base64
            import numpy as np
            name = msg.get("name", "")
            seq = msg.get("seq", 0)
            dets_raw = msg.get("detections", [])
            # 标注图解码（已有框+标签，直接显示避免双框）
            frame = None
            if msg.get("annot_b64"):
                raw = base64.b64decode(msg["annot_b64"])
                frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
                del raw  # 释放 base64 解码数据
            if frame is None:
                return
            # 预览锐化（仅显示用，不影响检测/存档）
            if self.page_realtime.get_preview_sharpen():
                frame = _usm(frame)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
            del rgb  # 释放 RGB 数据
            self.page_realtime.update_image(qimg)
            # 画面质量（对焦/曝光辅助，~1Hz 随帧推送）
            if msg.get("quality"):
                self.page_realtime.update_stream_quality(msg["quality"])
            # detections → UI 格式
            from core.class_names import resolve_class_names, class_name_of
            class_names = resolve_class_names(self.controller.nano_model_name)
            dets = []
            for d in dets_raw:
                box = d.get("box", [0, 0, 0, 0]) or [0, 0, 0, 0]
                b = []
                for v in box[:4]:
                    try:
                        b.append(float(v))
                    except (TypeError, ValueError):
                        b.append(0.0)
                while len(b) < 4:
                    b.append(0.0)
                cid = d.get("class_id", 0)
                cls = class_name_of(class_names, cid)
                dets.append((cls, float(d.get("confidence", 0) or 0), *b))
            # 标注帧已含框，不再 update_detections 重复画框（2026-09-01 去双框）
            # 详情表（Counter ×数量）
            path = f"stream://{name}"
            verdict = "NG" if dets else "OK"
            if dets:
                from collections import Counter
                cnt = Counter(d[0] for d in dets)
                type_str = ", ".join(f"{c} ×{n}" for c, n in cnt.items())
                biggest = max(dets, key=lambda dd: abs((dd[4]-dd[2])*(dd[5]-dd[3])))
                area = int(abs((biggest[4]-biggest[2])*(biggest[5]-biggest[3])))
                conf = f"{biggest[1]:.2f}"
            else:
                type_str, area, conf = "-", 0, "--"
            self.page_realtime.update_detail(
                "Product_A_v1", verdict, type_str, area, conf,
                time.strftime("%Y-%m-%d %H:%M:%S"), path)
            # 入管线（KPI/写库/NG 归档/PLC 剔除/历史）
            self.controller.ingest_result(dets, frame, path)
            del frame  # 释放 BGR 帧（ingest_result 之后再释放）
            ms = msg.get("timing", {}).get("total_with_read", 0)
            self.status_left.setText(
                f"产线流 #{seq} {name}　| {'NG ×'+str(len(dets)) if dets else 'OK'} {ms:.0f}ms")
        except Exception as e:
            import traceback as _tb
            self.controller.log_message.emit("ERROR", "检测", f"流帧处理异常: {e}")
            _crash_log(_tb.format_exc())

    def _on_tcp_reconnected(self):
        """断线重连后（2026-09-01 v2）：产线流仍在运行时自动重新订阅 + 自动重启流。
        服务重启后流状态丢失（running=False），若不恢复则预览永久冻结；
        start 幂等：服务端流已在运行时会返回"已在运行"，无副作用。"""
        if not getattr(self, "_stream_camera_active", False):
            return
        self.controller.tcp.subscribe_stream(True)
        try:
            fps = self.page_realtime.spin_stream_fps.value()
            src = self.page_realtime.get_stream_source()
            self.controller.tcp.control_stream("start", fps=fps, source=src)
            self.controller.log_message.emit(
                "INFO", "检测", f"TCP 已重连，产线流自动恢复（{src} {fps} FPS）")
        except Exception as e:
            self.controller.log_message.emit(
                "WARN", "检测", f"重连后自动恢复产线流失败: {e}")

    def _on_stream_control(self, msg: dict):
        """产线流控制/订阅/状态响应"""
        if msg.get("type") == "stream_subscribe_response":
            if msg.get("ok"):
                self.page_realtime.update_stream_state(msg)
            return
        if msg.get("type") == "stream_control_response":
            action = msg.get("action", "")
            src_cn = "真实相机" if msg.get("source") == "camera" else (
                "图片模拟" if msg.get("source") == "sim" else "")
            if action == "start" and msg.get("ok"):
                self.page_realtime.update_stream_state(msg)
                fps = msg.get("fps", 0)
                suffix = f"，帧源={src_cn}" if src_cn else ""
                self.controller.log_message.emit(
                    "INFO", "检测", f"产线检测已启动: {fps} FPS{suffix}")
            elif action == "stop":
                self.page_realtime.update_stream_state({})
                self.controller.log_message.emit("INFO", "检测", "产线检测已停止")
            elif action == "set_fps":
                self.page_realtime.update_stream_state(msg)
            elif action == "set_camera":
                # 2026-09-01：相机参数下发结果回显（applied=实际生效值）
                cam = msg.get("camera") or {}
                if msg.get("ok"):
                    exp = cam.get("exposure_us")
                    gain = cam.get("gain")
                    txt = (f"已生效: 曝光 {exp:.0f}μs / 增益 {gain:.1f}dB"
                           if exp is not None else "已生效")
                    self.page_realtime.lbl_cam_state.setText(txt)
                    self.controller.log_message.emit(
                        "INFO", "相机", txt)
                    if exp is not None and gain is not None:
                        self.cfg.setdefault("camera", {})
                        self.cfg["camera"]["exposure"] = float(exp)
                        self.cfg["camera"]["gain"] = float(gain)
                        cfg_mod.save_config(self.cfg)
                else:
                    self.page_realtime.lbl_cam_state.setText(
                        f"失败: {msg.get('error', '未知')}")
                    self.controller.log_message.emit(
                        "WARN", "相机", f"相机参数下发失败: {msg.get('error', '未知')}")
            elif action == "auto_exposure":
                # 2026-09-01：一键自动曝光结果回显 + 回填两个页面 + 持久化
                if msg.get("ok"):
                    cam = msg.get("camera") or {}
                    exp = cam.get("exposure_us")
                    gain = cam.get("gain")
                    if exp is not None and gain is not None:
                        txt = f"自动曝光完成: 曝光 {exp:.0f}μs / 增益 {gain:.1f}dB"
                        self.page_realtime.spin_exposure.setValue(float(exp))
                        self.page_realtime.spin_gain.setValue(float(gain))
                        sp = getattr(self.page_param, "spin_exposure", None)
                        if sp is not None:
                            sp.setValue(float(exp))
                        sp2 = getattr(self.page_param, "spin_gain", None)
                        if sp2 is not None:
                            sp2.setValue(float(gain))
                        self.cfg.setdefault("camera", {})
                        self.cfg["camera"]["exposure"] = float(exp)
                        self.cfg["camera"]["gain"] = float(gain)
                        cfg_mod.save_config(self.cfg)
                    else:
                        txt = "自动曝光完成"
                    self.page_realtime.lbl_cam_state.setText(txt)
                    self.controller.log_message.emit("INFO", "相机", txt)
                else:
                    self.page_realtime.lbl_cam_state.setText(
                        f"失败: {msg.get('error', '未知')}")
                    self.controller.log_message.emit(
                        "WARN", "相机", f"自动曝光失败: {msg.get('error', '未知')}")

    def _on_nano_image_detect(self):
        """「检测图库」入口：弹出检测图库选择器（与本地图片同交互）。
        单次/多次模式控制单选/多选；选图后主界面预览，点「开始检测」触发检测。"""
        if not self.controller.tcp.is_connected:
            self.controller.log_message.emit(
                "WARN", "检测", "检测图库需要连接推理服务，请先「重新连接」")
            if QApplication.platformName() != "offscreen":
                QMessageBox.information(
                    self, "检测图库",
                    "推理服务未连接。\n\n"
                    "请先点击「重新连接」连上推理服务，\n"
                    "再选择检测图库。")
            return
        if self.stream_engine.is_running:
            self._on_stop()
        from PyQt5.QtWidgets import QDialog
        from components.nano_image_picker import NanoImagePickerDialog
        multi = self.page_realtime.get_detect_mode() == "多次检测"
        dlg = NanoImagePickerDialog(
            tcp_client=self.controller.tcp, multi=multi, parent=self,
            default_dir=self.cfg.get("state", {}).get("last_nano_image_dir", ""))
        if dlg.exec_() != QDialog.Accepted or not dlg.selected_names:
            return
        names = list(dlg.selected_names)
        self._nano_image_dir = dlg.image_dir or self._nano_image_dir
        self._nano_image_names = names
        self._nano_image_active = True
        self._nano_image_idx = 0
        self._nano_file_pending = ""
        self._nano_wait_preview_detect = ""
        self._nano_start_pending = False
        self._multi_image_paths = [self._nano_path_of(n) for n in names]
        self._multi_image_idx = 0
        self._multi_results.clear()
        self._on_stop()
        self._save_state(last_nano_image_dir=self._nano_image_dir)
        self._nano_load_preview(self._multi_image_paths[0])
        self.page_realtime.update_nav(0, len(names))
        label = f"检测图库（{'多图 ' + str(len(names)) + ' 张' if len(names) > 1 else '单图'}）"
        self.page_realtime.set_source(label, "#22d3ee")
        self.controller.log_message.emit(
            "INFO", "检测", f"已选择检测图库 {len(names)} 张，点击「开始检测」检测")

    def _on_batch_result(self, data: dict):
        """批量检测结果回传：进入主流程（KPI / 历史 / 数据库）"""
        try:
            self.controller.ingest_result(
                data.get("dets", []), data.get("frame"), data.get("image_path", ""))
        except Exception as e:
            self._log_tcp_debug("WARN", f"批量检测结果入库失败: {e}")

    def closeEvent(self, event):
        try:
            self.stream_engine.stop()
            self.ng_saver.close()
        except Exception:
            pass
        self.controller.close()
        super().closeEvent(event)

    # ================= 边缘缩放 =================
    def _edge_dir(self, pos):
        r = self.rect()
        x, y = pos.x(), pos.y()
        m = _EDGE_MARGIN
        left, right = x <= m, x >= r.width() - m
        top, bottom = y <= m, y >= r.height() - m
        if top and left:
            return "top-left"
        if top and right:
            return "top-right"
        if bottom and left:
            return "bottom-left"
        if bottom and right:
            return "bottom-right"
        if left:
            return "left"
        if right:
            return "right"
        if top:
            return "top"
        if bottom:
            return "bottom"
        return None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and not self._maximized:
            d = self._edge_dir(event.pos())
            if d:
                self._resize_dir = d
                self._resize_start = event.globalPos()
                self._resize_rect = QRect(self.geometry())
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._resize_dir:
            delta = event.globalPos() - self._resize_start
            r = QRect(self._resize_rect)
            d = self._resize_dir
            mw, mh = self.minimumWidth(), self.minimumHeight()
            if "right" in d:
                r.setRight(max(r.left() + mw, r.right() + delta.x()))
            if "bottom" in d:
                r.setBottom(max(r.top() + mh, r.bottom() + delta.y()))
            if "left" in d:
                r.setLeft(min(r.right() - mw, r.left() + delta.x()))
            if "top" in d:
                r.setTop(min(r.bottom() - mh, r.top() + delta.y()))
            self.setGeometry(r)
            event.accept()
            return
        if not self._maximized:
            d = self._edge_dir(event.pos())
            cur = Qt.ArrowCursor
            if d:
                cur = {"left": Qt.SizeHorCursor, "right": Qt.SizeHorCursor,
                       "top": Qt.SizeVerCursor, "bottom": Qt.SizeVerCursor,
                       "top-left": Qt.SizeFDiagCursor,
                       "bottom-right": Qt.SizeFDiagCursor,
                       "top-right": Qt.SizeBDiagCursor,
                       "bottom-left": Qt.SizeBDiagCursor}[d]
            self.setCursor(cur)

    def mouseReleaseEvent(self, event):
        if self._resize_dir:
            self._resize_dir = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _toggle_maximize(self):
        if self._maximized:
            self.showNormal()
            self._maximized = False
        else:
            self.showMaximized()
            self._maximized = True


def _deep_default() -> dict:
    import json
    return json.loads(json.dumps(cfg_mod.DEFAULT_CONFIG))


def main():
    # 单实例 + 双击激活（2026-08-28）：autostart(HDMI:0) 与桌面图标(VNC:1) 并存
    # 曾出现双 GUI 实例 → 各自弹一次「连接成功」+ 两个窗口。
    # 方案：TCP 回环端口独占（127.0.0.1:43129）。
    #   - 端口 bind 成功 = 本实例是主实例（唯一）；退出/崩溃后 OS 自动释放端口
    #   - 端口 bind 失败 = 已有实例 → connect 发 activate → 已有窗口恢复（最小化则弹出）
    #   - 监听 socket 关闭无 TIME_WAIT → 用户关闭后立即双击可正常开新实例
    ctrl = _try_become_primary()
    if ctrl is None:
        # 已有实例：请求其恢复窗口，然后本实例退出。
        # 2026-09-01 修复：不再静默退出——给出可见提示，避免用户
        # 误以为「双击没反应 / 黑屏」（旧版双实例并存时的典型观感）。
        _activate_existing_instance()
        app = QApplication(sys.argv)
        _popup_info(
            None, "已有一个上位机窗口",
            "检测到上位机已在运行（HDMI 或 VNC 任一显示）。\n\n"
            "已激活其窗口，请切换到该窗口查看。\n"
            "同一时间只运行一个上位机实例，避免重复连接推理服务。")
        return 0

    app = QApplication(sys.argv)
    app.setFont(QFont("Noto Sans CJK SC", 10))
    qss = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "assets", "qss", "dark_theme.qss")
    if os.path.exists(qss):
        with open(qss, "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())
    win = MainWindow()
    win.showMaximized()
    ctrl.set_callback(_restore_window_cb(win))
    rc = app.exec_()
    try:
        ctrl.stop()
    except Exception:
        pass
    sys.exit(rc)


# ---------- 单实例 + 双击激活（TCP 回环端口独占，按 DISPLAY 区分） ----------
# 每个显示各自独立实例：HDMI(:0)=43129，VNC(:1)=43130，依次类推。
# 解决：:0 已有 GUI 时，:1 上双击也能开自己的 GUI（原全局单端口导致 VNC 上"打不开"）。
_BASE_ACTIVATE_PORT = 43129   # 专用端口（与Nano推理服务 8888 无关）


def _activate_port() -> int:
    """全局单实例（2026-09-01 修复）：固定返回 _BASE_ACTIVATE_PORT。
    原按 DISPLAY 分端口（:0→43129, :1→43130）导致 HDMI 与 VNC 双实例并存：
    两个 GUI 同时轮询 8888、各自触发相机/状态操作，叠加服务端事件循环
    冻结后表现为「检测预览图黑屏 / 连接失败」。现在无论从哪个显示启动，
    都抢占同一个回环端口，后启动者激活已有窗口后退出，保证任意时刻
    只有一个上位机实例。"""
    return _BASE_ACTIVATE_PORT


def _try_become_primary():
    """尝试独占激活端口成为主实例（端口随 DISPLAY 变化）。
    成功 → 返回控制对象（后台线程监听 activate）；
    失败（端口被占用 = 该显示已有实例）→ 返回 None。"""
    import socket
    import threading
    from PyQt5.QtCore import QObject, pyqtSignal

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # 注意：不能设 SO_REUSEADDR —— 否则 Windows 下可双 bind 导致检测失效
        srv.bind(("127.0.0.1", _activate_port()))
        srv.listen(4)
        srv.settimeout(0.5)
    except OSError as e:
        try:
            srv.close()
        except Exception:
            pass
        print(f"[single-instance] 端口 {_activate_port()} 被占用，判定已有实例: {e}",
              file=sys.stderr)
        return None

    # 跨线程投递：后台线程 emit 信号 → 主线程槽（Qt 自动排队，线程安全）
    class _Bridge(QObject):
        activated = pyqtSignal()

    bridge = _Bridge()
    _cb = {"fn": None}
    bridge.activated.connect(lambda: (_cb["fn"] or (lambda: None))())

    _stop = threading.Event()

    def _loop():
        while not _stop.is_set():
            try:
                conn, _ = srv.accept()
                try:
                    data = conn.recv(64)
                    if b"activate" in data:
                        bridge.activated.emit()
                finally:
                    try:
                        conn.close()
                    except Exception:
                        pass
            except socket.timeout:
                continue
            except OSError:
                break

    t = threading.Thread(target=_loop, daemon=True, name="gui-activate")
    t.start()

    class _Ctrl:
        @staticmethod
        def set_callback(fn):
            _cb["fn"] = fn

        @staticmethod
        def stop():
            _stop.set()
            try:
                srv.close()
            except Exception:
                pass

    ctrl = _Ctrl()
    ctrl._bridge = bridge   # 防 GC：bridge 必须存活到 stop()
    return ctrl


def _activate_existing_instance() -> bool:
    """连接主实例激活端口（同 DISPLAY），发送 activate 请求其恢复窗口。"""
    import socket
    try:
        s = socket.create_connection(("127.0.0.1", _activate_port()), timeout=0.6)
        try:
            s.sendall(b"activate")
            s.close()
            return True
        except OSError:
            try:
                s.close()
            except Exception:
                pass
            return False
    except OSError:
        return False


def _restore_window_cb(win):
    """返回激活回调：恢复/弹出主窗口（最小化则还原，置顶激活）。"""
    def _restore():
        try:
            if win.isMinimized():
                win.showNormal()
            win.showMaximized()
            win.raise_()
            win.activateWindow()
        except Exception:
            pass
    return _restore


def _crash_log(tb: str):
    """把异常写入 data/crash.log（pythonw 启动无控制台，用于排错）"""
    try:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "crash.log"), "a", encoding="utf-8") as f:
            f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {tb}\n")
    except Exception:
        pass


def _install_excepthook():
    """全局未捕获异常 → 写 crash.log（Qt 槽内异常默认只打到 stderr，pythonw 下不可见）"""
    def hook(exc_type, exc_val, exc_tb):
        import traceback as _tb
        text = "".join(_tb.format_exception(exc_type, exc_val, exc_tb))
        try:
            sys.stderr.write(text)
        except Exception:
            pass
        _crash_log(text)
    sys.excepthook = hook


if __name__ == "__main__":
    _install_excepthook()
    # 2026-09-01：faulthandler 捕获 SIGABRT/SIGSEGV（Qt/C++ 层崩溃）时转储 Python 调用栈
    try:
        import faulthandler
        _fh_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "data", "faulthandler.log")
        os.makedirs(os.path.dirname(_fh_path), exist_ok=True)
        faulthandler.enable(open(_fh_path, "a", buffering=1))
    except Exception:
        pass
    try:
        main()
    except Exception:
        traceback.print_exc()
        tb = traceback.format_exc()
        _crash_log(tb)
        QMessageBox.critical(None, "启动失败", f"程序启动失败：\n{tb}")
