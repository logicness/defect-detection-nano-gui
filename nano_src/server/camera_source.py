# -*- coding: utf-8 -*-
"""
下位机真实相机帧源封装（相机接入规划 P1-2，2026-08-28 真实版）
================================================================
- NanoCameraSource：open/read/close + 触发/曝光/增益参数（MVS Python 绑定）
- 自动对焦/图像优化功能
- 无 MVS SDK / 无相机时安全降级：open() 返回 False，read() 返回 (False, None)，不崩溃

MVS Python 绑定要点（实测 2026-08-28，SDK V5.0.2 aarch64 / 4.8.1 库）：
  - 环境变量 MVCAM_COMMON_RUNENV 必须指向含 aarch64/libMvCameraControl.so 的目录
    （默认 /opt/MVS/lib；绑定源码 os.getenv() + "/aarch64/..."，未设则 TypeError）
  - MvImport 包位于 /opt/MVS/Samples/aarch64/Python
  - 取流用 MV_CC_GetOneFrameTimeout（预分配缓冲 + MV_FRAME_OUT_INFO_EX）
  - 彩色相机数据为 Bayer/Mono/RGB，按 enPixelType 转 BGR numpy

设计目标（与产线流链路对接）：相机 → read() → 检测 → stream_frame（协议不变），
P2 阶段替换 camera_sim 的图片取帧部分。
"""
import ctypes
import logging
import os
import sys

log = logging.getLogger("camera_source")

MVS_PY_DIR = "/opt/MVS/Samples/aarch64/Python"
MVS_LIB_DIR = "/opt/MVS/lib"
LIB_ENV = "MVCAM_COMMON_RUNENV"
MAX_FRAME_BYTES = 32 * 1024 * 1024  # 32MB 缓冲（2448x2048 Bayer ≈ 5MB，留足余量）
OK = 0


def _load_mvs():
    """加载 MVS Python 绑定 → (MvCameraControl_class 模块, PixelType_header 模块)；失败 (None,None)"""
    try:
        os.environ.setdefault(LIB_ENV, MVS_LIB_DIR)
        if MVS_PY_DIR not in sys.path:
            sys.path.insert(0, MVS_PY_DIR)
        from MvImport import MvCameraControl_class as moc
        from MvImport import PixelType_header as pth
        return moc, pth
    except Exception as e:
        log.warning("MVS 绑定加载失败: %s", e)
        return None, None


class NanoCameraSource:
    """海康 MV-CS050-10UC（USB3.0）实时取流帧源。

    接口与上位机 core/frame_source.FrameSource 语义对齐：
      open() -> bool         打开相机并开始取流
      read() -> (ok, frame)  读一帧 BGR numpy（2448x2048 或 ROI）
      close()                停止取流并释放
    """

    def __init__(self, exposure_us: float = 22000.0, gain: float = 15.0,
                 trigger_mode: str = "continuous", roi=None, frame_rate: float = 10.0):
        self.exposure_us = exposure_us      # 曝光 us，默认 15ms（32 FPS，亮度合适）
        self.gain = gain                    # 增益 dB，默认 9dB
        self.trigger_mode = trigger_mode    # continuous / hardware / software
        self.roi = roi                      # dict(x,y,w,h) 或 None（全幅）
        self.frame_rate = float(frame_rate)  # 采集帧率上限（与消费速率匹配，防 SDK 缓冲池占满停摆）
        self._opened = False
        self._cam = None                    # MvCamera 实例
        self._moc = None                    # 绑定模块引用
        self._pth = None
        self._last_error = ""
        # 2026-09-01 17:2x 重写加固：预分配取流缓冲（只分配一次），
        # 消除每次读帧新建 32MB ctypes 缓冲导致的 320MB/s 分配抖动
        # （此前 10FPS 连续采集 ~20s 后 MVS 停摆的直接诱因之一）。
        self._buf = None

    # ---------------- 生命周期 ----------------
    def open(self) -> bool:
        """初始化 SDK → 枚举 → 打开 → 设置参数 → 开始取流。失败返回 False。"""
        if self._opened:
            return True
        self._moc, self._pth = _load_mvs()
        if not self._moc:
            self._last_error = "MVS SDK 绑定不可用（检查 MVCAM_COMMON_RUNENV 与 /opt/MVS/Samples/aarch64/Python）"
            log.warning("NanoCameraSource.open: %s", self._last_error)
            return False
        moc = self._moc
        try:
            # 0) 初始化 SDK（官方示例要求必须调用）
            ret = moc.MvCamera.MV_CC_Initialize()
            if ret != OK:
                log.warning("MV_CC_Initialize 返回 0x%08X，继续尝试", ret)

            # 1) 枚举（USB3 优先，其次 GigE）
            dev_list = moc.MV_CC_DEVICE_INFO_LIST()
            found = False
            for layer in (moc.MV_USB_DEVICE, moc.MV_GIGE_DEVICE):
                if moc.MvCamera.MV_CC_EnumDevices(layer, dev_list) == OK and dev_list.nDeviceNum > 0:
                    found = True
                    break
            if not found:
                self._last_error = "未枚举到相机（检查 USB3.0 连接 / 原装线缆 / lsusb）"
                log.warning("NanoCameraSource.open: %s", self._last_error)
                return False

            # 2) 创建句柄并打开（第一台，使用 cast 正确获取设备信息）
            cam = moc.MvCamera()
            stDeviceList = ctypes.cast(dev_list.pDeviceInfo[0], ctypes.POINTER(moc.MV_CC_DEVICE_INFO)).contents
            if cam.MV_CC_CreateHandle(stDeviceList) != OK:
                self._last_error = "MV_CC_CreateHandle 失败"
                return False
            if cam.MV_CC_OpenDevice(moc.MV_ACCESS_Exclusive, 0) != OK:
                self._last_error = "MV_CC_OpenDevice 失败（设备被占用/权限不足）"
                cam.MV_CC_DestroyHandle()
                return False

            # 3) 参数：触发模式 / 曝光 / 增益 / 采集模式
            cam.MV_CC_SetEnumValue("TriggerMode",
                                   moc.MV_TRIGGER_MODE_OFF if self.trigger_mode == "continuous"
                                   else moc.MV_TRIGGER_MODE_ON)
            # 2026-09-01 17:2x 重写加固：显式连续采集模式（默认可能为单帧/受限模式，
            # 导致连续取流一段时间后 MVS 内部停摆）。
            try:
                cam.MV_CC_SetEnumValue("AcquisitionMode",
                                       getattr(moc, "MV_ACQ_MODE_CONTINUOUS", 2))
            except Exception:
                pass
            # 2026-09-01 17:2x 重写加固：限制采集帧率与消费速率匹配。
            # 若不设，相机按最大帧率（~32FPS）全速出流，而消费端仅 ~10FPS 读，
            # MVS SDK 取流缓冲池被未读帧占满 → GetOneFrameTimeout 无限等空缓冲 →
            # 连续采集 ~20s 后读帧挂死停摆（实测根因）。
            try:
                cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
                cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(self.frame_rate))
            except Exception as e:
                log.warning("设置采集帧率失败（忽略）: %s", e)
            if self.exposure_us and self.exposure_us > 0:
                cam.MV_CC_SetFloatValue("ExposureTime", self.exposure_us)
            if self.gain and self.gain > 0:
                cam.MV_CC_SetFloatValue("Gain", self.gain)
            # 2026-09-03 修复偏色：开启自动白平衡（彩色相机 MV-CS050-10UC 此前从未设置
            # 白平衡 → 白色物体画面偏黄）。BalanceWhiteAuto: 0=Off 1=Once 2=Continuous。
            # Once=单次校准后锁定（固定光源产线推荐）；失败/黑白相机忽略不阻塞开流。
            try:
                cam.MV_CC_SetEnumValue("BalanceWhiteAuto", 2)
                log.info("已开启自动白平衡 BalanceWhiteAuto=Continuous")
                # 2026-09-03 P2（手册 §7.12 建议）：Narrow 色温模式下自动白平衡后色彩仍
                # 有偏差时，将 AWB Color Temperature Mode 放宽为 Wide 再校准（扩大色温
                # 校正范围）。失败/无此节点忽略，不阻塞开流。
                try:
                    cam.MV_CC_SetEnumValue("AWB Color Temperature Mode", 1)  # 1=Wide
                    log.info("已设置 AWB 色温模式 = Wide")
                except Exception as e2:
                    log.warning("设置 AWB 色温模式 Wide 失败（忽略）: %s", e2)
            except Exception as e:
                log.warning("设置自动白平衡失败（忽略，可能是黑白相机/无此节点）: %s", e)

            # 4) 预分配取流缓冲（一次性，避免每帧 32MB 分配抖动）
            self._buf = (ctypes.c_ubyte * MAX_FRAME_BYTES)()

            # 5) 开始取流
            if cam.MV_CC_StartGrabbing() != OK:
                self._last_error = "MV_CC_StartGrabbing 失败"
                cam.MV_CC_CloseDevice()
                cam.MV_CC_DestroyHandle()
                return False

            self._cam = cam
            self._opened = True
            self._last_error = ""
            log.info("相机已打开: exposure_us=%s gain=%s trigger=%s",
                     self.exposure_us, self.gain, self.trigger_mode)
            return True
        except Exception as e:
            self._last_error = f"打开相机异常: {e}"
            log.error("NanoCameraSource.open: %s", self._last_error)
            self._opened = False
            return False

    def read(self):
        """读一帧 BGR。未打开/取流失败返回 (False, None)。
        2026-09-01 17:2x 重写加固：复用 open() 预分配的 _buf（仅分配一次），
        消除每帧 32MB 分配抖动——此前的分配风暴是 10FPS 连续采集停摆诱因。"""
        if not self._opened or self._cam is None:
            return False, None
        try:
            cam = self._cam
            if self._buf is None:
                self._buf = (ctypes.c_ubyte * MAX_FRAME_BYTES)()
            st = self._moc.MV_FRAME_OUT_INFO_EX()
            ret = cam.MV_CC_GetOneFrameTimeout(self._buf, MAX_FRAME_BYTES, st, 1000)
            if ret != OK:
                self._last_error = f"取帧失败 ret={ret}"
                return False, None
            w, h = int(st.nWidth), int(st.nHeight)
            if w <= 0 or h <= 0:
                self._last_error = "帧尺寸无效"
                return False, None
            nbytes = max(w * h * 3, 1)
            raw = ctypes.string_at(ctypes.cast(self._buf, ctypes.c_void_p), nbytes)
            frame = self._convert(raw, w, h, int(st.enPixelType))
            if frame is None:
                self._last_error = f"不支持的像素格式 {st.enPixelType}"
                return False, None
            if self.roi:
                x, y, rw, rh = self.roi["x"], self.roi["y"], self.roi["w"], self.roi["h"]
                frame = frame[y:y + rh, x:x + rw]
            return True, frame
        except Exception as e:
            self._last_error = f"取流异常: {e}"
            log.error("NanoCameraSource.read: %s", self._last_error)
            return False, None

    def close(self):
        """停止取流并释放资源（幂等）。"""
        if not self._opened and self._cam is None:
            return
        try:
            if self._cam is not None:
                self._cam.MV_CC_StopGrabbing()
                self._cam.MV_CC_CloseDevice()
                self._cam.MV_CC_DestroyHandle()
                self._cam = None
            # 反初始化 SDK（与 MV_CC_Initialize 配对）
            if self._moc is not None:
                try:
                    self._moc.MvCamera.MV_CC_Finalize()
                except Exception:
                    pass
        except Exception as e:
            log.warning("NanoCameraSource.close: %s", e)
        finally:
            self._opened = False
            self._buf = None  # 释放预分配缓冲

    # ---------------- 参数 ----------------
    def set_params(self, exposure_us=None, gain=None, trigger_mode=None):
        """运行时更新参数并同步到相机。返回 (ok, error, applied)。
        applied = 读回的实际生效值 dict；未打开/下发失败返回 None。
        2026-09-01 完善：未打开明确失败；自动曝光/增益先关闭（确保手动值生效）；
        下发后读回确认；MVS 错误不再静默吞掉。"""
        if exposure_us is not None:
            self.exposure_us = exposure_us
        if gain is not None:
            self.gain = gain
        if trigger_mode is not None:
            self.trigger_mode = trigger_mode
        if not self._opened or self._cam is None:
            return False, "相机未打开（请先启动产线流再下发参数）", None
        cam = self._cam
        try:
            # 0) 关闭自动曝光/自动增益（若相机支持），确保手动值生效
            for key in ("ExposureAuto", "GainAuto"):
                try:
                    ev = self._moc.MVCC_ENUMVALUE()
                    if cam.MV_CC_GetEnumValue(key, ev) == OK \
                            and getattr(ev, "nCurValue", 0) != 0:
                        cam.MV_CC_SetEnumValue(key, 0)
                except Exception:
                    pass
            # 1) 触发模式
            if trigger_mode is not None:
                cam.MV_CC_SetEnumValue(
                    "TriggerMode",
                    self._moc.MV_TRIGGER_MODE_OFF if self.trigger_mode == "continuous"
                    else self._moc.MV_TRIGGER_MODE_ON)
            # 2) 曝光 / 增益（超范围 MVS 会返回错误码，不再静默）
            if exposure_us is not None and exposure_us > 0:
                ret = cam.MV_CC_SetFloatValue("ExposureTime", exposure_us)
                if ret != OK:
                    return False, f"设置曝光失败 (MVS 0x{ret:08X})", None
            if gain is not None and gain > 0:
                ret = cam.MV_CC_SetFloatValue("Gain", gain)
                if ret != OK:
                    return False, f"设置增益失败 (MVS 0x{ret:08X})", None
            # 3) 读回确认实际生效值（相机可能钳位到合法范围）
            applied = {}
            fv = self._moc.MVCC_FLOATVALUE()
            if cam.MV_CC_GetFloatValue("ExposureTime", fv) == OK:
                applied["exposure_us"] = round(float(fv.fCurValue), 1)
            if cam.MV_CC_GetFloatValue("Gain", fv) == OK:
                applied["gain"] = round(float(fv.fCurValue), 2)
            applied["trigger_mode"] = self.trigger_mode
            if "exposure_us" in applied:
                self.exposure_us = applied["exposure_us"]
            if "gain" in applied:
                self.gain = applied["gain"]
            return True, None, applied
        except Exception as e:
            return False, f"相机参数下发异常: {e}", None

    def query_params(self):
        """读回相机当前参数（GUI 展示用）。未打开返回 None。"""
        if not self._opened or self._cam is None:
            return None
        try:
            fv = self._moc.MVCC_FLOATVALUE()
            out = {"exposure_us": None, "gain": None,
                   "auto_exposure": None, "auto_gain": None}
            if self._cam.MV_CC_GetFloatValue("ExposureTime", fv) == OK:
                out["exposure_us"] = round(float(fv.fCurValue), 1)
            if self._cam.MV_CC_GetFloatValue("Gain", fv) == OK:
                out["gain"] = round(float(fv.fCurValue), 2)
            for key, slot in (("ExposureAuto", "auto_exposure"),
                              ("GainAuto", "auto_gain")):
                try:
                    ev = self._moc.MVCC_ENUMVALUE()
                    if self._cam.MV_CC_GetEnumValue(key, ev) == OK:
                        out[slot] = int(getattr(ev, "nCurValue", -1))
                except Exception:
                    pass
            return out
        except Exception as e:
            log.warning(f"查询相机参数失败: {e}")
            return None

    def device_summary(self) -> dict:
        """健康诊断摘要（上位机调试台展示用）"""
        return {
            "opened": self._opened,
            "exposure_us": self.exposure_us,
            "gain": self.gain,
            "trigger_mode": self.trigger_mode,
            "sdk_ready": _load_mvs()[0] is not None,
            "last_error": self._last_error,
        }

    @property
    def is_open(self) -> bool:
        """相机是否已打开（供 camera_sim.camera_online / auto_exposure 判断）。"""
        return bool(self._opened and self._cam is not None)

    def get_temperature(self):
        """读取相机设备温度（°C）。
        MV_CC_GetFloatValue("DeviceTemperature")；不支持/失败返回 None（GUI 显示 --）。
        2026-09-01 新增：状态栏展示用，约 5s 轮询一次，开销可忽略。
        """
        if not self._opened or self._cam is None:
            return None
        try:
            fv_cls = (getattr(self._moc, "MVCC_FLOATVALUE", None)
                      or getattr(self._moc, "MV_CC_FLOAT_VALUE", None))
            if fv_cls is None:
                return None
            st = fv_cls()
            ret = self._cam.MV_CC_GetFloatValue("DeviceTemperature", st)
            if ret == OK and st.fCurValue and st.fCurValue > 0:
                return round(float(st.fCurValue), 1)
        except Exception as e:
            log.debug(f"读取相机温度失败: {e}")
        return None

    # ---------------- 自动对焦/图像优化 ----------------
    def analyze_image(self, image):
        """分析图像质量"""
        try:
            from auto_focus import ImageQualityEvaluator
            evaluator = ImageQualityEvaluator()
            return evaluator.evaluate(image)
        except ImportError:
            log.warning("auto_focus 模块不可用")
            return None

    def auto_optimize_exposure(self, target_brightness=120.0, tolerance=20.0, max_iterations=5):
        """自动优化曝光参数
        
        Args:
            target_brightness: 目标亮度（0-255）
            tolerance: 容差范围
            max_iterations: 最大迭代次数
            
        Returns:
            (最优曝光, 最优增益, 图像质量分析)
        """
        if not self._opened:
            log.warning("相机未打开，无法自动优化")
            return self.exposure_us, self.gain, None
        
        try:
            from auto_focus import AutoExposureOptimizer
            optimizer = AutoExposureOptimizer(
                target_brightness=target_brightness,
                tolerance=tolerance,
                max_iterations=max_iterations
            )
            
            # 优化曝光
            best_exposure, best_gain = optimizer.optimize(
                self, self.exposure_us, self.gain
            )
            
            # 应用最优参数
            self.set_params(exposure_us=best_exposure, gain=best_gain)
            
            # 读取优化后的图像进行分析
            ret, frame = self.read()
            if ret and frame is not None:
                analysis = self.analyze_image(frame)
                return best_exposure, best_gain, analysis
            
            return best_exposure, best_gain, None
            
        except Exception as e:
            log.error(f"自动优化失败: {e}")
            return self.exposure_us, self.gain, None

    def enhance_image(self, image, sharpen=True, contrast=True, denoise=True):
        """增强图像质量"""
        try:
            from auto_focus import ImageEnhancer
            enhancer = ImageEnhancer()
            return enhancer.enhance(image, sharpen, contrast, denoise)
        except ImportError:
            log.warning("auto_focus 模块不可用，返回原图")
            return image

    def get_focus_suggestions(self, image):
        """获取对焦建议"""
        try:
            from auto_focus import AutoFocusManager
            manager = AutoFocusManager(self)
            return manager.analyze_image(image)
        except ImportError:
            log.warning("auto_focus 模块不可用")
            return None

    # ---------------- 内部 ----------------
    def _convert(self, raw, w, h, pix):
        """按像素格式转 BGR numpy；不支持返回 None。"""
        import cv2
        import numpy as np
        pth = self._pth
        V = lambda name, default: int(getattr(pth, name, default))
        MONO8 = V("MV_GVSP_PIX_MONO8", 0x01080001)
        BAYER_RG8 = V("MV_GVSP_PIX_BAYER_RG8", 0x01080009)
        BAYER_GR8 = V("MV_GVSP_PIX_BAYER_GR8", 0x0108000A)
        BAYER_GB8 = V("MV_GVSP_PIX_BAYER_GB8", 0x0108000B)
        BAYER_BG8 = V("MV_GVSP_PIX_BAYER_BG8", 0x01080008)
        RGB8 = V("MV_GVSP_PIX_RGB8", 0x02180014)
        BGR8 = V("MV_GVSP_PIX_BGR8", 0x02180015)
        if pix == MONO8:
            return np.frombuffer(raw, np.uint8).reshape(h, w)[:, :, None].repeat(3, 2)
        if pix in (BAYER_RG8, BAYER_GR8, BAYER_GB8, BAYER_BG8):
            bayer = np.frombuffer(raw[:w * h], np.uint8).reshape(h, w)
            # 2026-09-03 修复色偏（蓝线变黄/赭）：OpenCV 的 Bayer 命名按"第二行第2/3列"定义，
            # 与经典 Bayer（海康 PixelType 命名）相反。实测相机输出 BayerRG8（经典 RGGB），
            # 必须用 COLOR_BayerBG2BGR（OpenCV 文档标注等价于 RGGB）才能得到正确颜色；
            # 原 RG2BGR（等价 BGGR）导致 R/B 互换 → 蓝色显示为黄/红。映射如下（成对互换）：
            code = {BAYER_RG8: cv2.COLOR_BayerBG2BGR, BAYER_GR8: cv2.COLOR_BayerGB2BGR,
                    BAYER_GB8: cv2.COLOR_BayerGR2BGR, BAYER_BG8: cv2.COLOR_BayerRG2BGR}.get(pix)
            return cv2.cvtColor(bayer, code) if code is not None else None
        if pix == RGB8:
            return cv2.cvtColor(np.frombuffer(raw[:w * h * 3], np.uint8).reshape(h, w, 3),
                                cv2.COLOR_RGB2BGR)
        if pix == BGR8:
            return np.frombuffer(raw[:w * h * 3], np.uint8).reshape(h, w, 3).copy()
        log.warning("未支持像素格式 pix=%s", pix)
        return None


if __name__ == "__main__":
    # 自检：无相机时安全降级；有相机时 open 成功
    logging.basicConfig(level=logging.INFO)
    src = NanoCameraSource()
    ok = src.open()
    print(f"open() -> {ok}", src.device_summary())
    if ok:
        ok2, frame = src.read()
        print(f"read() -> {ok2}", None if frame is None else frame.shape)
        src.close()