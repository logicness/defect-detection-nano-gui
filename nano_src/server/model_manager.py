#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型仓库管理器（阶段 A：模型管理功能）
- 扫描 models/**/*.engine 建立模型清单
- 热切换模型（加载失败自动回滚旧引擎，状态写入 models.json 持久化）
- 分片接收上位机上传（.engine/.onnx/.pt）
- .onnx / .pt 后台自动编译为 engine（trtexec / yolo export）
- 删除模型（激活模型禁止删除）

配套 infer_server.py 使用。
"""
import os
import sys
import re
import json
import time
import gc
import base64
import hashlib
import logging
import subprocess
import threading

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "inference"))
from trt_infer import TRTInfer

log = logging.getLogger("model_mgr")

MODELS_DIR = "/home/nvidia/defect_detection/models"
UPLOAD_DIR = os.path.join(MODELS_DIR, "uploads")
COMPILE_DIR = os.path.join(MODELS_DIR, "compiled")
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models.json")
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"
INPUT_SIZE = 640
ALLOWED_EXT = (".engine", ".onnx", ".pt")
MAX_UPLOAD_SIZE = 2 * 1024 * 1024 * 1024  # 上传大小上限 2GB（防磁盘填满）


def _safe_basename(filename):
    """只保留文件名（拒绝绝对路径/../ 穿越），供上传文件名校验"""
    name = os.path.basename(str(filename).replace("\\", "/"))
    if not name or name in (".", "..") or ".." in name.split("/"):
        return None
    return name


def _safe_repo_name(name):
    """校验仓库名：允许子目录（neu/xxx.engine），拒绝绝对路径与 .. 穿越"""
    norm = str(name or "").strip().replace("\\", "/")
    if not norm or norm.startswith("/"):
        return None
    segs = norm.split("/")
    if any(seg in ("..", ".") for seg in segs) or any(not seg for seg in segs):
        return None
    return norm


def _rel_name(abs_path):
    """绝对路径 -> 相对 MODELS_DIR 的仓库名（/ 分隔）"""
    try:
        return os.path.relpath(os.path.abspath(abs_path), MODELS_DIR).replace("\\", "/")
    except ValueError:
        return os.path.abspath(abs_path)


def _load_names(engine_path):
    """读取 engine 同名 sidecar 类别表（.names，每行一个类名，# 开头为注释）。
    TRT engine 不携带类别名，缺省返回 [] → 标注回退显示 clsN。"""
    p = os.path.splitext(engine_path)[0] + ".names"
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            names = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
        return names
    except OSError:
        return []


def _parse_rois(raw):
    """校验/清洗 ROI 列表：[{x,y,w,h}(,enabled)] -> 仅保留启用的合法矩形"""
    out = []
    if not isinstance(raw, (list, tuple)):
        return out
    for r in raw:
        if not isinstance(r, dict):
            continue
        if not r.get("enabled", True):
            continue
        try:
            x, y, w, h = int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])
        except (KeyError, TypeError, ValueError):
            continue
        if w <= 0 or h <= 0:
            continue
        out.append({"x": x, "y": y, "w": w, "h": h})
    return out


def _box_in_rois(box, rois):
    """框中心点落在任一启用 ROI 内则保留（无 ROI 时调用方不会进入本函数）"""
    try:
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
    except (TypeError, ValueError, IndexError):
        return False
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    for r in rois:
        if r["x"] <= cx <= r["x"] + r["w"] and r["y"] <= cy <= r["y"] + r["h"]:
            return True
    return False


class EngineHandle:
    """推理引擎句柄：封装 TRTInfer，热切换时整体替换实例。"""

    def __init__(self, engine_path, conf_thres=0.25, iou_thres=0.45):
        log.info(f"加载 engine: {engine_path}")
        self.engine_path = engine_path
        self.infer = TRTInfer(engine_path, conf_thres=conf_thres, iou_thres=iou_thres)
        self.labels = _load_names(engine_path)   # sidecar .names 类别表（缺省 [] → 显示 clsN）
        self.rois = []                            # ROI 检测门控（框中心不在任一启用 ROI 内则丢弃）
        if self.labels:
            log.info(f"engine 加载完成（类别表 {len(self.labels)} 类）")
        else:
            log.info("engine 加载完成（无 .names 类别表，标注显示 clsN）")
        # 首帧预热：TensorRT 引擎首次推理会编译 CUDA kernel（首帧延迟可达数百 ms），
        # 提前用一张空图跑一次，避免产线启动/模型热切换后第一帧超时
        try:
            import numpy as np
            t0 = time.time()
            _h, _w = self.infer.input_shape[2], self.infer.input_shape[3]
            self.detect(np.zeros((_h, _w, 3), dtype=np.uint8))
            log.info(f"预热完成（首帧 CUDA kernel 已编译, {1000*(time.time()-t0):.0f}ms）")
        except Exception as e:
            log.warning(f"预热失败（忽略,不影响服务）: {e}")

    def detect(self, img_bgr):
        result = self.infer.infer(img_bgr)
        rois = self.rois
        if rois:
            dets = result.get("detections") or []
            kept = [d for d in dets if _box_in_rois(d.get("box"), rois)]
            dropped = len(dets) - len(kept)
            if dropped:
                result["detections"] = kept
                result["roi_dropped"] = dropped
        return result

    def update_rois(self, rois):
        self.rois = _parse_rois(rois)

    def update_params(self, conf_thres=None, iou_thres=None):
        if conf_thres is not None:
            self.infer.conf_thres = conf_thres
        if iou_thres is not None:
            self.infer.iou_thres = iou_thres

    @property
    def conf_thres(self):
        return self.infer.conf_thres

    @property
    def iou_thres(self):
        return self.infer.iou_thres

    def close(self):
        try:
            del self.infer
        except Exception:
            pass


class ModelManager:
    """模型仓库管理：扫描 / 热切换 / 上传 / 编译 / 删除"""

    def __init__(self, engine_path, conf_thres=0.25, iou_thres=0.45):
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        os.makedirs(COMPILE_DIR, exist_ok=True)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self._compiling = False
        self._state = self._load_state()
        self.current = self._pick_start_engine(engine_path)
        self._set_active(_rel_name(self.current))
        self.engine = EngineHandle(self.current, conf_thres, iou_thres)
        self.set_rois(self._state.get("rois") or [], persist=False)  # 恢复持久化的 ROI 门控
        self._upload = None
        log.info(f"当前模型: {self.current}")

    # ---------- 状态持久化 ----------
    def _load_state(self):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _set_active(self, name):
        self._state["active"] = name
        try:
            with open(STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(self._state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.warning(f"写入状态文件失败: {e}")

    def set_rois(self, rois, persist=True):
        """更新检测 ROI 门控：框中心须落在任一启用 ROI 内，否则丢弃。
        persist=True 时写入 models.json，服务重启后自动恢复。"""
        valid = _parse_rois(rois)
        self.engine.rois = valid
        if persist:
            self._state["rois"] = valid
            try:
                with open(STATE_FILE, "w", encoding="utf-8") as f:
                    json.dump(self._state, f, ensure_ascii=False, indent=2)
            except Exception as e:
                log.warning(f"写入状态文件失败: {e}")
        log.info(f"ROI 门控更新: {len(valid)} 个启用")

    def _pick_start_engine(self, engine_path):
        """启动模型选择：显式 --engine > models.json 记录 > 扫描第一个"""
        if engine_path and os.path.exists(engine_path):
            return os.path.abspath(engine_path)
        active = self._state.get("active")
        if active:
            p = os.path.join(MODELS_DIR, active)
            if os.path.exists(p):
                return os.path.abspath(p)
        models = self._scan()
        if models:
            return models[0]["path"]
        raise FileNotFoundError("models 目录下没有找到任何 .engine")

    # ---------- 扫描 ----------
    def _scan(self):
        out = []
        for root, dirs, files in os.walk(MODELS_DIR):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for f in files:
                if f.endswith(".engine"):
                    p = os.path.join(root, f)
                    try:
                        st = os.stat(p)
                    except OSError:
                        continue
                    out.append({
                        "name": _rel_name(p),
                        "path": p,
                        "size_bytes": st.st_size,
                        "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
                    })
        return sorted(out, key=lambda x: x["name"].lower())

    def list_models(self):
        cur_abs = os.path.abspath(self.current)
        models = []
        for m in self._scan():
            m["active"] = os.path.abspath(m["path"]) == cur_abs
            models.append(m)
        return {"models": models, "active": _rel_name(cur_abs)}

    def _resolve(self, name):
        # 拒绝绝对路径与 ../ 穿越（8888 无鉴权，文件名来自网络）
        name = _safe_repo_name(name)
        if name is None:
            return None
        p = os.path.join(MODELS_DIR, name)
        return p if os.path.exists(p) else None

    # ---------- 热切换 ----------
    def load(self, name):
        """切换模型。成功替换 self.engine；失败保持旧引擎（自动回滚）。"""
        path = self._resolve(name)
        if not path:
            return {"ok": False, "error": f"模型不存在: {name}"}
        t0 = time.perf_counter()
        try:
            new_handle = EngineHandle(path, self.conf_thres, self.iou_thres)
            new_handle.rois = list(self.engine.rois)  # ROI 门控随热切换保留（类别表随新 engine 重载）
            # 预热已由 EngineHandle.__init__ 承担（启动/热切换统一覆盖）
        except Exception as e:
            log.error(f"加载模型失败 {name}: {e}")
            return {"ok": False, "error": f"加载失败: {e}"}
        load_ms = (time.perf_counter() - t0) * 1000
        old = self.engine
        self.engine = new_handle
        self.current = os.path.abspath(path)
        self._set_active(_rel_name(self.current))
        old.close()
        del old
        gc.collect()
        log.info(f"模型切换: {name} ({load_ms:.0f}ms)")
        return {"ok": True, "load_ms": round(load_ms, 1)}

    # ---------- 分片上传 ----------
    def upload_begin(self, filename, size):
        # 文件名仅取 basename，防路径穿越（../../x.engine / 绝对路径）
        filename = _safe_basename(filename)
        if filename is None:
            return None, "非法文件名（仅支持文件名，不支持路径）"
        if not filename.endswith(ALLOWED_EXT):
            return None, f"仅支持 {'/'.join(ALLOWED_EXT)} 文件"
        try:
            size = int(size)
        except (TypeError, ValueError):
            return None, "非法文件大小"
        if size <= 0 or size > MAX_UPLOAD_SIZE:
            return None, f"文件大小非法（0 < size <= {MAX_UPLOAD_SIZE}）"
        if self._upload is not None:
            self.upload_abort()
        tmp = os.path.join(UPLOAD_DIR, ".uploading_" + filename)
        try:
            fh = open(tmp, "wb")
        except OSError as e:
            return None, f"创建临时文件失败: {e}"
        self._upload = {
            "name": filename, "tmp": tmp, "size": size,
            "received": 0, "fh": fh,
            "sha": hashlib.sha256(), "seq": -1,
        }
        return tmp, None

    def upload_chunk(self, filename, seq, data_b64):
        up = self._upload
        if up is None or up["name"] != filename:
            return False, "上传会话不存在或文件名不匹配，请先发送 model_upload_start"
        if seq != up["seq"] + 1:
            return False, f"分片序号不连续: 期望 {up['seq']+1}, 收到 {seq}"
        try:
            chunk = base64.b64decode(data_b64)
        except Exception as e:
            return False, f"base64 解码失败: {e}"
        # 超限保护：拒绝超过声明大小的分片（防磁盘填满）
        if up["received"] + len(chunk) > up["size"]:
            self.upload_abort()
            return False, f"数据超过声明大小（上限 {up['size']}）"
        up["fh"].write(chunk)
        up["sha"].update(chunk)
        up["received"] += len(chunk)
        up["seq"] = seq
        return True, up["received"]

    def upload_commit(self, filename, checksum):
        up = self._upload
        if up is None or up["name"] != filename:
            return None, "上传会话不存在"
        try:
            up["fh"].close()
        except Exception:
            pass
        self._upload = None
        if up["received"] != up["size"]:
            self._cleanup(up["tmp"])
            return None, f"字节数不符: 期望 {up['size']}, 收到 {up['received']}"
        digest = up["sha"].hexdigest()
        if checksum and checksum.lower() != digest:
            self._cleanup(up["tmp"])
            return None, "sha256 校验失败"
        final = os.path.join(UPLOAD_DIR, filename)
        try:
            os.replace(up["tmp"], final)
        except OSError as e:
            self._cleanup(up["tmp"])
            return None, f"保存失败: {e}"
        log.info(f"上传完成: {final} ({up['received']} bytes)")
        return final, None

    def upload_abort(self):
        if self._upload:
            try:
                self._upload["fh"].close()
            except Exception:
                pass
            self._cleanup(self._upload["tmp"])
            self._upload = None

    @staticmethod
    def _cleanup(p):
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

    # ---------- 删除 ----------
    def delete(self, name):
        path = self._resolve(name)
        if not path:
            return {"ok": False, "error": f"模型不存在: {name}"}
        if os.path.abspath(path) == os.path.abspath(self.current):
            return {"ok": False, "error": "正在激活的模型禁止删除"}
        try:
            os.remove(path)
            log.info(f"已删除模型: {name}")
            return {"ok": True}
        except OSError as e:
            return {"ok": False, "error": f"删除失败: {e}"}

    # ---------- 编译 ----------
    @property
    def compiling(self):
        return self._compiling

    def compile_async(self, upload_path, progress_cb):
        """后台编译 .onnx/.pt -> engine（不自动加载，编译完成回调 stage='compiled'，
        由 server 层持锁 load，避免与 detect 并发）。
        progress_cb(stage, progress, message) 在子线程中调用。"""
        self._compiling = True
        ext = os.path.splitext(upload_path)[1].lower()
        base = os.path.splitext(os.path.basename(upload_path))[0]
        out_engine = os.path.join(COMPILE_DIR, base + "_fp16.engine")
        threading.Thread(
            target=self._compile_worker,
            args=(upload_path, ext, out_engine, progress_cb),
            daemon=True,
        ).start()

    def _compile_worker(self, upload_path, ext, out_engine, progress_cb):
        try:
            if ext == ".pt":
                progress_cb("export", 5, "yolo 导出 onnx 中")
                onnx_path = os.path.join(
                    COMPILE_DIR, os.path.splitext(os.path.basename(upload_path))[0] + ".onnx")
                cmd = [sys.executable, "-m", "ultralytics", "yolo", "export",
                       f"model={upload_path}", "format=onnx", "imgsz=640",
                       "opset=19", "simplify=True", "dynamic=True"]
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
                if proc.returncode != 0 or not os.path.exists(onnx_path):
                    tail = (proc.stderr or proc.stdout or "")[-500:]
                    raise RuntimeError(f"yolo export 失败: {tail}")
                progress_cb("compile", 15, "onnx 导出完成，trtexec 编译中")
            else:
                onnx_path = upload_path
                progress_cb("compile", 10, "trtexec 编译中")
            self._run_trtexec(onnx_path, out_engine, progress_cb)
            progress_cb("compiled", 95, f"编译完成: {_rel_name(out_engine)}")
        except Exception as e:
            log.error(f"编译失败: {e}")
            progress_cb("error", 0, str(e))
        finally:
            self._compiling = False

    def _run_trtexec(self, onnx_path, out_engine, progress_cb):
        cmd = [
            TRTEXEC, f"--onnx={onnx_path}", f"--saveEngine={out_engine}",
            "--fp16", "--memPoolSize=workspace:64",
            f"--minShapes=images:1x3x{INPUT_SIZE}x{INPUT_SIZE}",
            f"--optShapes=images:1x3x{INPUT_SIZE}x{INPUT_SIZE}",
            f"--maxShapes=images:1x3x{INPUT_SIZE}x{INPUT_SIZE}",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        last = 0
        stage_note = False
        t_start = time.time()
        # 编译超时保护：防止 trtexec 挂起导致 _compiling 永远 True、服务拒绝推理
        TRTEXEC_TIMEOUT_S = 900
        for line in proc.stdout:
            if time.time() - t_start > TRTEXEC_TIMEOUT_S:
                proc.kill()
                proc.wait()
                raise RuntimeError(f"trtexec 编译超时(>{TRTEXEC_TIMEOUT_S}s)")
            # 兼容输出百分比的 trtexec 版本
            m = re.search(r"(\d+)%\s*-\s*", line)
            if m:
                pct = int(m.group(1))
                if pct >= last + 5:
                    last = pct
                    progress_cb("compile", 15 + int(pct * 0.8), f"trtexec 编译 {pct}%")
            # 无百分比版本：按里程碑行推送阶段进度
            elif "Building engine" in line and not stage_note:
                stage_note = True
                progress_cb("compile", 25, "trtexec 构建引擎中（FP16，约 2~5 分钟）")
            elif "Engine built" in line or "PASSED" in line or "Succeeded" in line:
                progress_cb("compile", 92, "trtexec 引擎构建完成")
        proc.wait()
        if proc.returncode != 0 or not os.path.exists(out_engine):
            raise RuntimeError(f"trtexec 编译失败 (exit={proc.returncode})")
