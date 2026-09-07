#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下位机图片仓库管理（下位机图片检测功能，2026-08-20 新增）
- 固定图片文件夹：默认 /home/nvidia/defect_detection/images/input，启动自动创建
- 目录持久化：server/images.json（仿 models.json，重启恢复；优先级 --image-dir > images.json > 默认）
- 扫描图片清单（白名单扩展名 + 数量上限）
- 缩略图生成（JPEG base64，256px 列表用 / 640px 预览标注用）
- 文件名安全校验：只允许 basename，拒绝绝对路径与 ../ 穿越（8888 无鉴权，文件名来自网络）

配套 server/infer_server.py 使用。
"""
import os
import json
import base64
import logging
import time

import cv2
import numpy as np

log = logging.getLogger("image_store")

DEFAULT_IMAGE_DIR = "/home/nvidia/defect_detection/images/input"
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images.json")
ALLOWED_EXT = (".jpg", ".jpeg", ".png", ".bmp")
MAX_IMAGES = 500          # 单次列表上限（防一次回传过多缩略图）
THUMB_QUALITY = 75        # 列表小缩略图 JPEG 质量
PREVIEW_QUALITY = 95      # 预览/标注图 JPEG 质量（2026-09-03 二次提升 88→95：探针实测 88 压缩把清晰度砍掉 2/3）


def _safe_basename(filename):
    """只保留文件名（拒绝绝对路径/../ 穿越），与 model_manager._safe_basename 同款"""
    name = os.path.basename(str(filename or "").replace("\\", "/"))
    if not name or name in (".", "..") or ".." in name.split("/"):
        return None
    return name


class ImageStore:
    """下位机图片文件夹管理：目录切换 / 扫描 / 缩略图 / 读图"""

    def __init__(self, default_dir=DEFAULT_IMAGE_DIR, state_file=STATE_FILE,
                 max_images=MAX_IMAGES):
        self._default_dir = default_dir
        self._state_file = state_file
        self._max_images = max_images
        self._dir = self._pick_start_dir()
        self._ensure_dir()

    # ---------- 目录管理 ----------
    @property
    def dir(self) -> str:
        return self._dir

    @property
    def max_images(self) -> int:
        return self._max_images

    def _load_state(self):
        try:
            with open(self._state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _pick_start_dir(self):
        """启动目录：显式 default_dir > images.json 记录 > 默认值"""
        if self._default_dir and os.path.isdir(self._default_dir):
            return self._default_dir
        saved = self._load_state().get("dir", "")
        if saved and os.path.isdir(saved):
            return saved
        return self._default_dir or DEFAULT_IMAGE_DIR

    def _ensure_dir(self):
        try:
            os.makedirs(self._dir, exist_ok=True)
        except OSError as e:
            log.warning(f"创建图片目录失败: {self._dir} -> {e}")

    def set_dir(self, path: str) -> dict:
        """切换图片文件夹。目录必须存在且可读；持久化到 images.json。"""
        path = str(path or "").strip()
        if not path:
            return {"ok": False, "error": "目录不能为空"}
        if not os.path.isdir(path):
            return {"ok": False, "error": f"目录不存在或不是文件夹: {path}"}
        if not os.access(path, os.R_OK):
            return {"ok": False, "error": f"目录不可读: {path}"}
        self._dir = path
        try:
            with open(self._state_file, "w", encoding="utf-8") as f:
                json.dump({"dir": path}, f, ensure_ascii=False, indent=2)
            log.info(f"图片目录已切换: {path}")
        except OSError as e:
            log.warning(f"写入 images.json 失败（目录已切换，重启后需重新设置）: {e}")
        return {"ok": True, "dir": path, "image_count": self.count_images()}

    def info(self) -> dict:
        return {"dir": self._dir, "exists": os.path.isdir(self._dir),
                "image_count": self.count_images(), "max_count": self._max_images}

    # ---------- 扫描 ----------
    def _scan(self, sort="name"):
        """扫描目录内白名单图片（含子目录？否——仅顶层，简单可控）"""
        out = []
        try:
            with os.scandir(self._dir) as it:
                for e in it:
                    if not e.is_file():
                        continue
                    name = e.name
                    if os.path.splitext(name)[1].lower() not in ALLOWED_EXT:
                        continue
                    try:
                        st = e.stat()
                    except OSError:
                        continue
                    out.append({
                        "name": name,
                        "size_bytes": st.st_size,
                        "mtime": time.strftime("%Y-%m-%d %H:%M:%S",
                                               time.localtime(st.st_mtime)),
                    })
        except OSError as e:
            log.warning(f"扫描图片目录失败: {self._dir} -> {e}")
            return []
        if sort == "mtime":
            out.sort(key=lambda x: x["mtime"], reverse=True)
        else:
            out.sort(key=lambda x: x["name"].lower())
        return out

    def count_images(self) -> int:
        return len(self._scan())

    def list_images(self, offset=0, limit=100, sort="name",
                    thumb=False, thumb_size=256) -> dict:
        """返回图片清单；thumb=True 时每张附 base64 小缩略图（w/h 也一并填充）"""
        all_imgs = self._scan(sort=sort)
        total = len(all_imgs)
        offset = max(0, int(offset or 0))
        limit = min(max(1, int(limit or 100)), self._max_images)
        page = all_imgs[offset:offset + limit]
        images = []
        for item in page:
            row = dict(item)
            if thumb:
                b64, err, w, h = self.thumb_b64(item["name"], max_side=thumb_size)
                if err is None:
                    row["thumb_b64"] = b64
                    row["w"] = w
                    row["h"] = h
                else:
                    row["error"] = err
            else:
                # 不带缩略图也尽量给出 w/h（读图成本高，v1 不强制）
                pass
            images.append(row)
        return {"total": total, "images": images, "offset": offset, "limit": limit}

    # ---------- 读图 / 缩略图 ----------
    def read(self, name) -> tuple:
        """读原图（BGR）。返回 (img 或 None, err)。"""
        name = _safe_basename(name)
        if name is None:
            return None, "非法文件名（仅支持文件名，不支持路径）"
        path = os.path.join(self._dir, name)
        if not os.path.isfile(path):
            return None, f"图片不存在: {name}"
        try:
            raw = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        except Exception as e:
            return None, f"读图失败: {e}"
        if img is None:
            return None, f"图片解码失败（可能损坏或格式不支持）: {name}"
        return img, None

    def thumb_b64(self, name, max_side=256, quality=THUMB_QUALITY) -> tuple:
        """生成缩略图 JPEG base64。返回 (b64 或 None, err, w, h)。"""
        img, err = self.read(name)
        if err is not None:
            return None, err, 0, 0
        h, w = img.shape[:2]
        scale = min(1.0, max_side / max(h, w)) if max_side > 0 else 1.0
        if scale < 1.0:
            img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                             interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return None, "缩略图编码失败", w, h
        return base64.b64encode(buf.tobytes()).decode("ascii"), None, w, h

    def annotate_b64(self, img_bgr, detections, labels=None, max_side=640, quality=PREVIEW_QUALITY) -> tuple:
        """画框后生成 JPEG base64（标注缩略图）。返回 (b64 或 None, err)。
        labels: 模型类别名列表（与训练 data.yaml 顺序一致）；提供且 id 合法时
        标签显示真实缺陷名，否则回退 clsN（与旧行为兼容）。"""
        if img_bgr is None:
            return None, "无图像"
        img = img_bgr.copy()
        for d in detections or []:
            box = d.get("box", [0, 0, 0, 0]) or [0, 0, 0, 0]
            x1, y1, x2, y2 = [int(v) for v in box[:4]]
            conf = float(d.get("confidence", 0) or 0)
            cls_id = int(d.get("class_id", 0) or 0)
            if labels and 0 <= cls_id < len(labels) and str(labels[cls_id]).strip():
                tag = str(labels[cls_id])
            else:
                tag = f"cls{cls_id}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"{tag} {conf:.2f}", (x1, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        h, w = img.shape[:2]
        scale = min(1.0, max_side / max(h, w)) if max_side > 0 else 1.0
        if scale < 1.0:
            img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                             interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return None, "标注图编码失败"
        return base64.b64encode(buf.tobytes()).decode("ascii"), None
