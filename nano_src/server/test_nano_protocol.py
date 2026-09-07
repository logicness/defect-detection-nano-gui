# -*- coding: utf-8 -*-
"""下位机图片检测 协议级集成测试（Windows 可跑：mock 模型 + 真实 TCP 收发）
覆盖：目录查询/设置、列表+缩略图、单张预览、单张检测、批量检测（进度/条目/完成）、停止。
"""
import asyncio
import json
import os
import struct
import sys
import time
import types

import numpy as np

# Windows 无 TensorRT/CUDA：stub 掉 trt_infer 模块（测试只走 mock 引擎，不实例化 TRTInfer）
_fake_trt = types.ModuleType("trt_infer")
_fake_trt.TRTInfer = object
sys.modules.setdefault("trt_infer", _fake_trt)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "inference"))
from image_store import ImageStore
from infer_server import DetectServer

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name} {detail}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


class FakeEngine:
    """模拟 TRT 引擎：固定返回 1 个检测框，可注入延迟"""
    def __init__(self, delay=0.01):
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.delay = delay

    def detect(self, img_bgr):
        if self.delay:
            time.sleep(self.delay)
        return {
            "detections": [{"box": [10, 20, 100, 80], "confidence": 0.87,
                            "class_id": 3, "result": "NG"}],
            "timing": {"gpu_inference_ms": 5.0, "total_ms": 6.0},
        }

    def update_params(self, conf_thres=None, iou_thres=None):
        if conf_thres is not None:
            self.conf_thres = conf_thres
        if iou_thres is not None:
            self.iou_thres = iou_thres


class FakeModelManager:
    compiling = False
    current = "/home/nvidia/defect_detection/models/neu/Universal_Metal_35c_fp16.engine"

    def __init__(self, engine):
        self.engine = engine

    def upload_abort(self):
        pass


class ProtoClient:
    """帧协议客户端（4 字节大端长度头 + JSON）"""

    def __init__(self, reader, writer):
        self.reader = reader
        self.writer = writer

    def send(self, obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.writer.write(struct.pack(">I", len(data)) + data)

    async def recv(self, timeout=5.0):
        try:
            header = await asyncio.wait_for(self.reader.readexactly(4), timeout)
        except (asyncio.IncompleteReadError, asyncio.TimeoutError):
            return None
        length = struct.unpack(">I", header)[0]
        body = await asyncio.wait_for(self.reader.readexactly(length), 5.0)
        return json.loads(body.decode("utf-8"))


async def main():
    tmp_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".imgstore_proto")
    os.makedirs(tmp_root, exist_ok=True)
    for i in range(3):
        img = np.full((100 + i * 20, 120 + i * 20, 3), 60, dtype=np.uint8)
        cv2_ok = __import__("cv2").imencode(".png", img)[1].tofile(
            os.path.join(tmp_root, f"p_{i:02d}.png"))

    mm = FakeModelManager(FakeEngine(delay=0.02))
    server = DetectServer(mm, "127.0.0.1", 0,
                          image_dir=os.path.join(tmp_root, "nonexist_x"))  # 默认目录
    # 指向测试目录
    server._store = ImageStore(default_dir=tmp_root,
                               state_file=os.path.join(tmp_root, "images.json"))

    srv = await asyncio.start_server(server._handle_client, "127.0.0.1", 0)
    port = srv.sockets[0].getsockname()[1]
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    c = ProtoClient(reader, writer)

    print("[1] 目录查询")
    c.send({"type": "nano_image_dir_request", "action": "get"})
    r = await c.recv()
    check("get ok", r and r.get("type") == "nano_image_dir_response" and r.get("ok"),
          f"dir={r and r.get('dir')}")

    print("[2] 图片列表 + 缩略图")
    c.send({"type": "nano_images_request", "thumb": True, "thumb_size": 64})
    r = await c.recv()
    check("images ok", r and r.get("type") == "nano_images_response"
          and r.get("total") == 3 and len(r.get("images", [])) == 3,
          f"total={r and r.get('total')}")
    imgs = r["images"]
    check("缩略图存在", all(i.get("thumb_b64") for i in imgs))

    print("[3] 单张预览")
    c.send({"type": "nano_image_request", "name": imgs[0]["name"], "size": 200})
    r = await c.recv()
    check("preview ok", r and r.get("ok") and r.get("image_b64"), f"name={r and r.get('name')}")
    check("尺寸字段", r and r.get("w") > 0 and r.get("h") > 0)

    print("[4] 单张检测")
    c.send({"type": "nano_detect_request", "name": imgs[0]["name"], "annotate": True})
    r = await c.recv()
    check("detect ok", r and r.get("ok") and len(r.get("detections", [])) == 1
          and r.get("annot_b64"), f"model={r and r.get('model')}")
    check("检测带模型名", r and "Universal_Metal_35c" in r.get("model", ""))

    print("[5] 批量检测 3张×2轮")
    c.send({"type": "nano_batch_request", "names": [i["name"] for i in imgs], "rounds": 2})
    r = await c.recv()
    check("batch 启动", r and r.get("type") == "nano_batch_response" and r.get("ok")
          and r.get("total") == 6, f"total={r and r.get('total')}")
    sid = r["session_id"]
    items = 0
    done = None
    progress = 0
    for _ in range(60):  # 收 6 item + 6 progress + 1 done
        m = await c.recv(timeout=5)
        if m is None:
            break
        t = m.get("type")
        if t == "nano_batch_item":
            items += 1
            check(f"item #{items} 带结果", m.get("ok") and len(m.get("detections", [])) == 1,
                  m.get("name", ""))
        elif t == "nano_batch_progress":
            progress = m.get("done", 0)
        elif t == "nano_batch_done":
            done = m
            break
    check("收到全部 item", items == 6, f"items={items}")
    check("收到 done 汇总", done is not None and done.get("ok_count", -1) >= 0
          and done.get("ng_count") == 6 and done.get("total") == 6,
          f"ok={done and done.get('ok_count')} ng={done and done.get('ng_count')}")

    print("[6] 停止批量")
    server._store.set_dir(tmp_root)
    c.send({"type": "nano_batch_request", "names": [i["name"] for i in imgs], "rounds": 5})
    r = await c.recv()
    check("batch2 启动", r and r.get("ok"))
    await asyncio.sleep(0.15)  # 让前几张跑完
    c.send({"type": "nano_batch_stop_request"})
    got_stop = False
    stopped_done = None
    for _ in range(60):  # stop_response 与批量推送并发到达，顺序不保证，循环找
        m = await c.recv(timeout=5)
        if m is None:
            break
        if m.get("type") == "nano_batch_stop_response":
            got_stop = m.get("stopped") is True
        elif m.get("type") == "nano_batch_done":
            stopped_done = m
            break
    check("stop 响应", got_stop)
    check("stop 后 done(stopped)", stopped_done is not None and stopped_done.get("stopped")
          and stopped_done.get("total") == 15 and stopped_done.get("done", 0) < 15,
          f"done={stopped_done and stopped_done.get('done')}/{stopped_done and stopped_done.get('total')}")
    check("会话已清空", server._batch is None)

    print("[7] 批量进行中拒绝新批量")
    c.send({"type": "nano_batch_request", "names": [i["name"] for i in imgs], "rounds": 1})
    r = await c.recv()
    check("启动成功", r and r.get("ok"))
    c.send({"type": "nano_batch_request", "names": [i["name"] for i in imgs], "rounds": 1})
    r = await c.recv()
    check("并发启动被拒", r and not r.get("ok") and "busy" in r.get("error", ""))
    c.send({"type": "nano_batch_stop_request"})
    await c.recv()
    for _ in range(40):
        m = await c.recv(timeout=5)
        if m is None or m.get("type") == "nano_batch_done":
            break

    print("[8] 非法文件名")
    c.send({"type": "nano_detect_request", "name": "../etc/passwd"})
    r = await c.recv()
    check("路径穿越被拒", r and not r.get("ok"))
    c.send({"type": "nano_images_request", "thumb": False})
    r = await c.recv()
    check("无缩略图列表", r and r.get("ok") and not r[ "images"][0].get("thumb_b64"))

    writer.close()
    srv.close()
    await srv.wait_closed()

    print(f"\n结果: PASS={PASS} FAIL={FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    asyncio.run(main())
