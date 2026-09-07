# -*- coding: utf-8 -*-
"""产线模拟相机 协议级集成测试（Windows 可跑：mock 引擎 + 真实 TCP）
覆盖：订阅/退订、start→收 stream_frame 连续帧、set_fps、stop→无帧、多订阅者广播、状态字段。
"""
import asyncio
import json
import os
import struct
import sys
import time
import types

import numpy as np

# Windows 无 TensorRT/CUDA：stub trt_infer（测试只走 mock 引擎）
_fake = types.ModuleType("trt_infer")
_fake.TRTInfer = object
sys.modules.setdefault("trt_infer", _fake)

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
    conf_thres = 0.25
    iou_thres = 0.45

    def detect(self, img_bgr):
        time.sleep(0.01)
        return {"detections": [{"box": [5, 5, 60, 50], "confidence": 0.9,
                                "class_id": 3, "result": "NG"}],
                "timing": {"gpu_inference_ms": 4.0, "total_ms": 5.0}}

    def update_params(self, conf_thres=None, iou_thres=None):
        pass


class FakeMM:
    compiling = False
    current = "/home/nvidia/defect_detection/models/neu/Universal_Metal_35c_fp16.engine"

    def __init__(self):
        self.engine = FakeEngine()

    def upload_abort(self):
        pass


class ProtoClient:
    def __init__(self, reader, writer):
        self.reader = reader
        self.writer = writer

    def send(self, obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.writer.write(struct.pack(">I", len(data)) + data)

    async def recv(self, timeout=8.0):
        try:
            hdr = await asyncio.wait_for(self.reader.readexactly(4), timeout)
        except (asyncio.IncompleteReadError, asyncio.TimeoutError):
            return None
        n = struct.unpack(">I", hdr)[0]
        body = await asyncio.wait_for(self.reader.readexactly(n), 5.0)
        return json.loads(body.decode("utf-8"))


async def main():
    tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".stream_imgs")
    os.makedirs(tmp, exist_ok=True)
    for i in range(4):
        img = np.full((120 + i * 10, 160 + i * 10, 3), 70, dtype=np.uint8)
        __import__("cv2").imencode(".png", img)[1].tofile(os.path.join(tmp, f"s_{i:02d}.png"))

    mm = FakeMM()
    server = DetectServer(mm, "127.0.0.1", 0)
    server._store = ImageStore(default_dir=tmp, state_file=os.path.join(tmp, "i.json"))
    server._camera_sim.store = server._store   # 同步覆盖（__init__ 的旧 store 指向 Nano 默认路径）
    # 测试绕过 server.start()，需手动设置事件循环（CameraSimulator 需要 loop）
    server._loop = asyncio.get_running_loop()
    server._camera_sim.loop = server._loop
    srv = await asyncio.start_server(server._handle_client, "127.0.0.1", 0)
    port = srv.sockets[0].getsockname()[1]

    async def mkclient():
        r, w = await asyncio.open_connection("127.0.0.1", port)
        return ProtoClient(r, w)

    print("[1] 订阅 + start → 收连续帧")
    c1 = await mkclient()
    c1.send({"type": "stream_subscribe_request", "subscribe": True})
    r = await c1.recv()
    check("订阅响应", r and r.get("type") == "stream_subscribe_response" and r.get("ok"),
          f"running={r and r.get('running')}")
    c1.send({"type": "stream_control_request", "action": "start"})
    r = await c1.recv()
    check("start 响应", r and r.get("type") == "stream_control_response"
          and r.get("ok") and r.get("running") is True, f"fps={r and r.get('fps')}")
    frames = []
    for _ in range(6):
        m = await c1.recv(timeout=5)
        if m is None or m.get("type") != "stream_frame":
            break
        frames.append(m)
    check("收到连续 stream_frame", len(frames) >= 5, f"frames={len(frames)}")
    check("帧 seq 递增", all(frames[i]["seq"] < frames[i + 1]["seq"]
                            for i in range(len(frames) - 1)),
          f"seqs={[f['seq'] for f in frames]}")
    check("帧带标注图/检测/模型", all(f.get("annot_b64") and f.get("detections")
                                and "Universal_Metal_35c" in f.get("model", "")
                                for f in frames[:2]))

    print("[2] set_fps 生效")
    c1.send({"type": "stream_control_request", "action": "set_fps", "fps": 25})
    r = await c1.recv()
    check("set_fps 响应", r and r.get("ok") and r.get("fps") == 25, f"fps={r and r.get('fps')}")

    print("[3] 多订阅者广播")
    c2 = await mkclient()
    c2.send({"type": "stream_subscribe_request", "subscribe": True})
    r = await c2.recv()
    check("第二订阅者", r and r.get("ok"))
    got1 = got2 = None
    for _ in range(20):
        m1 = await c1.recv(timeout=3)
        m2 = await c2.recv(timeout=3)
        if m1 and m1.get("type") == "stream_frame" and got1 is None:
            got1 = m1
        if m2 and m2.get("type") == "stream_frame" and got2 is None:
            got2 = m2
        if got1 and got2:
            break
    check("双订阅者同帧", got1 is not None and got2 is not None
          and got1.get("seq") == got2.get("seq"),
          f"seq1={got1 and got1.get('seq')} seq2={got2 and got2.get('seq')}")

    print("[4] stop → 不再推帧")
    c1.send({"type": "stream_control_request", "action": "stop"})
    r = await c1.recv()
    check("stop 响应", r and r.get("ok") and r.get("running") is False,
          f"running={r and r.get('running')}")
    c2.send({"type": "stream_control_request", "action": "set_fps", "fps": 1})
    r = await c2.recv()
    m = await c2.recv(timeout=1.5)
    check("stop 后无新帧", m is None or m.get("type") != "stream_frame")

    print("[5] 退订后不推送")
    c2.send({"type": "stream_subscribe_request", "subscribe": False})
    r = await c2.recv()
    check("退订响应", r and r.get("ok") and r.get("subscribed") is False)
    c1.send({"type": "stream_control_request", "action": "start"})
    r = await c1.recv()
    check("重启", r and r.get("ok") and r.get("running") is True)
    m = await c2.recv(timeout=1.2)
    check("退订者收不到帧", m is None or m.get("type") != "stream_frame")
    c1.send({"type": "stream_control_request", "action": "stop"})
    await c1.recv()

    c1.writer.close()
    c2.writer.close()
    srv.close()
    await srv.wait_closed()
    print(f"\n结果: PASS={PASS} FAIL={FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    asyncio.run(main())
