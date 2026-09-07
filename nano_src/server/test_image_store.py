# -*- coding: utf-8 -*-
"""ImageStore 本地功能测试（Windows 环境，不依赖 Nano）"""
import base64
import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from image_store import ImageStore, _safe_basename

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


def main():
    tmp_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".imgstore_test")
    tmp = os.path.join(tmp_root, "dir1")
    os.makedirs(tmp, exist_ok=True)
    # 生成 3 张测试图（cv2.imwrite 不支持中文路径，用 imencode + tofile）
    for i, (w, h) in enumerate(((640, 480), (200, 300), (1280, 720))):
        img = np.full((h, w, 3), (i + 1) * 40, dtype=np.uint8)
        cv2.rectangle(img, (w // 4, h // 4), (w // 2, h // 2), (0, 0, 255), 3)
        cv2.imencode(".png", img)[1].tofile(os.path.join(tmp, f"test_{i:02d}.png"))
    # 非白名单文件
    with open(os.path.join(tmp, "note.txt"), "w") as f:
        f.write("x")
    with open(os.path.join(tmp, "bad.gif"), "wb") as f:
        f.write(b"GIF89a")

    print(f"[1] 目录初始化/扫描")
    st = ImageStore(default_dir=tmp, state_file=os.path.join(tmp, "images.json"))
    check("扫描只含 3 张白名单图", st.count_images() == 3, f"count={st.count_images()}")
    r = st.list_images(thumb=True, thumb_size=64)
    check("列表 total=3", r["total"] == 3, f"total={r['total']}")
    imgs = r["images"]
    check("每张带缩略图/尺寸", all("thumb_b64" in i and i["w"] > 0 for i in imgs))
    if imgs and "thumb_b64" in imgs[0]:
        raw = base64.b64decode(imgs[0]["thumb_b64"])
        img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        check("缩略图可解码", img is not None,
              f"size={None if img is None else img.shape}")
        check("缩略图最长边<=64", img is not None and max(img.shape[:2]) <= 64)

    print(f"[2] 分页")
    r2 = st.list_images(offset=1, limit=1)
    check("offset/limit 生效", r2["total"] == 3 and len(r2["images"]) == 1)

    print(f"[3] 读图")
    img, err = st.read("test_00.png")
    check("读图成功", err is None and img.shape[:2] == (480, 640))
    img, err = st.read("not_exist.jpg")
    check("不存在报错", err is not None)
    img, err = st.read("bad.gif")
    check("非白名单读图被拒", err is not None)

    print(f"[4] 文件名安全")
    check("拒绝 ../", _safe_basename("../x.jpg") == "x.jpg")  # basename 只留 x.jpg，安全
    check("拒绝绝对路径", _safe_basename("/etc/passwd") == "passwd")
    check("拒绝空", _safe_basename("") is None)
    check("拒绝 ..", _safe_basename("..") is None)

    print(f"[5] 目录切换")
    tmp2 = os.path.join(tmp_root, "dir2")
    os.makedirs(tmp2, exist_ok=True)
    cv2.imencode(".png", np.zeros((10, 10, 3), dtype=np.uint8))[1].tofile(
        os.path.join(tmp2, "a.png"))
    rr = st.set_dir(tmp2)
    check("切换成功", rr["ok"] and st.dir == tmp2, str(rr))
    rr = st.set_dir("Z:/not_exist_dir_xyz")
    check("切换不存在目录被拒", not rr["ok"])
    # 持久化恢复
    st2 = ImageStore(default_dir="", state_file=os.path.join(tmp, "images.json"))
    check("重启恢复目录", st2.dir == tmp2, st2.dir)

    print(f"[6] 标注图")
    img, err = st.read("a.png")
    dets = [{"box": [1, 1, 8, 8], "confidence": 0.9, "class_id": 3}]
    b64, aerr = st.annotate_b64(img, dets, max_side=0)
    check("标注图原尺寸生成", aerr is None and b64)
    b64, aerr = st.annotate_b64(img, dets, max_side=64)
    check("标注图缩放生成", aerr is None and b64)
    raw = base64.b64decode(b64)
    ai = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    check("标注图可解码且<=64", ai is not None and max(ai.shape[:2]) <= 64)

    print(f"\n结果: PASS={PASS} FAIL={FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
