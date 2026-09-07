# -*- coding: utf-8 -*-
"""板端批量检测 28 张测试图，输出每张检出明细（类别/置信度）"""
import socket
import struct
import json
import sys

sys.path.insert(0, r"C:\Users\机械革命\Desktop\基于深度学习的缺陷检测边缘设备开发\nano_src\deploy")
import deploy_to_nano as d

HOST, PORT = "<NANO_LAN_IP>", 8888


def send(sock, obj):
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    sock.sendall(struct.pack(">I", len(data)) + data)


def recv(sock, timeout=30):
    sock.settimeout(timeout)
    hdr = b""
    while len(hdr) < 4:
        c = sock.recv(4 - len(hdr))
        if not c:
            return None
        hdr += c
    n = struct.unpack(">I", hdr)[0]
    buf = b""
    while len(buf) < n:
        c = sock.recv(n - len(buf))
        if not c:
            return None
        buf += c
    return json.loads(buf.decode("utf-8"))


def main():
    sock = socket.create_connection((HOST, PORT), timeout=10)
    send(sock, {"type": "nano_images_request", "thumb": False, "limit": 500})
    r = recv(sock)
    names = [i["name"] for i in r["images"]]
    print(f"共 {len(names)} 张\n")

    send(sock, {"type": "nano_batch_request", "names": names, "rounds": 1,
                "annotate": False})
    r = recv(sock)
    if not r.get("ok"):
        print("批量启动失败:", r)
        return

    results = []
    while True:
        m = recv(sock, timeout=60)
        if m is None:
            break
        if m.get("type") == "nano_batch_item":
            results.append(m)
        elif m.get("type") == "nano_batch_done":
            break

    # 按检出/未检出分组输出
    hit, miss = [], []
    for it in sorted(results, key=lambda x: x["name"]):
        dets = it.get("detections", [])
        if dets:
            top = max(dets, key=lambda dd: dd.get("confidence", 0))
            hit.append((it["name"], len(dets), top.get("class_id"), top.get("confidence")))
        else:
            miss.append(it["name"])

    print(f"=== 检出 {len(hit)} 张 / 未检出 {len(miss)} 张 ===")
    print("\n-- 检出明细（文件名 | 目标数 | 最高置信度类别id | conf）--")
    for name, n, cid, conf in hit:
        print(f"  {name:<28} {n:>2} 个目标  cls{cid:<3} conf={conf:.2f}")
    if miss:
        print("\n-- 未检出 --")
        for name in miss:
            print(f"  {name}")

    # 类别 id → 名称映射（35 类模型，取 NEU 6 类已知映射）
    neu_map = {0: "crazing", 1: "inclusion", 2: "patches",
               3: "pitted_surface", 4: "rolled-in_scale", 5: "scratches"}
    print("\n提示: 上位机显示的是 35 类模型的实际类别名；NEU 图对应类名映射:",
          neu_map)
    sock.close()


if __name__ == "__main__":
    main()
