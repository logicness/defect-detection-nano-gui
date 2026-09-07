# -*- coding: utf-8 -*-
"""S2：部署产线模拟相机到板端 + 测试"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deploy_to_nano as d

LOCAL_SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server")


def main():
    c = d.ssh_connect()
    stamp = time.strftime("%Y%m%d_%H%M%S")

    # 1) 备份
    d.banner("备份")
    rc, out, err = d.run(c, f"cp -v {d.SERVER_DIR}/infer_server.py {d.SERVER_DIR}/infer_server.py.bak_{stamp}")
    print(out or err)

    # 2) 上传 2 个文件
    d.banner("上传 camera_sim.py + infer_server.py")
    sftp = c.open_sftp()
    for f in ["camera_sim.py", "infer_server.py"]:
        local = os.path.join(LOCAL_SERVER, f)
        if not os.path.isfile(local):
            print(f"!! 本地缺失 {local}")
            continue
        remote = f"{d.SERVER_DIR}/{f}"
        sftp.put(local, remote)
        print(f"  上传 {f} -> {remote}")
    sftp.close()

    # 3) 远端语法检查
    d.banner("远端语法检查")
    rc, out, err = d.run(c, f"cd {d.SERVER_DIR} && python3 -m py_compile camera_sim.py infer_server.py && echo SYNTAX_OK || echo SYNTAX_FAIL")
    print(out or err)

    # 4) 重启服务
    d.banner("重启 defect-infer.service")
    rc, out, err = d.run(c, "systemctl restart defect-infer.service", sudo=True, timeout=60)
    print(out or err)
    time.sleep(4)

    # 5) 自检：服务状态 + 8888 + stream 协议
    d.banner("服务状态 + 端口")
    rc, out, err = d.run(c, "systemctl is-active defect-infer.service && ss -tlnp | grep 8888 || echo NO_8888")
    print(out or err)

    # 6) 流协议自检（本地 TCP 连板端）
    d.banner("流协议自检")
    import socket, struct, json
    sock = socket.create_connection(("192.168.1.101", 8888), timeout=10)
    def send(obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        sock.sendall(struct.pack(">I", len(data)) + data)
    def recv(timeout=10):
        sock.settimeout(timeout)
        hdr = b""
        while len(hdr) < 4:
            c2 = sock.recv(4 - len(hdr))
            if not c2: return None
            hdr += c2
        n = struct.unpack(">I", hdr)[0]
        buf = b""
        while len(buf) < n:
            c2 = sock.recv(n - len(buf))
            if not c2: return None
            buf += c2
        return json.loads(buf.decode("utf-8"))

    # subscribe
    send({"type": "stream_subscribe_request", "subscribe": True})
    r = recv()
    print(f"[subscribe] {r}")

    # start
    send({"type": "stream_control_request", "action": "start"})
    r = recv()
    print(f"[start] {r}")

    # 收 3 帧
    for i in range(3):
        m = recv(timeout=5)
        if m:
            print(f"[frame] seq={m.get('seq')} name={m.get('name')} "
                  f"dets={len(m.get('detections',[]))} annot={'Y' if m.get('annot_b64') else 'N'}")
        else:
            print(f"[frame] timeout")

    # stop
    send({"type": "stream_control_request", "action": "stop"})
    r = recv()
    print(f"[stop] {r}")

    # stop 后无帧
    m = recv(timeout=1.5)
    print(f"[no-frame-after-stop] {'PASS' if m is None or m.get('type') != 'stream_frame' else 'FAIL'}")

    sock.close()
    print("\nS2 板端部署 + 流协议自检完成")


if __name__ == "__main__":
    main()
