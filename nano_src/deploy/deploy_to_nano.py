# -*- coding: utf-8 -*-
"""Nano 板端部署与联调工具（paramiko）
用法：
  python deploy_to_nano.py status   # 只读状态检查（服务/端口/文件）
  python deploy_to_nano.py deploy   # 备份并部署 server 代码 + 重启服务 + 自检
  python deploy_to_nano.py selftest # 部署后协议自检（发 nano_images/detect/batch）
"""
import os
import sys
import time

import paramiko

# 连接参数：通过环境变量 NANO_USER / NANO_PASS / NANO_SUDO 传入，禁止硬编码密码。
# 可用环境变量 NANO_HOST/USER/PASS/SUDO 覆盖（与 deploy_camera.py 保持一致）。
HOST = os.environ.get("NANO_HOST", "<NANO_LAN_IP>")
USER = os.environ.get("NANO_USER", "nvidia")
PASS = os.environ.get("NANO_PASS", "")
SUDO = os.environ.get("NANO_SUDO", "")
PORT = 22
BASE = "/home/nvidia/defect_detection"
SERVER_DIR = BASE + "/server"
IMAGE_DIR = BASE + "/images/input"

LOCAL_SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "server")


def ssh_connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15,
              look_for_keys=False, allow_agent=False)
    return c


def run(c, cmd, sudo=False, timeout=120):
    if sudo:
        cmd = f"echo '{SUDO}' | sudo -S -p '' {cmd}"
    stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    rc = stdout.channel.recv_exit_status()
    return rc, out, err


def banner(t):
    print("\n" + "=" * 60)
    print(t)
    print("=" * 60)


def status():
    c = ssh_connect()
    checks = [
        ("主机/用户", "hostname && whoami"),
        ("推理服务状态", "systemctl status defect-infer.service --no-pager -l 2>&1 | head -30"),
        ("8888 监听", "ss -tlnp 2>/dev/null | grep 8888 || echo '8888 NOT LISTENING'"),
        ("server 目录", f"ls -la {SERVER_DIR}/ 2>&1"),
        ("图片目录", f"ls -la {IMAGE_DIR}/ 2>&1 || echo '(images/input 不存在)'"),
        ("测试图", f"ls -la {BASE}/tests/test_images/ 2>&1 || echo '(无 tests/test_images)'"),
        ("models.json", f"cat {SERVER_DIR}/models.json 2>&1"),
        ("运行中的 server 版本", "grep -n 'nano_batch_request' " + SERVER_DIR + "/infer_server.py | head -3 || echo '(未部署新协议)'"),
    ]
    for label, cmd in checks:
        banner(label)
        rc, out, err = run(c, cmd)
        print(out.rstrip())
        if err.strip():
            print("[stderr]", err.rstrip())
    c.close()


def deploy():
    c = ssh_connect()
    stamp = time.strftime("%Y%m%d_%H%M%S")

    # 1) 备份旧文件
    banner("备份旧 infer_server.py")
    rc, out, err = run(c, f"cp -v {SERVER_DIR}/infer_server.py {SERVER_DIR}/infer_server.py.bak_{stamp}")
    print(out or err)

    # 2) SFTP 上传新文件
    banner("上传新文件")
    sftp = c.open_sftp()
    files = ["image_store.py", "infer_server.py", "nano_local_detect.py"]
    for f in files:
        local = os.path.join(LOCAL_SERVER, f)
        if not os.path.isfile(local):
            print(f"!! 本地缺失 {local}")
            continue
        remote = f"{SERVER_DIR}/{f}"
        sftp.put(local, remote)
        print(f"  上传 {f} -> {remote}")
    sftp.close()

    # 3) 语法检查（python3 -m py_compile，写 __pycache__ 无妨）
    banner("远端语法检查")
    rc, out, err = run(c, f"cd {SERVER_DIR} && python3 -m py_compile image_store.py infer_server.py nano_local_detect.py && echo SYNTAX_OK || echo SYNTAX_FAIL")
    print(out or err)

    # 4) 确保图片目录存在
    banner("确保图片目录存在")
    run(c, f"mkdir -p {IMAGE_DIR} && chown nvidia:nvidia {IMAGE_DIR}")

    # 5) 重启服务
    banner("重启 defect-infer.service")
    rc, out, err = run(c, "systemctl restart defect-infer.service", sudo=True, timeout=60)
    print(out or err)
    time.sleep(3)

    # 6) 自检：服务状态 + 端口
    banner("服务状态 + 端口")
    rc, out, err = run(c, "systemctl is-active defect-infer.service && ss -tlnp | grep 8888 || echo 'NO_8888'")
    print(out or err)
    rc, out, err = run(c, "journalctl -u defect-infer.service -n 15 --no-pager -o cat 2>&1")
    print(out.rstrip())
    c.close()


def seed():
    """把测试图复制到 images/input（联调用），并写一张合成图验证中文名/格式"""
    c = ssh_connect()
    banner("复制测试图到 images/input")
    rc, out, err = run(c, f"cp -vf {BASE}/tests/test_images/bus.jpg {IMAGE_DIR}/ 2>&1; "
                         f"cp -vf {BASE}/tests/test_images/random.jpg {IMAGE_DIR}/ 2>&1")
    print(out or err)
    rc, out, err = run(c, f"ls -la {IMAGE_DIR}/")
    print(out.rstrip())
    c.close()


def selftest():
    """部署后协议自检：从 PC 直连 8888，发目录/列表/单张/批量，不依赖上位机 GUI"""
    import socket
    import struct
    import json
    import base64

    def frame_send(sock, obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        sock.sendall(struct.pack(">I", len(data)) + data)

    def frame_recv(sock, timeout=10):
        sock.settimeout(timeout)
        hdr = _recv_exact(sock, 4)
        if not hdr:
            return None
        n = struct.unpack(">I", hdr)[0]
        body = _recv_exact(sock, n)
        return json.loads(body.decode("utf-8"))

    def _recv_exact(sock, n):
        buf = b""
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    sock = socket.create_connection((HOST, 8888), timeout=10)
    print("连接 8888 成功")

    # 目录
    frame_send(sock, {"type": "nano_image_dir_request", "action": "get"})
    r = frame_recv(sock)
    print("[dir]", r)

    # 列表
    frame_send(sock, {"type": "nano_images_request", "thumb": True, "thumb_size": 128, "limit": 50})
    r = frame_recv(sock)
    names = [i["name"] for i in r.get("images", [])]
    print(f"[images] total={r.get('total')}, 前几张={names[:5]}")

    if not names:
        print("!! 图片目录为空，跳过单张/批量检测。请放入图片后重跑 selftest。")
        sock.close()
        return

    # 单张检测
    frame_send(sock, {"type": "nano_detect_request", "name": names[0], "annotate": True, "thumb_size": 640})
    r = frame_recv(sock)
    print(f"[detect] name={r.get('name')} ok={r.get('ok')} 目标数={len(r.get('detections', []))} model={r.get('model')} annot={'有' if r.get('annot_b64') else '无'}")

    # 批量（最多 4 张 × 1 轮）
    pick = names[:4]
    frame_send(sock, {"type": "nano_batch_request", "names": pick, "rounds": 1, "annotate": True})
    r = frame_recv(sock)
    print(f"[batch] {r}")
    if r.get("ok"):
        items = 0
        done = None
        while True:
            m = frame_recv(sock, timeout=30)
            if m is None:
                break
            if m.get("type") == "nano_batch_item":
                items += 1
            elif m.get("type") == "nano_batch_done":
                done = m
                break
        print(f"[batch结果] items={items} done={done}")
    sock.close()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "status"
    {"status": status, "deploy": deploy, "seed": seed, "selftest": selftest}[mode]()
