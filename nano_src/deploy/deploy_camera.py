# -*- coding: utf-8 -*-  
"""
相机代码一键部署工具（相机接入规划 P1-4，2026-08-27）
==========================================================
- 默认只推送相机代码（camera_test.py / camera_source.py）到 Nano 并自检 import
- --with-sdk <包路径> 才会执行 SDK 安装（install_mvs_sdk.sh）
- **默认不安装 SDK、不重启服务**（defect-infer.service 等现有链路零触碰）

用法：
  python deploy_camera.py push                # 推送相机代码 + import 自检（只读）
  python deploy_camera.py push --with-sdk /path/to/MVS-*.tar.gz   # 推送+装 SDK
  python deploy_camera.py status              # 板端相关状态检查（只读）
"""
import argparse
import os
import sys
import tempfile
import time

import paramiko

# 连接参数：通过环境变量 NANO_USER / NANO_PASS / NANO_SUDO 传入，禁止硬编码密码。
# 文档此前假设 123456 有误，已修正；可用环境变量 NANO_HOST/USER/PASS/SUDO 覆盖。
HOST = os.environ.get("NANO_HOST", "<NANO_LAN_IP>")
USER = os.environ.get("NANO_USER", "nvidia")
PASS = os.environ.get("NANO_PASS", "")
SUDO = os.environ.get("NANO_SUDO", "")
PORT = 22
BASE = "/home/nvidia/defect_detection"

LOCAL = os.path.dirname(os.path.abspath(__file__))
CAMERA_TEST_LOCAL = os.path.join(LOCAL, "..", "inference", "camera_test.py")
CAMERA_SRC_LOCAL = os.path.join(LOCAL, "..", "server", "camera_source.py")
INSTALL_SH_LOCAL = os.path.join(LOCAL, "install_mvs_sdk.sh")

REMOTE_TEST = BASE + "/inference/camera_test.py"
REMOTE_SRC = BASE + "/server/camera_source.py"
REMOTE_SH = "/home/nvidia/install_mvs_sdk.sh"


def ssh_connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15,
              look_for_keys=False, allow_agent=False)
    return c


def run(c, cmd, sudo=False, timeout=120):
    if sudo:
        cmd = f"echo '{SUDO}' | sudo -S -p '' {cmd}"
    _, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    rc = stdout.channel.recv_exit_status()
    return rc, out, err


def sftp_put(sftp, local, remote, chmod=None):
    sftp.put(local, remote)
    if chmod is not None:
        sftp.chmod(remote, chmod)
    print(f"  → {remote}")


def banner(t):
    print("\n" + "=" * 60)
    print(t)
    print("=" * 60)


def status():
    c = ssh_connect()
    checks = [
        ("python 版本", "python3 --version"),
        ("相机代码已部署", f"ls -la {REMOTE_TEST} {REMOTE_SRC} 2>&1 || echo '(未部署)'"),
        ("MVS SDK", "ls /opt/MVS/bin 2>&1 | head -5 || echo '(MVS 未安装)'"),
        ("USB 设备", "lsusb 2>&1 | grep -i -E 'hik|mv|machine vision' || echo '(未见相机 USB)'"),
        ("推理服务（确认不受影响）", "systemctl is-active defect-infer.service"),
    ]
    for label, cmd in checks:
        banner(label)
        rc, out, err = run(c, cmd)
        print(out.rstrip())
        if err.strip():
            print("[stderr]", err.rstrip())
    c.close()


def push(with_sdk=None):
    c = ssh_connect()
    sftp = c.open_sftp()
    banner("推送相机代码")
    for local, remote, chmod in (
        (CAMERA_TEST_LOCAL, REMOTE_TEST, None),
        (CAMERA_SRC_LOCAL, REMOTE_SRC, None),
        (INSTALL_SH_LOCAL, REMOTE_SH, 0o755),
    ):
        if os.path.isfile(local):
            sftp_put(sftp, local, remote, chmod)
        else:
            print(f"  ⚠ 本地缺失 {local}，跳过")

    # 自检：import camera_source（无 SDK 时应安全降级）
    banner("自检 camera_source import")
    rc, out, err = run(c, f"cd {BASE} && python3 -c 'from server.camera_source import NanoCameraSource; print(NanoCameraSource().device_summary())'")
    print(out.rstrip())
    if err.strip():
        print("[stderr]", err.rstrip())
    print("  → import 自检退出码:", rc)

    # 可选：安装 SDK（显式 --with-sdk 才执行）
    if with_sdk:
        banner("推送并安装 MVS SDK")
        remote_sdk = "/home/nvidia/mvs_sdk.tar.gz"
        sftp_put(sftp, with_sdk, remote_sdk)
        rc, out, err = run(c, f"bash {REMOTE_SH} {remote_sdk}", sudo=True, timeout=600)
        print(out.rstrip())
        if err.strip():
            print("[stderr]", err.rstrip())
        print("  → SDK 安装退出码:", rc)

    sftp.close()
    c.close()
    print("\n完成：相机代码已就位；未重启任何服务。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="相机代码部署（默认不装 SDK、不重启服务）")
    ap.add_argument("action", choices=["push", "status"])
    ap.add_argument("--with-sdk", metavar="SDK包路径", help="同时推送并安装 MVS SDK（需本机有安装包）")
    args = ap.parse_args()

    if args.action == "status":
        status()
    elif args.action == "push":
        push(with_sdk=args.with_sdk)
