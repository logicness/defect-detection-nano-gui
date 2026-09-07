#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下位机相机取流测试 CLI（相机接入规划 P1-1，2026-08-28 真实版）
================================================================
- 枚举 MVS 设备 / 单帧抓图 / 保存样例图到 outputs/（复用 NanoCameraSource）
- 无 MVS SDK / 无相机时给出清晰降级提示（不崩溃）

用法：
  python3 inference/camera_test.py --list          # 仅枚举设备
  python3 inference/camera_test.py --snap          # 抓 1 帧保存
  python3 inference/camera_test.py --snap --count 3 --output outputs/camera
  python3 inference/camera_test.py --snap --exposure 5000 --gain 5 --trigger continuous
退出码：0 成功 / 1 无设备或失败 / 2 SDK 不可用
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "server"))
from camera_source import NanoCameraSource, _load_mvs, OK  # noqa: E402

OUTPUT_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "outputs")


def list_devices() -> list:
    """枚举 MVS 设备，返回描述字符串列表。"""
    moc, _ = _load_mvs()
    if not moc:
        return []
    devs = []
    for layer, kind in ((moc.MV_USB_DEVICE, "USB3"), (moc.MV_GIGE_DEVICE, "GigE")):
        dev_list = moc.MV_CC_DEVICE_INFO_LIST()
        if moc.MvCamera.MV_CC_EnumDevices(layer, dev_list) != OK:
            continue
        for i in range(dev_list.nDeviceNum):
            devs.append(f"相机 {len(devs) + 1}: 接口={kind}")
    return devs


def grab_frame(output: str, count: int = 1, exposure_us: float = -1.0, gain: float = -1.0,
               trigger_mode: str = "continuous") -> int:
    """抓取 count 帧并保存样例图。返回成功保存帧数。"""
    os.makedirs(output, exist_ok=True)
    src = NanoCameraSource(exposure_us=exposure_us, gain=gain, trigger_mode=trigger_mode)
    if not src.open():
        print(f"[camera_test] 打开相机失败：{src.device_summary().get('last_error')}", file=sys.stderr)
        return 0
    saved = 0
    try:
        for i in range(count):
            ok, frame = src.read()
            if not ok or frame is None:
                print(f"[camera_test] 第 {i + 1} 帧取流失败：{src.device_summary().get('last_error')}",
                      file=sys.stderr)
                continue
            import time
            from datetime import datetime
            import cv2
            fn = os.path.join(output, f"camera_{datetime.now():%Y%m%d_%H%M%S}_{i:02d}.png")
            cv2.imwrite(fn, frame)
            print(f"  保存: {fn} ({frame.shape[1]}x{frame.shape[0]}, "
                  f"BGR, {os.path.getsize(fn) // 1024} KB)")
            saved += 1
            time.sleep(0.1)  # 给相机节奏
    finally:
        src.close()
    return saved


def main():
    ap = argparse.ArgumentParser(description="MVS 相机取流测试")
    ap.add_argument("--list", action="store_true", help="只枚举设备")
    ap.add_argument("--snap", action="store_true", help="单帧/多帧抓图")
    ap.add_argument("--count", type=int, default=1, help="抓图帧数")
    ap.add_argument("--output", default=OUTPUT_DEFAULT, help="样例图输出目录")
    ap.add_argument("--exposure", type=float, default=-1.0, help="曝光时间 us（-1 用默认）")
    ap.add_argument("--gain", type=float, default=-1.0, help="增益 dB（-1 用默认）")
    ap.add_argument("--trigger", choices=["continuous", "hardware", "software"],
                    default="continuous", help="触发模式")
    args = ap.parse_args()

    moc, _ = _load_mvs()
    if not moc:
        print("[camera_test] MVS SDK 绑定不可用（MVCAM_COMMON_RUNENV 或 MvImport 缺失）。", file=sys.stderr)
        print("[camera_test] 安装 SDK 后确认：ls /opt/MVS/Samples/aarch64/Python/MvImport", file=sys.stderr)
        sys.exit(2)

    devs = list_devices()
    if not devs:
        print("[camera_test] 未枚举到相机设备（相机未连接 / USB 未识别）。")
        print("[camera_test] 检查：相机接 Nano USB3.0、原装线缆、lsusb 可见。")
        sys.exit(1)

    if args.list:
        for d in devs:
            print(d)
        sys.exit(0)

    if args.snap:
        n = grab_frame(args.output, args.count, args.exposure, args.gain, args.trigger)
        print(f"[camera_test] 抓图完成：{n}/{args.count} 帧 → {args.output}")
        sys.exit(0 if n > 0 else 1)

    ap.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()