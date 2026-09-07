#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下位机独立图片检测 CLI（不依赖上位机，Nano 本机 / SSH 直跑）
- 固定扫描一个图片文件夹（默认 images/input），用当前激活模型逐张检测
- 支持多轮检测（--rounds）
- 标注图保存到 --output 目录；汇总 CSV 保存到 --csv
- 与 TCP 批量检测共用 ImageStore / 同一检测内核（TRTInfer），行为一致

示例：
  python3 server/nano_local_detect.py
  python3 server/nano_local_detect.py --dir /home/nvidia/defect_detection/images/input \
      --rounds 2 --output outputs/nano_local --csv results.csv
"""
import argparse
import csv
import os
import sys
import time
import logging

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "inference"))
from image_store import ImageStore, DEFAULT_IMAGE_DIR
from model_manager import ModelManager

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("nano_local_detect")


def main():
    parser = argparse.ArgumentParser(description="下位机独立图片检测 CLI")
    parser.add_argument("--engine", default="/home/nvidia/defect_detection/models/baseline/yolov8s_fp16.engine",
                        help="TensorRT engine 路径（默认走 ModelManager 自动选择）")
    parser.add_argument("--dir", default="", help=f"图片文件夹（默认 {DEFAULT_IMAGE_DIR}）")
    parser.add_argument("--rounds", type=int, default=1, help="检测轮数（多次检测）")
    parser.add_argument("--output", default="outputs/nano_local", help="标注图输出目录")
    parser.add_argument("--csv", default="", help="汇总 CSV 路径（默认 output/records.csv）")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--limit", type=int, default=0, help="最多检测张数（0=全部）")
    args = parser.parse_args()

    store = ImageStore(default_dir=args.dir or DEFAULT_IMAGE_DIR)
    os.makedirs(args.output, exist_ok=True)
    csv_path = args.csv or os.path.join(args.output, "records.csv")

    images = store.list_images(limit=store.max_images)["images"]
    if args.limit > 0:
        images = images[:args.limit]
    if not images:
        log.warning(f"图片目录为空: {store.dir}")
        return

    log.info(f"加载模型: {args.engine}")
    mm = ModelManager(args.engine, conf_thres=args.conf, iou_thres=args.iou)
    log.info(f"当前激活模型: {mm.current}")

    total = len(images) * max(1, args.rounds)
    done = ok_c = ng_c = 0
    times = []
    failed = []
    csv_rows = []
    t_start = time.time()

    for r in range(1, max(1, args.rounds) + 1):
        for item in images:
            name = item["name"]
            img, err = store.read(name)
            if err is not None:
                log.error(f"读图失败: {name} -> {err}")
                failed.append(name)
                done += 1
                continue
            t0 = time.perf_counter()
            result = mm.engine.detect(img)
            total_ms = (time.perf_counter() - t0) * 1000
            times.append(total_ms)
            dets = result["detections"]
            done += 1
            if dets:
                ng_c += 1
            else:
                ok_c += 1
            # 保存标注图
            out_path = os.path.join(args.output, f"r{r}_{name}")
            b64, aerr = store.annotate_b64(img, dets, getattr(mm.engine, "labels", None), max_side=0)  # max_side=0 -> 不缩放(原图)
            if aerr is None:
                import base64
                with open(out_path, "wb") as f:
                    f.write(base64.b64decode(b64))
            csv_rows.append({
                "round": r, "name": name, "result": "NG" if dets else "OK",
                "count": len(dets), "ms": round(total_ms, 1),
            })
            log.info(f"[{done}/{total}] r{r} {name}: "
                     f"{'NG 缺陷×' + str(len(dets)) if dets else 'OK'} {total_ms:.0f}ms")

    avg = sum(times) / len(times) if times else 0
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["round", "name", "result", "count", "ms"])
        w.writeheader()
        w.writerows(csv_rows)

    log.info("=" * 56)
    log.info(f"完成: {done}/{total} 张, OK={ok_c}, NG={ng_c}, 失败={len(failed)}")
    log.info(f"平均单张: {avg:.1f}ms, 总耗时: {time.time() - t_start:.1f}s")
    log.info(f"标注图目录: {args.output}")
    log.info(f"汇总 CSV: {csv_path}")
    if failed:
        log.warning(f"失败清单: {failed}")


if __name__ == "__main__":
    main()
