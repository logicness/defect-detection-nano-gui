# -*- coding: utf-8 -*-
"""收集缺陷测试图并上传到下位机 images/input（2026-08-24）
- NEU val：6 类各取 3 张 → neu_<class>_<n>.jpg（文件名自带类别，直观）
- SDX val：取 10 个不同前缀各 1 张 → sdx_<原名>.jpg
- 清掉旧的 bus.jpg / random.jpg（源在 tests/test_images，可随时恢复）
"""
import io
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deploy_to_nano as d

NEU_VAL = r"D:\RK3588&Orin Nano\ORIN NANO\Model Training\datasets\NEU\images\val"
SDX_VAL = r"D:\RK3588&Orin Nano\ORIN NANO\Model Training\datasets\SteelDefectX_YOLO\val\images"
STAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_images_staging")

NEU_CLASSES = ["crazing", "inclusion", "patches", "pitted_surface",
               "rolled-in_scale", "scratches"]
NEU_PER_CLASS = 3
SDX_MAX = 10


def collect():
    if os.path.isdir(STAGE):
        shutil.rmtree(STAGE)
    os.makedirs(STAGE)
    picked = []

    # NEU：每类取前 N 张（文件名按类别排序）
    for cls in NEU_CLASSES:
        files = sorted(f for f in os.listdir(NEU_VAL) if f.startswith(cls + "_"))
        for i, f in enumerate(files[:NEU_PER_CLASS]):
            dst = f"neu_{cls}_{i + 1:02d}.jpg"
            shutil.copyfile(os.path.join(NEU_VAL, f), os.path.join(STAGE, dst))
            picked.append((dst, f))

    # SDX：不同前缀各 1 张（保留原名便于对照 25 类）
    prefix_map = {}
    for f in sorted(os.listdir(SDX_VAL)):
        pre = f.split("_")[0]
        if pre not in prefix_map:
            prefix_map[pre] = f
        if len(prefix_map) >= SDX_MAX:
            break
    for pre, f in sorted(prefix_map.items()):
        dst = f"sdx_{f}"
        shutil.copyfile(os.path.join(SDX_VAL, f), os.path.join(STAGE, dst))
        picked.append((dst, f))

    print(f"本地暂存 {len(picked)} 张 -> {STAGE}")
    for dst, src in picked:
        print(f"  {dst}  <-  {src}")
    return picked


def upload():
    c = d.ssh_connect()
    # 清空旧测试图（源在 tests/test_images 可恢复）
    rc, out, err = d.run(c, f"rm -f {d.IMAGE_DIR}/bus.jpg {d.IMAGE_DIR}/random.jpg && "
                            f"ls -la {d.IMAGE_DIR}/ | wc -l")
    print("[清空旧图]", out or err)

    sftp = c.open_sftp()
    files = sorted(os.listdir(STAGE))
    for f in files:
        sftp.put(os.path.join(STAGE, f), f"{d.IMAGE_DIR}/{f}")
    sftp.close()
    print(f"上传完成 {len(files)} 张")

    rc, out, err = d.run(c, f"ls -la {d.IMAGE_DIR}/")
    print(out)
    c.close()


if __name__ == "__main__":
    collect()
    if len(sys.argv) > 1 and sys.argv[1] == "upload":
        upload()
