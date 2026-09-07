# -*- coding: utf-8 -*-
"""扩充板端测试图库（2026-08-24）：每个数据集都放一点
- Universal_Metal(MVIT 产线 10 类) val：后 10 类各 2 张 → um_<class>_<n>.jpg（当前生产模型训练数据）
- SD10 val：6 个代表性类各 1-2 张 → sd10_<class>_<n>.jpg
- guangdong_detect(铝型材) val：4 张 → gd_<n>.jpg（未见域泛化测试）
按 labels txt 首列 class id 归类。
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deploy_to_nano as d

UM = r"D:\RK3588&Orin Nano\ORIN NANO\Model Training\datasets\Universal_Metal"
SD10 = r"D:\RK3588&Orin Nano\ORIN NANO\Model Training\datasets\SD10"
GD = r"D:\RK3588&Orin Nano\ORIN NANO\dataset\guangdong_detect\images"
STAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_images_staging")


def load_names(yaml_path, key="names"):
    import re
    with open(yaml_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    m = re.search(r"names\s*:\s*(\[.*?\])", text, re.S)
    if not m:
        return []
    return [x.strip().strip("'\"") for x in m.group(1).strip("[]").split(",")]


def pick_by_class(img_dir, lbl_dir, class_ids, per=2, prefer_sub=None):
    """按 labels 首列 class id 归类选图。返回 [(class_id, img_path)]"""
    picked = []
    counts = {}
    files = sorted(os.listdir(lbl_dir))
    for f in files:
        if not f.endswith(".txt"):
            continue
        base = os.path.splitext(f)[0]
        try:
            with open(os.path.join(lbl_dir, f), "r") as lf:
                line = lf.readline().strip()
            cid = int(line.split()[0]) if line else -1
        except Exception:
            continue
        if cid not in class_ids:
            continue
        # 优先选含指定前缀的文件名（区分产线）
        if prefer_sub and prefer_sub not in base:
            continue
        img = os.path.join(img_dir, base + ".jpg")
        if not os.path.isfile(img):
            img = os.path.join(img_dir, base + ".png")
        if not os.path.isfile(img):
            continue
        counts[cid] = counts.get(cid, 0) + 1
        if counts[cid] <= per:
            picked.append((cid, img))
        if all(counts.get(c, 0) >= per for c in class_ids):
            break
    return picked


def collect():
    if os.path.isdir(STAGE):
        shutil.rmtree(STAGE)
    os.makedirs(STAGE)
    total = 0

    # 1) Universal_Metal：MVIT 产线 10 类（class 25-34）各 2 张
    um_names = load_names(os.path.join(UM, "data.yaml"))
    um_picked = pick_by_class(os.path.join(UM, "val", "images"),
                              os.path.join(UM, "val", "labels"),
                              list(range(25, 35)), per=2)
    for cid, img in um_picked:
        cls = um_names[cid] if cid < len(um_names) else f"cls{cid}"
        n = sum(1 for _c, _i in um_picked[:um_picked.index((cid, img))]
                if _c == cid) + 1
        dst = f"um_{cls}_{n:02d}.jpg"
        shutil.copyfile(img, os.path.join(STAGE, dst))
        total += 1
    print(f"UM(MVIT 产线) 取 {len(um_picked)} 张")

    # 2) SD10：6 个类各 2 张（crazing/inclusion/patches/scratches/blowhole/break）
    sd_names = load_names(os.path.join(SD10, "SD10.yaml"))
    sd_ids = [0, 1, 2, 5, 6, 7] if len(sd_names) > 7 else list(range(6))
    sd_picked = pick_by_class(os.path.join(SD10, "images", "val"),
                              os.path.join(SD10, "labels", "val"),
                              sd_ids, per=2)
    for cid, img in sd_picked:
        cls = sd_names[cid] if cid < len(sd_names) else f"cls{cid}"
        n = sum(1 for _c, _i in sd_picked[:sd_picked.index((cid, img))]
                if _c == cid) + 1
        dst = f"sd10_{cls}_{n:02d}.jpg"
        shutil.copyfile(img, os.path.join(STAGE, dst))
        total += 1
    print(f"SD10 取 {len(sd_picked)} 张")

    # 3) guangdong_detect（铝型材）：取 4 张
    gd_files = sorted(os.listdir(GD))
    gd_files = [f for f in gd_files if f.lower().endswith((".jpg", ".png"))][:4]
    for i, f in enumerate(gd_files, 1):
        dst = f"gd_{i:02d}.jpg"
        shutil.copyfile(os.path.join(GD, f), os.path.join(STAGE, dst))
        total += 1
    print(f"guangdong(铝型材) 取 {len(gd_files)} 张")

    print(f"\n本地暂存共 {total} 张 -> {STAGE}")
    for f in sorted(os.listdir(STAGE)):
        print(f"  {f}")


def upload():
    c = d.ssh_connect()
    sftp = c.open_sftp()
    files = sorted(os.listdir(STAGE))
    for f in files:
        sftp.put(os.path.join(STAGE, f), f"{d.IMAGE_DIR}/{f}")
    sftp.close()
    rc, out, err = d.run(c, f"ls {d.IMAGE_DIR}/ | wc -l && ls {d.IMAGE_DIR}/")
    print(f"上传完成 {len(files)} 张；目录现有文件数:")
    print(out)
    c.close()


if __name__ == "__main__":
    collect()
    if len(sys.argv) > 1 and sys.argv[1] == "upload":
        upload()
