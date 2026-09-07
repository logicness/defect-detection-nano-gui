#!/usr/bin/env bash
# =============================================================
# 下位机 MVS SDK 安装脚本（相机接入规划 P1-3，2026-08-27）
# 用法：
#   bash install_mvs_sdk.sh <MVS安装包路径>.tar.gz
# 例：
#   bash install_mvs_sdk.sh /home/nvidia/sdk/MVS-4.x_linux_aarch64.tar.gz
#
# 功能：
#   1. 解压安装包到临时目录
#   2. 执行 SDK 自带安装脚本（通常 setup.sh / install.sh）
#   3. 将用户加入 video 组（USB 相机权限）
#   4. 清理临时目录，验证 /opt/MVS/bin 存在
# 幂等：已安装则跳过安装步骤，仅验证。
# 安全：不触碰 defect-infer.service；不改动系统 Python 环境。
# =============================================================
set -euo pipefail

SDK_TAR="${1:-}"
MVS_BIN="/opt/MVS/bin"

usage() {
  echo "用法: bash $0 <MVS安装包>" >&2
  echo "  MVS安装包如 MVS-4.x_linux_aarch64.tar.gz（海康官网下载，Nano 无外网需 PC 传过来）" >&2
  exit 1
}

if [ -z "$SDK_TAR" ] || [ ! -f "$SDK_TAR" ]; then
  echo "[install_mvs_sdk] 找不到安装包: ${SDK_TAR:-<未提供>}" >&2
  usage
fi

if [ -x "$MVS_BIN" ]; then
  echo "[install_mvs_sdk] SDK 已安装，跳过安装步骤。"
  ls -la "$MVS_BIN" | head -20
  exit 0
fi

WORK=$(mktemp -d /tmp/mvs_sdk.XXXXXX)
trap 'rm -rf "$WORK"' EXIT

echo "[install_mvs_sdk] 解压到 $WORK ..."
tar -xzf "$SDK_TAR" -C "$WORK"

# 定位 SDK 安装脚本（常见：setup.sh / install.sh / setup 可执行文件）
SETUP=$(find "$WORK" -maxdepth 3 \( -name "setup.sh" -o -name "install.sh" \) | head -1)
if [ -z "$SETUP" ]; then
  SETUP=$(find "$WORK" -maxdepth 3 -type f -perm -111 | head -1)
fi

if [ -z "$SETUP" ]; then
  echo "[install_mvs_sdk] 未找到安装脚本，请检查包内容。" >&2
  ls -la "$WORK" >&2
  exit 1
fi

echo "[install_mvs_sdk] 执行安装脚本: $SETUP"
cd "$(dirname "$SETUP")"
# 海康 MVS setup.sh 需要 root 权限；脚本内部会引导 + 拷贝到 /opt/MVS
sudo bash "$SETUP"

# USB 相机权限：加入 video 组
sudo usermod -aG video "$USER"

echo "[install_mvs_sdk] 权限配置完成。"

# 验证
if [ -x "$MVS_BIN" ]; then
  echo "[install_mvs_sdk] ✅ 安装成功: $MVS_BIN"
  ls -la "$MVS_BIN" | head -20
  # 提示：虚拟相机/WSL 等场景下 /opt/MVS/bin 亦可由 SDK 安装到其他位置，失败时请检查
  exit 0
else
  echo "[install_mvs_sdk] ❌ 未在 $MVS_BIN 找到可执行文件；请检查安装日志。" >&2
  exit 1
fi
