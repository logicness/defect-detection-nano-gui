#!/usr/bin/env bash
# =============================================================
# 缺陷检测工作站 GUI 启动脚本（HDMI/VNC 通用，全局单实例）
# v3 2026-09-01：
#   1) DISPLAY 跟随当前会话（${DISPLAY:-:0}），不再硬编码 :0
#      （修复 autostart(:0) 与桌面图标(:1) 硬编码导致的观感不一致）
#   2) 启动前等待推理服务 8888 就绪（最长 60s），避免开机竞态
#   3) 看门狗只拉崩溃（3 次），正常退出（rc=0）不重启
#   4) 配合 main.py 全局单实例锁（127.0.0.1:43129）：后启动实例
#      自动激活已有窗口并退出，任意时刻只有一个 GUI（不再双实例）
# 日志：crash.log（stderr/faulthandler）、run_gui.log（自拉记录）
# =============================================================
cd "$HOME/nano_gui/app" || exit 1
DISPLAY="${DISPLAY:-:0}"

# 等待推理服务就绪（bash /dev/tcp，最长 60s，5s 间隔）
_wait_infer() {
  for i in $(seq 1 12); do
    if (exec 3<>/dev/tcp/127.0.0.1/8888) 2>/dev/null; then
      exec 3>&- 2>/dev/null
      return 0
    fi
    sleep 5
  done
  return 1
}
if ! _wait_infer; then
  echo "[run_gui] 60s 内推理服务(8888)未就绪，仍尝试启动 GUI（GUI 有重连机制）" \
    >> "$HOME/nano_gui/run_gui.log"
fi

fail=0
# 2026-09-01：开启 core dump + stderr 落盘（faulthandler/Qt abort 现场可查）
ulimit -c unlimited 2>/dev/null
while [ "$fail" -lt 3 ]; do
  DISPLAY="$DISPLAY" "$HOME/nano_gui/.venv/bin/python" main.py 2>>"$HOME/nano_gui/crash.log"
  rc=$?
  if [ "$rc" -eq 0 ]; then
    # 用户正常关闭（单实例锁已释放）→ 不再拉起
    exit 0
  fi
  fail=$((fail + 1))
  echo "[run_gui] GUI 异常退出(rc=$rc)，第 ${fail}/3 次自拉..." >> "$HOME/nano_gui/run_gui.log"
  sleep 5
done
