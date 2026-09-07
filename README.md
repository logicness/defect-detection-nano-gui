# 缺陷检测下位机 (Jetson Orin Nano) GUI 与推理服务

工业表面缺陷检测项目的下位机部分，运行于 NVIDIA Jetson Orin Nano Super，
包含本地 GUI 应用与基于 TensorRT 的实时推理服务。

## 目录结构
- `nano_gui_app/`：下位机本地 GUI 应用
- `nano_src/`：推理服务、部署脚本与通信协议实现

## 凭据说明（重要）
部署脚本（`nano_src/deploy/`）通过环境变量读取板端 SSH 凭据，**不要硬编码密码**：

- `NANO_USER`：板端用户名（默认 `nvidia`，即 Jetson 默认用户）
- `NANO_PASS`：SSH 密码（**留空，必须从环境变量传入**）
- `NANO_SUDO`：sudo 密码（**留空，必须从环境变量传入**）

示例：

```bash
export NANO_PASS="你的板端密码"
export NANO_SUDO="你的sudo密码"
python nano_src/deploy/deploy_to_nano.py
```

## 许可
MIT

## 网络占位符说明
- `<LAN_IP>`：其它局域网地址占位符。
部署前请按你的实际网络环境替换为真实 IP。

## 网络占位符说明
- `<NANO_LAN_IP>`：下位机（Jetson Orin Nano）的局域网 IP 占位符，部署前请替换为真实 IP。
- `<PLC_LAN_IP>`：PLC 的局域网 IP 占位符，部署前请替换为真实 IP。
- `<LAN_IP>`：其它局域网地址占位符。
