#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志配置模块 - 统一日志格式和关键指标记录
==========================================

提供标准化的日志格式，便于：
1. 问题定位：统一时间戳、模块、级别格式
2. 性能分析：关键操作耗时记录
3. 长期监控：健康指标结构化输出

使用方式：
    from log_config import setup_logging, log_metric
    
    setup_logging(level=logging.INFO)
    log_metric("detect", 11.5, {"model": "yolov8s", "detections": 3})
"""

import logging
import time
import json
from functools import wraps
from typing import Optional, Dict, Any

# 日志格式常量
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 关键指标前缀（便于 grep 和解析）
METRIC_PREFIX = "[METRIC]"
PERF_PREFIX = "[PERF]"
HEALTH_PREFIX = "[HEALTH]"


def setup_logging(level: int = logging.INFO, 
                  log_file: Optional[str] = None,
                  module_name: str = "nano_server"):
    """配置全局日志
    
    Args:
        level: 日志级别
        log_file: 日志文件路径（可选）
        module_name: 模块名称
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 清除现有处理器
    root_logger.handlers.clear()
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # 文件处理器（可选）
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    logging.getLogger(module_name).info(f"日志系统初始化完成: level={logging.getLevelName(level)}")


def log_metric(name: str, value: float, tags: Optional[Dict[str, Any]] = None):
    """记录关键指标（结构化格式，便于后续分析）
    
    Args:
        name: 指标名称 (如 "detect_latency", "frame_count")
        value: 指标值
        tags: 标签字典 (如 {"model": "yolov8s", "result": "NG"})
    
    Example:
        log_metric("detect_latency", 11.5, {"model": "yolov8s", "detections": 3})
        # 输出: 2026-08-25 10:30:00 [INFO] nano_server: [METRIC] detect_latency=11.5 model=yolov8s detections=3
    """
    logger = logging.getLogger("nano_server")
    
    # 构建日志消息
    parts = [f"{METRIC_PREFIX} {name}={value}"]
    if tags:
        for k, v in tags.items():
            parts.append(f"{k}={v}")
    
    logger.info(" ".join(parts))


def log_performance(operation: str, duration_ms: float, 
                    success: bool = True, details: Optional[str] = None):
    """记录性能数据
    
    Args:
        operation: 操作名称 (如 "detect", "model_load", "image_read")
        duration_ms: 耗时（毫秒）
        success: 是否成功
        details: 附加详情
    """
    logger = logging.getLogger("nano_server")
    
    status = "OK" if success else "FAIL"
    msg = f"{PERF_PREFIX} {operation} {duration_ms:.1f}ms [{status}]"
    if details:
        msg += f" {details}"
    
    if success:
        logger.info(msg)
    else:
        logger.warning(msg)


def log_health(component: str, status: str, metrics: Dict[str, Any]):
    """记录健康状态
    
    Args:
        component: 组件名称 (如 "camera_sim", "gpu_sampler")
        status: 状态 (如 "running", "stopped", "error")
        metrics: 健康指标字典
    
    Example:
        log_health("camera_sim", "running", {
            "fps": 10,
            "total_frames": 1000,
            "error_count": 5,
            "avg_ms": 11.2
        })
    """
    logger = logging.getLogger("nano_server")
    
    # 构建结构化消息
    parts = [f"{HEALTH_PREFIX} component={component} status={status}"]
    for k, v in metrics.items():
        parts.append(f"{k}={v}")
    
    logger.info(" ".join(parts))


def log_error_with_context(logger: logging.Logger, error: Exception, 
                           context: str, **kwargs):
    """记录带上下文的错误
    
    Args:
        logger: 日志记录器
        error: 异常对象
        context: 错误上下文描述
        **kwargs: 额外上下文信息
    """
    parts = [f"{context}: {type(error).__name__}: {error}"]
    for k, v in kwargs.items():
        parts.append(f"{k}={v}")
    
    logger.error(" | ".join(parts), exc_info=True)


class PerformanceTimer:
    """性能计时器上下文管理器
    
    使用方式：
        with PerformanceTimer("detect") as timer:
            result = do_detect()
            timer.add_tag("detections", len(result))
    """
    
    def __init__(self, operation: str, auto_log: bool = True):
        self.operation = operation
        self.auto_log = auto_log
        self.start_time = 0.0
        self.end_time = 0.0
        self.tags = {}
        self.success = True
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        duration_ms = (self.end_time - self.start_time) * 1000
        
        if exc_type is not None:
            self.success = False
            self.tags["error"] = str(exc_val)
        
        if self.auto_log:
            log_performance(self.operation, duration_ms, self.success, 
                          " ".join(f"{k}={v}" for k, v in self.tags.items()))
        
        return False  # 不抑制异常
    
    def add_tag(self, key: str, value: Any):
        """添加标签"""
        self.tags[key] = value
        return self
    
    @property
    def duration_ms(self) -> float:
        """获取耗时（毫秒）"""
        if self.end_time == 0:
            return (time.perf_counter() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000


def log_function_call(operation: Optional[str] = None):
    """装饰器：自动记录函数调用耗时
    
    使用方式：
        @log_function_call("detect")
        def detect(image):
            ...
    """
    def decorator(func):
        func_name = operation or func.__name__
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            with PerformanceTimer(func_name) as timer:
                result = func(*args, **kwargs)
                return result
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with PerformanceTimer(func_name) as timer:
                result = await func(*args, **kwargs)
                return result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    
    return decorator


# 便捷导出
__all__ = [
    'setup_logging',
    'log_metric',
    'log_performance', 
    'log_health',
    'log_error_with_context',
    'PerformanceTimer',
    'log_function_call',
    'METRIC_PREFIX',
    'PERF_PREFIX',
    'HEALTH_PREFIX',
]
