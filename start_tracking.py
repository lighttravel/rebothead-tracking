#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人体头部跟踪系统启动脚本（Python版本）
跨平台兼容，避免行尾符问题
"""

import os
import sys
import signal
import subprocess
import time
from pathlib import Path

# 颜色输出
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'

# 配置参数
CONFIG = {
    'CAMERA_SOURCE': os.getenv('CAMERA_SOURCE', '36'),
    'DETECT_FPS': os.getenv('DETECT_FPS', '8'),
    'STREAM_FPS': os.getenv('STREAM_FPS', '30'),
    'HTTP_PORT': os.getenv('HTTP_PORT', '8080'),
    'COORDS_FILE': os.getenv('COORDS_FILE', 'tracker_coords.json'),
    'SERIAL_PORT': os.getenv('SERIAL_PORT', '/dev/ttyS8'),
    'BAUDRATE': os.getenv('BAUDRATE', '115200'),
    'MAX_LOST_FRAMES': os.getenv('MAX_LOST_FRAMES', '10'),
}

def print_header():
    print(f"{Colors.GREEN}{'='*40}{Colors.NC}")
    print(f"{Colors.GREEN}   人体头部跟踪系统启动脚本{Colors.NC}")
    print(f"{Colors.GREEN}{'='*40}{Colors.NC}")
    print()

def print_config():
    print(f"{Colors.BLUE}配置参数:{Colors.NC}")
    print(f"  摄像头: /dev/video{CONFIG['CAMERA_SOURCE']}")
    print(f"  检测帧率: {CONFIG['DETECT_FPS']} fps")
    print(f"  推流帧率: {CONFIG['STREAM_FPS']} fps")
    print(f"  HTTP端口: {CONFIG['HTTP_PORT']}")
    print(f"  坐标文件: {CONFIG['COORDS_FILE']}")
    print(f"  串口: {CONFIG['SERIAL_PORT']} @ {CONFIG['BAUDRATE']}")
    print(f"  丢失阈值: {CONFIG['MAX_LOST_FRAMES']} 帧")
    print()

def check_files():
    """检查必需文件"""
    if not Path('person_tracking.py').exists():
        print(f"{Colors.RED}错误: 找不到 person_tracking.py{Colors.NC}")
        sys.exit(1)

    if not Path('horizontal_tracking.py').exists():
        print(f"{Colors.RED}错误: 找不到 horizontal_tracking.py{Colors.NC}")
        sys.exit(1)

    if not Path('yolov8.rknn').exists():
        print(f"{Colors.YELLOW}警告: 找不到 yolov8.rknn 模型文件{Colors.NC}")
        print()

    # 创建日志目录
    Path('logs').mkdir(exist_ok=True)

def get_local_ip():
    """获取本机IP"""
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "localhost"

def main():
    print_header()
    print_config()
    check_files()

    # 获取本机IP
    local_ip = get_local_ip()

    print(f"{Colors.GREEN}{'='*40}{Colors.NC}")
    print(f"{Colors.GREEN}启动中...{Colors.NC}")
    print(f"{Colors.GREEN}{'='*40}{Colors.NC}")
    print()
    print(f"{Colors.BLUE}视频流地址:{Colors.NC} http://{local_ip}:{CONFIG['HTTP_PORT']}/stream")
    print(f"{Colors.YELLOW}按 Ctrl+C 停止所有程序{Colors.NC}")
    print()

    processes = []

    # 清理函数
    def cleanup(signum, frame):
        print()
        print(f"{Colors.YELLOW}正在停止所有程序...{Colors.NC}")
        for proc in processes:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except:
                proc.kill()
        print(f"{Colors.GREEN}所有程序已停止{Colors.NC}")
        sys.exit(0)

    # 捕获退出信号
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        # 启动检测程序
        print(f"{Colors.GREEN}[1/2] 启动 YOLOv8 检测程序...{Colors.NC}")
        detect_cmd = [
            'python3', 'person_tracking.py',
            '--source', CONFIG['CAMERA_SOURCE'],
            '--detect-fps', CONFIG['DETECT_FPS'],
            '--stream-fps', CONFIG['STREAM_FPS'],
            '--port', CONFIG['HTTP_PORT'],
            '--output-coords', CONFIG['COORDS_FILE'],
        ]

        detect_log = open('logs/detection.log', 'w')
        detect_proc = subprocess.Popen(
            detect_cmd,
            stdout=detect_log,
            stderr=subprocess.STDOUT
        )
        processes.append(detect_proc)
        print(f"   进程 PID: {detect_proc.pid}")

        # 等待坐标文件生成
        print(f"{Colors.YELLOW}等待坐标文件生成...{Colors.NC}")
        time.sleep(2)

        # 启动电机控制程序
        print(f"{Colors.GREEN}[2/2] 启动电机控制程序...{Colors.NC}")
        motor_cmd = [
            'python3', 'horizontal_tracking.py',
            '--coords', CONFIG['COORDS_FILE'],
            '--serial', CONFIG['SERIAL_PORT'],
            '--baudrate', CONFIG['BAUDRATE'],
            '--max-lost-frames', CONFIG['MAX_LOST_FRAMES'],
        ]

        motor_log = open('logs/motor_control.log', 'w')
        motor_proc = subprocess.Popen(
            motor_cmd,
            stdout=motor_log,
            stderr=subprocess.STDOUT
        )
        processes.append(motor_proc)
        print(f"   进程 PID: {motor_proc.pid}")

        print()
        print(f"{Colors.GREEN}{'='*40}{Colors.NC}")
        print(f"{Colors.GREEN}系统运行中！{Colors.NC}")
        print(f"{Colors.GREEN}{'='*40}{Colors.NC}")
        print()
        print(f"{Colors.BLUE}进程信息:{Colors.NC}")
        print(f"  检测进程 PID: {detect_proc.pid}")
        print(f"  电机控制 PID: {motor_proc.pid}")
        print()
        print(f"{Colors.BLUE}日志文件:{Colors.NC}")
        print(f"  检测日志: logs/detection.log")
        print(f"  电机控制日志: logs/motor_control.log")
        print()
        print(f"{Colors.BLUE}实时查看日志:{Colors.NC}")
        print(f"  检测: tail -f logs/detection.log")
        print(f"  电机控制: tail -f logs/motor_control.log")
        print()
        print(f"{Colors.YELLOW}按 Ctrl+C 停止所有程序{Colors.NC}")
        print()

        # 等待所有进程
        for proc in processes:
            proc.wait()

    except Exception as e:
        print(f"{Colors.RED}错误: {e}{Colors.NC}")
        cleanup(None, None)

if __name__ == '__main__':
    main()
