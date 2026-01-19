#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的电机测试脚本 - 用于快速诊断串口和电机问题
"""

import sys
import time

def test_imports():
    """测试必要的库是否已安装"""
    print("=" * 60)
    print("步骤 1/4: 检查 Python 依赖")
    print("=" * 60)

    missing = []

    # 测试 serial
    try:
        import serial
        print(f"✅ pyserial 已安装 (版本: {serial.VERSION})")
    except ImportError:
        print(f"❌ pyserial 未安装")
        missing.append("pyserial")

    # 测试 struct
    try:
        import struct
        print(f"✅ struct 模块可用")
    except ImportError:
        print(f"❌ struct 模块不可用")
        missing.append("struct")

    if missing:
        print(f"\n请安装缺失的库:")
        print(f"  pip3 install {' '.join(missing)}")
        return False

    print()
    return True


def test_serial_ports():
    """测试可用的串口"""
    print("=" * 60)
    print("步骤 2/4: 检查可用的串口")
    print("=" * 60)

    import os
    tty_ports = []

    # 查找所有 ttyS* 设备
    for i in range(0, 10):
        port = f"/dev/ttyS{i}"
        if os.path.exists(port):
            tty_ports.append(port)

    if not tty_ports:
        print("❌ 未找到任何 /dev/ttyS* 设备")
        print("\n请检查:")
        print("1. 串口驱动是否加载")
        print("2. 设备树是否正确配置")
        return False

    print(f"✅ 找到 {len(tty_ports)} 个串口设备:")
    for port in tty_ports:
        print(f"   - {port}")
    print()

    return tty_ports


def test_serial_connection(port):
    """测试串口连接"""
    print("=" * 60)
    print(f"步骤 3/4: 测试串口连接 ({port})")
    print("=" * 60)

    import serial

    try:
        ser = serial.Serial(
            port=port,
            baudrate=115200,
            timeout=1,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE
        )
        print(f"✅ 串口已打开: {ser.name} @ {ser.baudrate}")
        print(f"   字节大小: {ser.bytesize}")
        print(f"   停止位: {ser.stopbits}")
        print(f"   校验位: {ser.parity}")

        # 清空缓冲区
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        ser.close()
        print(f"✅ 串口测试成功")
        print()
        return True

    except serial.SerialException as e:
        print(f"❌ 串口连接失败: {e}")
        print(f"\n可能的原因:")
        print(f"1. 串口被其他程序占用")
        print(f"2. 权限不足（尝试: sudo chmod 666 {port}）")
        print(f"3. 串口硬件未连接")
        print()
        return False
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        print()
        return False


def test_send_command(port):
    """测试发送电机命令"""
    print("=" * 60)
    print(f"步骤 4/4: 测试发送电机命令 ({port})")
    print("=" * 60)

    import serial
    import struct

    try:
        ser = serial.Serial(
            port=port,
            baudrate=115200,
            timeout=1
        )
        print(f"✅ 串口已打开")

        # 构造零位命令
        yaw = 0.0
        pitch = 0.0
        yaw_bytes = struct.pack('<f', yaw)
        pitch_bytes = struct.pack('<f', pitch)

        header = 0xAA
        cmd = 0x10  # 绝对位置命令
        payload = yaw_bytes + pitch_bytes
        checksum = (cmd + sum(payload)) % 256
        packet = bytes([header, 0x0A, cmd]) + payload + bytes([checksum])

        # 发送命令
        ser.write(packet)
        print(f"✅ 命令已发送")
        print(f"   数据包: {packet.hex().upper()}")
        print(f"   长度: {len(packet)} 字节")
        print(f"   命令: 0x{cmd:02X} (零位)")
        print(f"   Yaw: {yaw:+.3f} rad")
        print(f"   Pitch: {pitch:+.3f} rad")
        print(f"   校验和: 0x{checksum:02X}")

        ser.close()
        print()
        print("=" * 60)
        print("🎉 所有测试通过！电机控制应该可以正常工作")
        print("=" * 60)
        print()
        print("下一步:")
        print(f"1. 运行姿态检测: python3 pose_head_tracking.py --source 36")
        print(f"2. 运行电机控制: python3 bidirectional_tracking.py --serial {port}")
        print()

        return True

    except Exception as e:
        print(f"❌ 发送命令失败: {e}")
        print()
        return False


def main():
    print()
    print("=" * 60)
    print("电机控制诊断工具")
    print("=" * 60)
    print()

    # 步骤1: 检查依赖
    if not test_imports():
        print("\n❌ 依赖检查失败，请先安装必要的库")
        sys.exit(1)

    # 步骤2: 检查串口
    ports = test_serial_ports()
    if not ports:
        print("\n❌ 未找到可用串口")
        sys.exit(1)

    # 选择要测试的串口
    # 默认使用 /dev/ttyS8
    test_port = '/dev/ttyS8'

    # 如果 ttyS8 不存在，使用第一个可用的串口
    if test_port not in ports:
        test_port = ports[0]
        print(f"⚠️  /dev/ttyS8 不存在，使用 {test_port} 代替\n")

    # 步骤3: 测试串口连接
    if not test_serial_connection(test_port):
        print(f"\n❌ 串口 {test_port} 连接失败")
        print("\n尝试其他串口:")
        for port in ports:
            if port != test_port:
                print(f"  python3 {sys.argv[0]} --port {port}")
        sys.exit(1)

    # 步骤4: 测试发送命令
    if not test_send_command(test_port):
        print(f"\n❌ 命令发送失败")
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        sys.exit(0)
