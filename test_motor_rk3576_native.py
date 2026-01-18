#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RK3576 原生 UART → STM32 电机控制
使用 /dev/ttyS2 (或其他可用的 ttyS*)
"""
import struct
import serial
import sys
import time

# 根据扫描结果，选择一个可用的串口
# 可用: /dev/ttyS1, /dev/ttyS2, /dev/ttyS3, /dev/ttyS4, /dev/ttyS7, /dev/ttyS8
SERIAL_PORT = '/dev/ttyS8'  # ← RK3576 J33 连接的串口

def send_motor_command(ser, yaw, pitch):
    """发送电机控制命令（绝对位置）"""
    yaw_bytes = struct.pack('<f', yaw)
    pitch_bytes = struct.pack('<f', pitch)

    header = 0xAA
    cmd = 0x10  # PYCTL_CMD_SET_MOTORS_21_22 (绝对位置)
    payload = yaw_bytes + pitch_bytes
    checksum = (cmd + sum(payload)) % 256
    packet = bytes([header, 0x0A, cmd]) + payload + bytes([checksum])

    ser.write(packet)
    print(f"已发送: 电机21={yaw:.3f}rad, 电机22={pitch:.3f}rad")
    print(f"数据包: {packet.hex().upper()}")

def send_motor_delta(ser, yaw_delta, pitch_delta):
    """发送电机控制命令（增量）"""
    yaw_bytes = struct.pack('<f', yaw_delta)
    pitch_bytes = struct.pack('<f', pitch_delta)

    header = 0xAA
    cmd = 0x11  # PYCTL_CMD_SET_MOTORS_21_22_DELTA (增量)
    payload = yaw_bytes + pitch_bytes
    checksum = (cmd + sum(payload)) % 256
    packet = bytes([header, 0x0A, cmd]) + payload + bytes([checksum])

    ser.write(packet)
    print(f"已发送增量: 电机21={yaw_delta:+.3f}rad, 电机22={pitch_delta:+.3f}rad")
    print(f"数据包: {packet.hex().upper()}")

def main():
    print("=" * 60)
    print("RK3576 → STM32 电机控制测试")
    print("=" * 60)
    print(f"串口: {SERIAL_PORT}")
    print()

    try:
        # 打开串口
        ser = serial.Serial(
            port=SERIAL_PORT,
            baudrate=115200,
            timeout=1,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE
        )

        # 清空缓冲区
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        print(f"✅ 串口已打开: {ser.name}")
        print(f"   波特率: {ser.baudrate}")
        print()

        # 测试1: 电机转动到 +0.5 弧度
        print("[测试 1/6] 电机21 转动到 +0.5 弧度")
        send_motor_command(ser, 0.5, 0.0)
        time.sleep(2)  # 等待电机运动

        # 测试2: 增量控制
        print()
        print("[测试 2/6] 电机21 再转动 +0.2 弧度（增量）")
        send_motor_delta(ser, 0.2, 0.0)
        time.sleep(2)

        # 测试3: 回到零位
        print()
        print("[测试 3/6] 回到零位")
        send_motor_command(ser, 0.0, 0.0)
        time.sleep(2)

        # 测试4: 电机22转动到 +0.3 弧度
        print()
        print("[测试 4/6] 电机22 转动到 +0.3 弧度")
        send_motor_command(ser, 0.0, 0.3)
        time.sleep(2)  # 等待电机运动

        # 测试5: 电机22增量控制
        print()
        print("[测试 5/6] 电机22 再转动 -0.1 弧度（增量）")
        send_motor_delta(ser, 0.0, -0.1)
        time.sleep(2)

        # 测试6: 两个电机都回到零位
        print()
        print("[测试 6/6] 两个电机都回到零位")
        send_motor_command(ser, 0.0, 0.0)
        time.sleep(2)

        ser.close()
        print()
        print("=" * 60)
        print("测试完成！")
        print("=" * 60)

    except serial.SerialException as e:
        print(f"\n❌ 串口错误: {e}")
        print("\n请检查:")
        print("1. TX/RX/GND 是否正确连接:")
        print("   RK3576 UART2_TX ──▶ STM32 UART7_RX")
        print("   RK3576 UART2_RX ──◀ STM32 UART7_TX")
        print("   GND ──────────────▶ GND")
        print()
        print("2. STM32 是否已烧录固件并正在运行")
        print("3. 尝试其他串口: /dev/ttyS1, /dev/ttyS3, /dev/ttyS4")
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
