#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电机控制器
负责通过串口发送电机控制命令
与 test_motor_rk3576_native.py 中的发送方式完全一致
"""

import struct
import serial
import time


class MotorController:
    """电机控制器"""

    def __init__(self, serial_port='/dev/ttyS8', baudrate=115200):
        """
        初始化电机控制器

        参数:
            serial_port: 串口设备
            baudrate: 波特率
        """
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.ser = None

    def connect(self):
        """连接串口"""
        try:
            self.ser = serial.Serial(
                port=self.serial_port,
                baudrate=self.baudrate,
                timeout=1,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )

            # 清空缓冲区
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()

            print(f"✅ 串口已连接: {self.ser.name} @ {self.baudrate}")
            return True

        except serial.SerialException as e:
            print(f"❌ 串口连接失败: {e}")
            print(f"\n请检查:")
            print(f"1. TX/RX/GND 是否正确连接:")
            print(f"   RK3576 UART2_TX ──▶ STM32 UART7_RX")
            print(f"   RK3576 UART2_RX ──◀ STM32 UART7_TX")
            print(f"   GND ──────────────▶ GND")
            print()
            print(f"2. STM32 是否已烧录固件并正在运行")
            print(f"3. 尝试其他串口: /dev/ttyS1, /dev/ttyS3, /dev/ttyS4")
            return False

    def send_motor_command(self, yaw, pitch):
        """发送电机控制命令（绝对位置）"""
        if not self.ser or not self.ser.is_open:
            print("❌ 串口未打开")
            return False

        try:
            yaw_bytes = struct.pack('<f', yaw)
            pitch_bytes = struct.pack('<f', pitch)

            header = 0xAA
            cmd = 0x10  # PYCTL_CMD_SET_MOTORS_21_22 (绝对位置)
            payload = yaw_bytes + pitch_bytes
            checksum = (cmd + sum(payload)) % 256
            packet = bytes([header, 0x0A, cmd]) + payload + bytes([checksum])

            self.ser.write(packet)
            print(f"已发送绝对位置: 电机21={yaw:+.3f}rad, 电机22={pitch:+.3f}rad")
            print(f"数据包: {packet.hex().upper()}")
            return True

        except Exception as e:
            print(f"❌ 发送失败: {e}")
            return False

    def send_motor_delta(self, yaw_delta, pitch_delta):
        """发送电机控制命令（增量）"""
        if not self.ser or not self.ser.is_open:
            print("❌ 串口未打开")
            return False

        try:
            yaw_bytes = struct.pack('<f', yaw_delta)
            pitch_bytes = struct.pack('<f', pitch_delta)

            header = 0xAA
            cmd = 0x11  # PYCTL_CMD_SET_MOTORS_21_22_DELTA (增量)
            payload = yaw_bytes + pitch_bytes
            checksum = (cmd + sum(payload)) % 256
            packet = bytes([header, 0x0A, cmd]) + payload + bytes([checksum])

            # 🔍 调试输出
            print(f"🔍 [MOTOR] 发送增量命令: yaw={yaw_delta:+.4f}rad, pitch={pitch_delta:+.4f}rad")
            print(f"🔍 [MOTOR] 数据包: {packet.hex().upper()} ({len(packet)} 字节)")

            self.ser.write(packet)
            print(f"✅ 已发送增量: 电机21={yaw_delta:+.3f}rad, 电机22={pitch_delta:+.3f}rad")
            print(f"数据包: {packet.hex().upper()}")
            return True

        except Exception as e:
            print(f"❌ 发送失败: {e}")
            return False

    def reset_to_zero(self):
        """回到零位"""
        return self.send_motor_command(0.0, 0.0)

    def disconnect(self):
        """断开串口"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("串口已关闭")


# 测试代码（与 test_motor_rk3576_native.py 相同的测试）
def test():
    """测试电机控制"""
    print("=" * 60)
    print("电机控制测试")
    print("=" * 60)
    print()

    # 创建控制器
    controller = MotorController(serial_port='/dev/ttyS8')

    # 连接串口
    if not controller.connect():
        return

    try:
        # 测试1: 电机转动到 +0.5 弧度
        print("[测试 1/6] 电机21 转动到 +0.5 弧度")
        controller.send_motor_command(0.5, 0.0)
        time.sleep(2)  # 等待电机运动

        # 测试2: 增量控制
        print()
        print("[测试 2/6] 电机21 再转动 +0.2 弧度（增量）")
        controller.send_motor_delta(0.2, 0.0)
        time.sleep(2)

        # 测试3: 回到零位
        print()
        print("[测试 3/6] 回到零位")
        controller.reset_to_zero()
        time.sleep(2)

        # 测试4: 电机22转动到 +0.3 弧度
        print()
        print("[测试 4/6] 电机22 转动到 +0.3 弧度")
        controller.send_motor_command(0.0, 0.3)
        time.sleep(2)  # 等待电机运动

        # 测试5: 电机22增量控制
        print()
        print("[测试 5/6] 电机22 再转动 -0.1 弧度（增量）")
        controller.send_motor_delta(0.0, -0.1)
        time.sleep(2)

        # 测试6: 两个电机都回到零位
        print()
        print("[测试 6/6] 两个电机都回到零位")
        controller.reset_to_zero()
        time.sleep(2)

        print()
        print("=" * 60)
        print("测试完成！")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    finally:
        controller.disconnect()


if __name__ == '__main__':
    test()
