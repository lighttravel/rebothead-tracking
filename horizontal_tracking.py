#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
水平方向左右跟踪算法（增量控制模式）
- 读取person_tracking.py输出的坐标数据
- 根据目标位置发送增量命令控制电机21（偏航轴）左右转动
- 使用 motor_controller.py 发送命令
- 中心区域：图像宽度的 3/8 (37.5%) 到 5/8 (62.5%)
"""

import json
import time
import argparse
from pathlib import Path
from motor_controller import MotorController


class HorizontalTracker:
    """水平方向跟踪控制器"""

    def __init__(self, serial_port='/dev/ttyS8', baudrate=115200,
                 center_zone_start=0.375, center_zone_end=0.625,
                 deadzone=0.02, max_delta=0.15, kp=0.5, max_lost_frames=10):
        """
        初始化跟踪控制器

        参数:
            serial_port: 串口设备
            baudrate: 波特率
            center_zone_start: 中心区域左边界（相对位置，0-1）
            center_zone_end: 中心区域右边界（相对位置，0-1）
            deadzone: 死区（在中心区域内时的死区范围）
            max_delta: 单次最大转动角度（弧度）
            kp: 比例系数（控制响应速度）
            max_lost_frames: 目标丢失多少帧后回到零位
        """
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.center_zone_start = center_zone_start  # 3/8 = 0.375
        self.center_zone_end = center_zone_end      # 5/8 = 0.625
        self.deadzone = deadzone
        self.max_delta = max_delta
        self.kp = kp

        # 目标丢失计数
        self.lost_count = 0
        self.max_lost_frames = max_lost_frames  # 连续丢失多少帧后回到零位

        # 创建电机控制器
        self.motor = MotorController(serial_port, baudrate)

        # 统计信息
        self.stats = {
            'total_updates': 0,
            'left_moves': 0,
            'right_moves': 0,
            'no_moves': 0,
            'lost_frames': 0,
            'resets_to_zero': 0,
            'errors': 0
        }

    def open_serial(self):
        """连接串口"""
        return self.motor.connect()

    def send_motor_delta(self, yaw_delta, pitch_delta):
        """发送电机控制命令（增量模式）"""
        success = self.motor.send_motor_delta(yaw_delta, pitch_delta)
        if not success:
            self.stats['errors'] += 1
        return success

    def send_motor_absolute(self, yaw, pitch):
        """发送电机控制命令（绝对位置模式）"""
        success = self.motor.send_motor_command(yaw, pitch)
        if not success:
            self.stats['errors'] += 1
        return success

    def calculate_motor_command(self, rel_x):
        """
        根据目标相对位置计算电机控制指令

        参数:
            rel_x: 目标中心点的X相对位置（-1到1，0为中心）
                    或者在JSON中使用的是position.relative.x

        返回:
            (should_move, delta_yaw, direction)
            should_move: 是否需要移动
            delta_yaw: 转动角度（弧度，正数向右，负数向左）
            direction: 方向描述 ('left', 'right', 'center')
        """
        # 将rel_x从[-1, 1]转换到[0, 1]
        # 如果已经是0-1范围则不用转换
        if -1 <= rel_x <= 1:
            normalized_x = (rel_x + 1) / 2  # 转换到0-1
        else:
            normalized_x = rel_x

        # 检查目标位置
        if normalized_x < self.center_zone_start - self.deadzone:
            # 目标在左边，需要向左转（反转符号）
            distance = self.center_zone_start - normalized_x
            delta_yaw = distance * self.kp  # 正数向左转（修正）
            delta_yaw = min(self.max_delta, delta_yaw)  # 限制最大角度
            return True, delta_yaw, 'left'

        elif normalized_x > self.center_zone_end + self.deadzone:
            # 目标在右边，需要向右转（反转符号）
            distance = normalized_x - self.center_zone_end
            delta_yaw = -distance * self.kp  # 负数向右转（修正）
            delta_yaw = max(-self.max_delta, delta_yaw)  # 限制最大角度
            return True, delta_yaw, 'right'

        else:
            # 目标在中心区域，不需要移动
            return False, 0.0, 'center'

    def update(self, coords_file):
        """
        读取坐标文件并更新电机位置

        参数:
            coords_file: 坐标JSON文件路径
        """
        try:
            # 读取坐标文件
            with open(coords_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 获取主目标
            primary = data.get('primary_target')
            if not primary:
                # 目标丢失
                self.lost_count += 1
                self.stats['lost_frames'] += 1

                if self.lost_count <= self.max_lost_frames:
                    print(f"⚠️  目标丢失 ({self.lost_count}/{self.max_lost_frames})")
                elif self.lost_count == self.max_lost_frames + 1:
                    # 连续丢失max_lost_frames帧，回到零位
                    print(f"🔄 目标丢失超过{self.max_lost_frames}帧，回到零位")
                    success = self.send_motor_absolute(0.0, 0.0)
                    if success:
                        self.stats['resets_to_zero'] += 1
                return

            # 目标重新出现，重置丢失计数
            if self.lost_count > 0:
                print(f"✅ 目标重新出现 (丢失了{self.lost_count}帧)")
                self.lost_count = 0

            # 获取相对位置
            rel_x = primary['position']['relative']['x']
            confidence = primary['confidence']
            target_id = primary['id']

            # 计算控制指令
            should_move, delta_yaw, direction = self.calculate_motor_command(rel_x)

            # 更新统计
            self.stats['total_updates'] += 1

            if should_move:
                # 发送增量命令
                success = self.send_motor_delta(delta_yaw, 0.0)

                if success:
                    if direction == 'left':
                        self.stats['left_moves'] += 1
                        print(f"🔴 目标{target_id}在左侧 | rel_x={rel_x:.3f} | 向左转 {abs(delta_yaw):.3f}rad")
                    else:
                        self.stats['right_moves'] += 1
                        print(f"🔵 目标{target_id}在右侧 | rel_x={rel_x:.3f} | 向右转 {abs(delta_yaw):.3f}rad")
                else:
                    print(f"❌ 发送失败")
            else:
                self.stats['no_moves'] += 1
                # 每10次打印一次状态
                if self.stats['no_moves'] % 10 == 0:
                    print(f"✅ 目标{target_id}在中心区域 | rel_x={rel_x:.3f} | 保持位置")

        except FileNotFoundError:
            print(f"⚠️  坐标文件不存在: {coords_file}")
            self.stats['errors'] += 1
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON解析失败: {e}")
            self.stats['errors'] += 1
        except KeyError as e:
            print(f"⚠️  数据格式错误: {e}")
            self.stats['errors'] += 1
        except Exception as e:
            print(f"❌ 错误: {e}")
            self.stats['errors'] += 1

    def print_stats(self):
        """打印统计信息"""
        print("\n" + "="*60)
        print("📊 跟踪统计")
        print("="*60)
        print(f"总更新次数: {self.stats['total_updates']}")
        print(f"向左移动: {self.stats['left_moves']}")
        print(f"向右移动: {self.stats['right_moves']}")
        print(f"保持位置: {self.stats['no_moves']}")
        print(f"丢失帧数: {self.stats['lost_frames']}")
        print(f"回到零位: {self.stats['resets_to_zero']} 次")
        print(f"错误次数: {self.stats['errors']}")
        print("="*60 + "\n")

    def close(self):
        """关闭串口"""
        self.motor.disconnect()


def main():
    parser = argparse.ArgumentParser(description='水平方向左右跟踪算法')
    parser.add_argument('--coords', type=str, default='tracker_coords.json',
                       help='坐标JSON文件路径（默认：tracker_coords.json）')
    parser.add_argument('--serial', type=str, default='/dev/ttyS8',
                       help='串口设备（默认：/dev/ttyS8）')
    parser.add_argument('--baudrate', type=int, default=115200,
                       help='波特率（默认：115200）')
    parser.add_argument('--center-start', type=float, default=0.375,
                       help='中心区域左边界（默认：0.375 = 3/8）')
    parser.add_argument('--center-end', type=float, default=0.625,
                       help='中心区域右边界（默认：0.625 = 5/8）')
    parser.add_argument('--deadzone', type=float, default=0.02,
                       help='死区大小（默认：0.02）')
    parser.add_argument('--max-delta', type=float, default=0.15,
                       help='单次最大转动角度/弧度（默认：0.15）')
    parser.add_argument('--kp', type=float, default=0.5,
                       help='比例系数（默认：0.5）')
    parser.add_argument('--max-lost-frames', type=int, default=10,
                       help='目标丢失多少帧后回到零位（默认：10）')
    parser.add_argument('--interval', type=float, default=0.125,
                       help='更新间隔/秒（默认：0.125，即8Hz）')
    parser.add_argument('--stats-interval', type=int, default=50,
                       help='统计信息打印间隔（默认：50次）')

    args = parser.parse_args()

    print("="*60)
    print("水平方向左右跟踪算法")
    print("="*60)
    print(f"坐标文件: {args.coords}")
    print(f"串口: {args.serial} @ {args.baudrate}")
    print(f"中心区域: {args.center_start*100:.1f}% - {args.center_end*100:.1f}%")
    print(f"死区: ±{args.deadzone*100:.1f}%")
    print(f"最大转动: {args.max_delta} rad")
    print(f"比例系数: {args.kp}")
    print(f"丢失阈值: {args.max_lost_frames} 帧（约{args.max_lost_frames/8:.1f}秒）")
    print(f"更新频率: {1/args.interval:.1f} Hz")
    print("="*60 + "\n")

    # 创建跟踪器
    tracker = HorizontalTracker(
        serial_port=args.serial,
        baudrate=args.baudrate,
        center_zone_start=args.center_start,
        center_zone_end=args.center_end,
        deadzone=args.deadzone,
        max_delta=args.max_delta,
        kp=args.kp,
        max_lost_frames=args.max_lost_frames
    )

    # 打开串口
    if not tracker.open_serial():
        print("❌ 无法打开串口，退出")
        return

    # 等待坐标文件生成
    coords_path = Path(args.coords)
    print("等待坐标文件生成...")
    while not coords_path.exists():
        time.sleep(0.5)
    print("✅ 坐标文件已找到，开始跟踪\n")

    try:
        update_count = 0
        last_stats_time = time.time()

        while True:
            # 更新跟踪
            tracker.update(args.coords)

            update_count += 1

            # 定期打印统计信息
            if update_count % args.stats_interval == 0:
                tracker.print_stats()

            # 等待下一次更新
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    finally:
        tracker.print_stats()
        tracker.close()
        print("程序结束")


if __name__ == '__main__':
    main()
