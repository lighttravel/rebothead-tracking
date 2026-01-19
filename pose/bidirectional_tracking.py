#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双向跟踪算法（水平+垂直，增量控制模式）
- 读取pose_head_tracking.py输出的坐标数据
- 根据目标位置发送增量命令同时控制电机21（偏航轴）和电机22（俯仰轴）
- 使用 motor_controller.py 发送命令
- 中心区域：图像宽度和高度的 3/8 (37.5%) 到 5/8 (62.5%)
- 智能抬头找头功能：检测到人但没检测到头时自动抬头搜索
"""

import json
import time
import argparse
from pathlib import Path
from motor_controller import MotorController


class BidirectionalTracker:
    """双向跟踪控制器（水平+垂直）"""

    def __init__(self, serial_port='/dev/ttyS8', baudrate=115200,
                 center_zone_start=0.375, center_zone_end=0.625,
                 deadzone=0.02, max_delta=0.15, kp=0.5, max_lost_frames=10):
        """
        初始化双向跟踪控制器

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

        # 智能低头找头功能（当检测到身体但没头部时）
        self.head_search_mode = False      # 是否处于低头搜索模式
        self.head_search_start_pitch = 0.0 # 搜索开始时的俯仰角度
        self.head_search_steps = 0         # 已搜索步数
        self.max_search_steps = 8          # 最大搜索步数
        self.search_pitch_delta = 0.05     # 每次低头角度（正值=低头）
        self.no_head_detected_count = 0    # 连续没检测到头的帧数
        self.max_no_head_frames = 5        # 连续多少帧没头才触发搜索

        # 超时回到零点功能
        self.last_detection_time = time.time()  # 最后一次检测到人的时间
        self.timeout_seconds = 3.0              # 超时时间（秒）
        self.timeout_check_interval = 10         # 每10次更新检查一次超时

        # 创建电机控制器
        self.motor = MotorController(serial_port, baudrate)

        # 统计信息
        self.stats = {
            'total_updates': 0,
            'left_moves': 0,
            'right_moves': 0,
            'up_moves': 0,
            'down_moves': 0,
            'no_moves': 0,
            'lost_frames': 0,
            'resets_to_zero': 0,
            'head_searches': 0,
            'heads_found_by_search': 0,
            'timeout_resets': 0,  # 超时回到零点次数
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

    def _calculate_horizontal(self, rel_x):
        """
        计算水平方向电机控制指令

        参数:
            rel_x: 目标中心点的X相对位置（-1到1，0为中心）

        返回:
            (delta_yaw, direction)
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
            # 目标在左边，需要向左转
            distance = self.center_zone_start - normalized_x
            delta_yaw = distance * self.kp
            delta_yaw = min(self.max_delta, delta_yaw)  # 限制最大角度
            return delta_yaw, 'left'

        elif normalized_x > self.center_zone_end + self.deadzone:
            # 目标在右边，需要向右转
            distance = normalized_x - self.center_zone_end
            delta_yaw = -distance * self.kp
            delta_yaw = max(-self.max_delta, delta_yaw)  # 限制最大角度
            return delta_yaw, 'right'

        else:
            # 目标在中心区域，不需要移动
            return 0.0, 'center'

    def _calculate_vertical(self, rel_y):
        """
        计算垂直方向电机控制指令

        参数:
            rel_y: 目标中心点的Y相对位置（-1到1，0为中心）
                    注意：在图像坐标系中，y向下为正

        返回:
            (delta_pitch, direction)
            delta_pitch: 转动角度（弧度，正数向上，负数向下）
            direction: 方向描述 ('up', 'down', 'center')
        """
        # 将rel_y从[-1, 1]转换到[0, 1]
        # 如果已经是0-1范围则不用转换
        if -1 <= rel_y <= 1:
            normalized_y = (rel_y + 1) / 2  # 转换到0-1
        else:
            normalized_y = rel_y

        # 检查目标位置
        if normalized_y < self.center_zone_start - self.deadzone:
            # 目标在上方，需要向上转（正值）
            distance = self.center_zone_start - normalized_y
            delta_pitch = distance * self.kp
            delta_pitch = min(self.max_delta, delta_pitch)  # 限制最大角度
            return delta_pitch, 'up'

        elif normalized_y > self.center_zone_end + self.deadzone:
            # 目标在下方，需要向下转（负值）
            distance = normalized_y - self.center_zone_end
            delta_pitch = -distance * self.kp
            delta_pitch = max(-self.max_delta, delta_pitch)  # 限制最大角度
            return delta_pitch, 'down'

        else:
            # 目标在中心区域，不需要移动
            return 0.0, 'center'

    def calculate_motor_command(self, rel_x, rel_y):
        """
        根据目标相对位置计算电机控制指令（双向）

        参数:
            rel_x: 目标中心点的X相对位置（-1到1，0为中心）
            rel_y: 目标中心点的Y相对位置（-1到1，0为中心）

        返回:
            (should_move, yaw_delta, pitch_delta, direction_h, direction_v)
            should_move: 是否需要移动
            yaw_delta: 水平转动角度（弧度）
            pitch_delta: 垂直转动角度（弧度）
            direction_h: 水平方向描述 ('left', 'right', 'center')
            direction_v: 垂直方向描述 ('up', 'down', 'center')
        """
        # 计算水平方向
        yaw_delta, direction_h = self._calculate_horizontal(rel_x)

        # 计算垂直方向
        pitch_delta, direction_v = self._calculate_vertical(rel_y)

        # 判断是否需要移动
        should_move = (yaw_delta != 0 or pitch_delta != 0)

        return should_move, yaw_delta, pitch_delta, direction_h, direction_v

    def start_head_search(self):
        """开始低头搜索头部"""
        self.head_search_mode = True
        self.head_search_steps = 0
        # 记录当前俯仰角度（这里假设为0，实际应该从电机获取）
        self.head_search_start_pitch = 0.0
        self.stats['head_searches'] += 1
        print(f"🔍 开始低头搜索头部...")

    def stop_head_search(self, success=False):
        """停止低头搜索"""
        if self.head_search_mode:
            if success:
                print(f"✅ 低头搜索成功找到头部！")
                self.stats['heads_found_by_search'] += 1
            else:
                # 回到低头前的位置（抬回原位）
                total_pitch = self.head_search_steps * self.search_pitch_delta
                print(f"❌ 未找到头部，回到原位置（抬头 {abs(total_pitch):.3f}rad）")
                # 发送反向命令（抬头=负值）
                self.send_motor_delta(0, -total_pitch)

            self.head_search_mode = False
            self.head_search_steps = 0

    def check_timeout_reset(self):
        """检查是否超时，超时则回到零点"""
        current_time = time.time()
        time_since_last_detection = current_time - self.last_detection_time

        if time_since_last_detection > self.timeout_seconds:
            print(f"⏰ 超过 {self.timeout_seconds} 秒未检测到人，回到零点")
            success = self.send_motor_absolute(0.0, 0.0)
            if success:
                self.stats['resets_to_zero'] += 1
                self.stats['timeout_resets'] += 1
            # 更新最后检测时间，避免重复触发
            self.last_detection_time = current_time
            return True
        return False

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

            # 获取所有检测（用于判断是否有身体检测）
            all_detections = data.get('all_detections', [])
            has_body = len(all_detections) > 0

            # 🔍 调试输出
            if primary:
                head_center = primary.get('position', {}).get('center')
                print(f"🔍 [DEBUG] primary_target: id={primary.get('id')}, head_center={head_center}")
            else:
                print(f"🔍 [DEBUG] primary_target: None, has_body={has_body}, all_detections={len(all_detections)}")

            # 如果没有 primary_target，检查是否有身体检测
            if not primary:
                # 没有主目标（没有检测到头部）
                self.lost_count += 1
                self.stats['lost_frames'] += 1

                # 如果处于搜索模式，停止搜索
                if self.head_search_mode:
                    self.stop_head_search(success=False)

                # 如果有身体检测，不立即返回（继续检查是否需要触发搜索）
                # 如果没有身体检测，才是真正的目标丢失
                if not has_body:
                    if self.lost_count <= self.max_lost_frames:
                        print(f"⚠️  目标丢失 ({self.lost_count}/{self.max_lost_frames})")
                    elif self.lost_count == self.max_lost_frames + 1:
                        # 连续丢失max_lost_frames帧，回到零位
                        print(f"🔄 目标丢失超过{self.max_lost_frames}帧，回到零位")
                        success = self.send_motor_absolute(0.0, 0.0)
                        if success:
                            self.stats['resets_to_zero'] += 1
                    return

            # ========== 更新最后检测时间（用于超时检测） ==========
            if has_body:
                self.last_detection_time = time.time()

            # 检查是否有头部检测
            head_center = primary.get('position', {}).get('center') if primary else None
            has_head = head_center is not None

            # ========== 智能低头找头逻辑 ==========
            if has_head:
                # 检测到头部
                self.no_head_detected_count = 0

                # 如果处于搜索模式，成功找到头部
                if self.head_search_mode:
                    self.stop_head_search(success=True)
            else:
                # 没有检测到头部
                self.no_head_detected_count += 1

                if has_body and self.no_head_detected_count >= self.max_no_head_frames:
                    # 检测到身体但没检测到头，持续一定帧数，触发低头搜索
                    if not self.head_search_mode:
                        self.start_head_search()

                # 如果处于搜索模式
                if self.head_search_mode:
                    if self.head_search_steps < self.max_search_steps:
                        # 继续低头，但水平方向也要跟踪躯体
                        print(f"🔍 低头搜索 {self.head_search_steps + 1}/{self.max_search_steps}...")

                        # 🔥 关键修改：计算躯体中心的水平位置
                        # 从 all_detections 中找到距离中心最近的人体
                        yaw_delta = 0.0  # 默认值（如果没有身体检测）

                        if has_body:
                            # 从 frame_info 获取图像尺寸
                            frame_info = data.get('frame_info', {})
                            img_width = frame_info.get('width', 1280)
                            img_height = frame_info.get('height', 720)
                            img_center_x = img_width / 2
                            img_center_y = img_height / 2

                            # 计算所有检测框的中心点
                            best_target = None
                            min_distance = float('inf')

                            for det in all_detections:
                                bbox = det['bbox']  # [xmin, ymin, xmax, ymax]
                                center_x = (bbox[0] + bbox[2]) / 2
                                center_y = (bbox[1] + bbox[3]) / 2

                                # 计算距离图像中心的距离
                                dist_x = center_x - img_center_x
                                dist_y = center_y - img_center_y
                                distance = (dist_x ** 2 + dist_y ** 2) ** 0.5

                                if distance < min_distance:
                                    min_distance = distance
                                    best_target = det

                            if best_target:
                                bbox = best_target['bbox']
                                center_x = (bbox[0] + bbox[2]) / 2
                                # 计算相对位置（-1到1）
                                rel_x = (center_x - img_center_x) / (img_center_x)

                                # 计算水平方向需要移动的角度
                                if -1 <= rel_x <= 1:
                                    normalized_x = (rel_x + 1) / 2
                                else:
                                    normalized_x = rel_x

                                # 计算水平方向的电机命令
                                if normalized_x < self.center_zone_start - self.deadzone:
                                    # 目标在左边，需要向左转
                                    distance = self.center_zone_start - normalized_x
                                    yaw_delta = distance * self.kp
                                    yaw_delta = min(self.max_delta, yaw_delta)
                                elif normalized_x > self.center_zone_end + self.deadzone:
                                    # 目标在右边，需要向右转
                                    distance = normalized_x - self.center_zone_end
                                    yaw_delta = -distance * self.kp
                                    yaw_delta = max(-self.max_delta, yaw_delta)
                                # 如果在中心区域，yaw_delta 保持为 0.0

                        # 发送命令：水平跟踪 + 低头搜索
                        pitch_delta = self.search_pitch_delta  # 低头
                        self.send_motor_delta(yaw_delta, pitch_delta)

                        self.head_search_steps += 1
                        time.sleep(0.2)  # 等待电机动作
                        return  # 这一次完成
                    else:
                        # 达到最大搜索次数，停止搜索
                        self.stop_head_search(success=False)
                        self.no_head_detected_count = 0  # 重置计数
                        return  # 没找到头部，返回等待下一次

            # 如果没有 primary_target，直接返回（不能进行正常跟踪）
            if not primary:
                return

            # 目标重新出现，重置丢失计数
            if self.lost_count > 0:
                print(f"✅ 目标重新出现 (丢失了{self.lost_count}帧)")
                self.lost_count = 0

            # 获取相对位置
            rel_x = primary['position']['relative']['x']
            rel_y = primary['position']['relative']['y']
            confidence = primary['confidence']
            target_id = primary['id']

            # 计算控制指令（双向）
            should_move, yaw_delta, pitch_delta, dir_h, dir_v = self.calculate_motor_command(rel_x, rel_y)

            # 更新统计
            self.stats['total_updates'] += 1

            if should_move:
                # 发送增量命令（同时控制水平和垂直）
                print(f"🔍 [DEBUG] 准备发送电机命令: yaw_delta={yaw_delta:.4f}, pitch_delta={pitch_delta:.4f}")
                success = self.send_motor_delta(yaw_delta, pitch_delta)

                if success:
                    # 更新水平方向统计
                    if dir_h == 'left':
                        self.stats['left_moves'] += 1
                    elif dir_h == 'right':
                        self.stats['right_moves'] += 1

                    # 更新垂直方向统计
                    if dir_v == 'up':
                        self.stats['up_moves'] += 1
                    elif dir_v == 'down':
                        self.stats['down_moves'] += 1

                    # 构建方向描述
                    h_desc = f"向左转{abs(yaw_delta):.3f}rad" if dir_h == 'left' else (f"向右转{abs(yaw_delta):.3f}rad" if dir_h == 'right' else "水平保持")
                    v_desc = f"向上{abs(pitch_delta):.3f}rad" if dir_v == 'up' else (f"向下{abs(pitch_delta):.3f}rad" if dir_v == 'down' else "垂直保持")

                    print(f"🎯 目标{target_id} | rel=({rel_x:.2f},{rel_y:.2f}) | {h_desc}, {v_desc}")
                else:
                    print(f"❌ 发送失败")
            else:
                self.stats['no_moves'] += 1
                # 每10次打印一次状态
                if self.stats['no_moves'] % 10 == 0:
                    print(f"✅ 目标{target_id}在中心区域 | rel=({rel_x:.2f},{rel_y:.2f}) | 保持位置")

            # ========== 超时检测 ==========
            # 定期检查是否超时（每10次更新检查一次）
            if self.stats['total_updates'] % self.timeout_check_interval == 0:
                self.check_timeout_reset()

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
        print("📊 双向跟踪统计")
        print("="*60)
        print(f"总更新次数: {self.stats['total_updates']}")
        print(f"\n水平方向:")
        print(f"  向左移动: {self.stats['left_moves']}")
        print(f"  向右移动: {self.stats['right_moves']}")
        print(f"\n垂直方向:")
        print(f"  向上移动: {self.stats['up_moves']}")
        print(f"  向下移动: {self.stats['down_moves']}")
        print(f"\n其他:")
        print(f"  保持位置: {self.stats['no_moves']}")
        print(f"  丢失帧数: {self.stats['lost_frames']}")
        print(f"  回到零位: {self.stats['resets_to_zero']} 次")
        print(f"\n智能低头搜索:")
        print(f"  触发次数: {self.stats['head_searches']}")
        print(f"  成功找到头: {self.stats['heads_found_by_search']}")
        print(f"  成功率: {self.stats['heads_found_by_search']/self.stats['head_searches']*100 if self.stats['head_searches'] > 0 else 0:.1f}%")
        print(f"\n超时回到零点:")
        print(f"  超时重置次数: {self.stats['timeout_resets']} 次")
        print(f"  超时阈值: {self.timeout_seconds} 秒")
        print(f"  错误次数: {self.stats['errors']}")
        print("="*60 + "\n")

    def close(self):
        """关闭串口"""
        self.motor.disconnect()


def main():
    parser = argparse.ArgumentParser(description='双向跟踪算法（水平+垂直）')
    parser.add_argument('--coords', type=str, default='tracker_coords.json',
                       help='坐标JSON文件路径（默认：tracker_coords.json）')
    parser.add_argument('--serial', type=str, default='/dev/ttyS8',
                       help='串口设备（默认：/dev/ttyS8）')
    parser.add_argument('--baudrate', type=int, default=115200,
                       help='波特率（默认：115200）')
    parser.add_argument('--center-start', type=float, default=0.375,
                       help='中心区域边界（默认：0.375 = 3/8，适用于水平和垂直）')
    parser.add_argument('--center-end', type=float, default=0.625,
                       help='中心区域边界（默认：0.625 = 5/8，适用于水平和垂直）')
    parser.add_argument('--deadzone', type=float, default=0.02,
                       help='死区大小（默认：0.02）')
    parser.add_argument('--max-delta', type=float, default=0.15,
                       help='单次最大转动角度/弧度（默认：0.15）')
    parser.add_argument('--kp', type=float, default=0.5,
                       help='比例系数（默认：0.5）')
    parser.add_argument('--max-lost-frames', type=int, default=10,
                       help='目标丢失多少帧后回到零位（默认：10）')
    parser.add_argument('--max-no-head-frames', type=int, default=5,
                       help='连续多少帧没检测到头才触发低头搜索（默认：5）')
    parser.add_argument('--max-search-steps', type=int, default=8,
                       help='低头搜索最大步数（默认：8）')
    parser.add_argument('--search-pitch', type=float, default=0.05,
                       help='每次低头角度（弧度，正值=低头，默认：0.05）')
    parser.add_argument('--timeout-seconds', type=float, default=3.0,
                       help='超时多少秒后回到零点（默认：3.0秒）')
    parser.add_argument('--interval', type=float, default=0.125,
                       help='更新间隔/秒（默认：0.125，即8Hz）')
    parser.add_argument('--stats-interval', type=int, default=50,
                       help='统计信息打印间隔（默认：50次）')

    args = parser.parse_args()

    print("="*60)
    print("双向跟踪算法（水平+垂直）+ 智能低头找头 + 超时回零")
    print("="*60)
    print(f"坐标文件: {args.coords}")
    print(f"串口: {args.serial} @ {args.baudrate}")
    print(f"中心区域: {args.center_start*100:.1f}% - {args.center_end*100:.1f}%")
    print(f"  （适用于水平和垂直两个方向）")
    print(f"死区: ±{args.deadzone*100:.1f}%")
    print(f"最大转动: {args.max_delta} rad")
    print(f"比例系数: {args.kp}")
    print(f"丢失阈值: {args.max_lost_frames} 帧（约{args.max_lost_frames/8:.1f}秒）")
    print(f"\n智能低头找头:")
    print(f"  触发条件: 连续 {args.max_no_head_frames} 帧没检测到头")
    print(f"  搜索步数: 最多 {args.max_search_steps} 步")
    print(f"  每步角度: {args.search_pitch} rad ({'低头' if args.search_pitch > 0 else '抬头'})")
    print(f"  ⚠️  搜索期间水平方向继续跟踪，只改变俯仰角")
    print(f"\n超时回到零点:")
    print(f"  超时阈值: {args.timeout_seconds} 秒")
    print(f"  更新频率: {1/args.interval:.1f} Hz")
    print("="*60 + "\n")

    # 创建跟踪器
    tracker = BidirectionalTracker(
        serial_port=args.serial,
        baudrate=args.baudrate,
        center_zone_start=args.center_start,
        center_zone_end=args.center_end,
        deadzone=args.deadzone,
        max_delta=args.max_delta,
        kp=args.kp,
        max_lost_frames=args.max_lost_frames
    )

    # 设置低头搜索参数
    tracker.max_no_head_frames = args.max_no_head_frames
    tracker.max_search_steps = args.max_search_steps
    tracker.search_pitch_delta = args.search_pitch
    tracker.timeout_seconds = args.timeout_seconds

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
