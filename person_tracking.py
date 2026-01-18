#!/usr/bin/env python3
"""
RK3576 YOLOv8 人体跟踪 + MJPEG 推流
专为电机跟踪设计，包含：
- 可配置检测频率
- 位置平滑滤波
- 目标持续跟踪
- 输出电机控制坐标
"""

import cv2
import numpy as np
import argparse
import threading
import time
from io import BytesIO
from collections import deque
from rknnlite.api import RKNNLite
from http.server import HTTPServer, BaseHTTPRequestHandler
import socket
import json

# ============== 参数设置 ==============
OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = (640, 640)

# COCO 80类（只需要 person）
CLASSES = ("person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
           "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
           "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
           "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
           "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
           "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush")


# ============== 后处理函数 ===============
def dfl(position):
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num
    position = position.reshape(n, p_num, mc, h, w)
    position = np.exp(position) / (np.exp(position).sum(axis=2, keepdims=True) + 1e-6)
    acc_metrix = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    position = (position * acc_metrix).sum(axis=2)
    return position


def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1] // grid_h, IMG_SIZE[0] // grid_w]).reshape(1, 2, 1, 1)
    position = dfl(position)
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)
    return xyxy


def filter_boxes(boxes, box_confidences, box_class_probs):
    box_confidences = box_confidences.reshape(-1)
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]
    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    return boxes, classes, scores


def nms_boxes(boxes, scores):
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    areas = w * h
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])
        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def post_process(input_data):
    boxes, scores, classes_conf = [], [], []
    default_branch = 3
    pair_per_branch = len(input_data) // default_branch

    for i in range(default_branch):
        boxes.append(box_process(input_data[pair_per_branch * i]))
        classes_conf.append(input_data[pair_per_branch * i + 1])
        scores.append(np.ones_like(input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)
        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)
    return boxes, classes, scores


class YOLOv8Detector:
    def __init__(self, rknn_model='yolov8.rknn', target_size=640):
        self.target_size = target_size
        self.img_size = (target_size, target_size)

        print(f'--> Loading RKNN model: {rknn_model}')
        self.rknn_lite = RKNNLite()
        ret = self.rknn_lite.load_rknn(rknn_model)
        if ret != 0:
            raise RuntimeError('Load RKNN model failed!')

        print('--> Init RKNN runtime')
        ret = self.rknn_lite.init_runtime()
        if ret != 0:
            raise RuntimeError('Init runtime failed!')

        print('✅ Model loaded successfully')

    def letterbox(self, img):
        img_h, img_w = img.shape[:2]
        shape = (img_h, img_w)
        scale = min(self.img_size[0] / shape[0], self.img_size[1] / shape[1])
        new_unpad = (int(round(shape[1] * scale)), int(round(shape[0] * scale)))
        dw_total, dh_total = self.img_size[1] - new_unpad[0], self.img_size[0] - new_unpad[1]
        dw = dw_total / 2
        dh = dh_total / 2

        if shape[::-1] != new_unpad:
            resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        else:
            resized = img

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img_padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                        cv2.BORDER_CONSTANT, value=(0, 0, 0))

        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(img_rgb, axis=0)

        return input_data, scale, dw, dh

    def detect_frame(self, frame):
        img_h, img_w = frame.shape[:2]

        # 预处理
        input_data, scale, dw, dh = self.letterbox(frame)

        # 推理
        try:
            outputs = self.rknn_lite.inference(inputs=[input_data])
        except KeyboardInterrupt:
            raise
        except Exception as e:
            return None, None, None

        # 后处理
        boxes, classes, scores = post_process(outputs)

        if boxes is None:
            return None, None, None

        # 坐标转换
        boxes[:, 0] = (boxes[:, 0] - dw) / scale
        boxes[:, 1] = (boxes[:, 1] - dh) / scale
        boxes[:, 2] = (boxes[:, 2] - dw) / scale
        boxes[:, 3] = (boxes[:, 3] - dh) / scale

        # 限制在图像范围内
        boxes[:, 0] = np.clip(boxes[:, 0], 0, img_w)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, img_h)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, img_w)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, img_h)

        return boxes, classes, scores

    def release(self):
        if hasattr(self, 'rknn_lite'):
            self.rknn_lite.release()


class TrackedTarget:
    """被跟踪的目标（增强版）"""
    def __init__(self, track_id, box, score, img_width, img_height):
        self.id = track_id
        self.score = score
        self.first_seen = time.time()  # 首次检测时间
        self.last_seen = time.time()   # 最后检测时间
        self.frame_count = 1           # 跟踪帧数

        # 边界框 (x1, y1, x2, y2)
        self.box = box
        self.box_width = box[2] - box[0]
        self.box_height = box[3] - box[1]
        self.box_area = self.box_width * self.box_height

        # 中心点
        self.center_x = (box[0] + box[2]) / 2
        self.center_y = (box[1] + box[3]) / 2

        # 平滑滤波（使用移动平均）
        self.history_x = deque(maxlen=5)
        self.history_y = deque(maxlen=5)
        self.history_x.append(self.center_x)
        self.history_y.append(self.center_y)

        # 平滑后的中心点
        self.smooth_x = self.center_x
        self.smooth_y = self.center_y

        # 相对位置（-1 到 1，用于电机控制）
        self.rel_x = (self.center_x - img_width / 2) / (img_width / 2)
        self.rel_y = (self.center_y - img_height / 2) / (img_height / 2)

        # 距离图像中心的像素距离
        self.dist_x = self.center_x - img_width / 2
        self.dist_y = self.center_y - img_height / 2
        self.distance = np.sqrt(self.dist_x ** 2 + self.dist_y ** 2)

        # 速度（像素/秒）
        self.vx = 0.0
        self.vy = 0.0
        self.speed = 0.0

        # 运动方向（角度，0-360度）
        self.angle = 0.0

        # 丢失计数（用于判断目标是否消失）
        self.lost_count = 0

        # 运动状态
        self.is_moving = False
        self.direction_str = "静止"

    def update(self, box, score, img_width, img_height):
        """更新目标位置"""
        current_time = time.time()

        # 记录上一帧位置
        prev_center_x = self.smooth_x
        prev_center_y = self.smooth_y
        prev_time = self.last_seen

        # 更新基本信息
        self.box = box
        self.score = score
        self.box_width = box[2] - box[0]
        self.box_height = box[3] - box[1]
        self.box_area = self.box_width * self.box_height

        # 新的中心点
        new_center_x = (box[0] + box[2]) / 2
        new_center_y = (box[1] + box[3]) / 2

        # 添加到历史记录
        self.history_x.append(new_center_x)
        self.history_y.append(new_center_y)

        # 计算平滑后的位置（移动平均）
        self.smooth_x = sum(self.history_x) / len(self.history_x)
        self.smooth_y = sum(self.history_y) / len(self.history_y)

        # 相对位置
        self.rel_x = (self.smooth_x - img_width / 2) / (img_width / 2)
        self.rel_y = (self.smooth_y - img_height / 2) / (img_height / 2)

        # 距离图像中心
        self.dist_x = self.smooth_x - img_width / 2
        self.dist_y = self.smooth_y - img_height / 2
        self.distance = np.sqrt(self.dist_x ** 2 + self.dist_y ** 2)

        # 计算速度和时间差
        dt = current_time - prev_time
        if dt > 0:
            dx = self.smooth_x - prev_center_x
            dy = self.smooth_y - prev_center_y
            self.vx = dx / dt  # 像素/秒
            self.vy = dy / dt
            self.speed = np.sqrt(self.vx ** 2 + self.vy ** 2)

            # 计算运动方向（角度）
            if self.speed > 5:  # 速度大于5像素/秒才认为在运动
                self.angle = np.arctan2(-dy, dx) * 180 / np.pi  # -dy因为y轴向下
                if self.angle < 0:
                    self.angle += 360
                self.is_moving = True

                # 方向描述
                if 337.5 <= self.angle or self.angle < 22.5:
                    self.direction_str = "向右"
                elif 22.5 <= self.angle < 67.5:
                    self.direction_str = "右下"
                elif 67.5 <= self.angle < 112.5:
                    self.direction_str = "向下"
                elif 112.5 <= self.angle < 157.5:
                    self.direction_str = "左下"
                elif 157.5 <= self.angle < 202.5:
                    self.direction_str = "向左"
                elif 202.5 <= self.angle < 247.5:
                    self.direction_str = "左上"
                elif 247.5 <= self.angle < 292.5:
                    self.direction_str = "向上"
                elif 292.5 <= self.angle < 337.5:
                    self.direction_str = "右上"
            else:
                self.is_moving = False
                self.direction_str = "静止"
        else:
            self.vx = 0.0
            self.vy = 0.0
            self.speed = 0.0
            self.is_moving = False
            self.direction_str = "静止"

        # 更新时间
        self.last_seen = current_time
        self.frame_count += 1

        # 重置丢失计数
        self.lost_count = 0

    def get_smooth_center(self):
        """获取平滑后的中心点"""
        return self.smooth_x, self.smooth_y

    def get_relative_position(self):
        """获取相对位置（-1 到 1）"""
        return self.rel_x, self.rel_y

    def get_velocity(self):
        """获取速度（vx, vy, speed）"""
        return self.vx, self.vy, self.speed

    def to_dict(self, img_width, img_height):
        """转换为字典（用于 JSON 输出）"""
        return {
            'id': self.id,
            'confidence': float(self.score),
            'tracking': {
                'frames': self.frame_count,
                'duration': float(self.last_seen - self.first_seen),
                'lost_count': self.lost_count
            },
            'position': {
                'center': {'x': float(self.smooth_x), 'y': float(self.smooth_y)},
                'bbox': {
                    'x1': float(self.box[0]),
                    'y1': float(self.box[1]),
                    'x2': float(self.box[2]),
                    'y2': float(self.box[3]),
                    'width': float(self.box_width),
                    'height': float(self.box_height),
                    'area': float(self.box_area)
                },
                'relative': {  # -1 到 1，适合电机控制
                    'x': float(self.rel_x),
                    'y': float(self.rel_y)
                },
                'distance_from_center': {
                    'x': float(self.dist_x),  # 像素
                    'y': float(self.dist_y),  # 像素
                    'euclidean': float(self.distance)  # 像素
                }
            },
            'motion': {
                'is_moving': self.is_moving,
                'velocity': {
                    'vx': float(self.vx),  # 像素/秒，向右为正
                    'vy': float(self.vy),  # 像素/秒，向下为正
                    'speed': float(self.speed)  # 像素/秒
                },
                'direction': {
                    'angle': float(self.angle),  # 度数，0=右，90=上，180=左，270=下
                    'text': self.direction_str
                }
            },
            'timestamp': float(self.last_seen)
        }


class PersonTracker:
    """人体跟踪器"""
    def __init__(self, max_distance=100, iou_threshold=0.3):
        self.max_distance = max_distance  # 最大距离（像素）
        self.iou_threshold = iou_threshold  # IOU 阈值
        self.next_id = 1
        self.targets = {}  # {track_id: TrackedTarget}
        self.img_width = 1280
        self.img_height = 720

    def set_image_size(self, width, height):
        """设置图像尺寸"""
        self.img_width = width
        self.img_height = height

    def calculate_iou(self, box1, box2):
        """计算 IOU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def calculate_distance(self, box1, box2):
        """计算中心点距离"""
        center1_x = (box1[0] + box1[2]) / 2
        center1_y = (box1[1] + box1[3]) / 2
        center2_x = (box2[0] + box2[2]) / 2
        center2_y = (box2[1] + box2[3]) / 2

        return np.sqrt((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2)

    def update(self, boxes, classes, scores):
        """更新跟踪目标"""
        # 处理没有检测到目标的情况
        if boxes is None or classes is None or scores is None:
            # 没有检测到目标，增加所有目标的丢失计数
            for track_id in list(self.targets.keys()):
                self.targets[track_id].lost_count += 1
                # 如果丢失太久，删除目标
                if self.targets[track_id].lost_count > 10:  # 丢失10帧后删除
                    del self.targets[track_id]
            return

        # 只跟踪 person (class_id = 0)
        person_indices = [i for i, cls in enumerate(classes) if cls == 0]
        person_boxes = boxes[person_indices] if len(person_indices) > 0 else None

        if person_boxes is None or len(person_boxes) == 0:
            # 没有检测到人，增加所有目标的丢失计数
            for track_id in list(self.targets.keys()):
                self.targets[track_id].lost_count += 1
                # 如果丢失太久，删除目标
                if self.targets[track_id].lost_count > 10:  # 丢失10帧后删除
                    del self.targets[track_id]
            return

        # 为每个检测框找到匹配的跟踪目标
        matched_tracks = set()
        matched_detections = set()

        # 先尝试用 IOU 匹配
        for i, box in enumerate(person_boxes):
            best_iou = 0
            best_track_id = None

            for track_id, target in self.targets.items():
                if track_id in matched_tracks:
                    continue

                iou = self.calculate_iou(target.box, box)
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_track_id = track_id

            if best_track_id is not None:
                # 找到匹配，更新目标
                self.targets[best_track_id].update(box, scores[person_indices[i]], self.img_width, self.img_height)
                matched_tracks.add(best_track_id)
                matched_detections.add(i)
            else:
                # 没找到匹配，尝试用距离匹配
                best_dist = float('inf')
                best_track_id = None

                for track_id, target in self.targets.items():
                    if track_id in matched_tracks:
                        continue

                    dist = self.calculate_distance(target.box, box)
                    if dist < best_dist and dist < self.max_distance:
                        best_dist = dist
                        best_track_id = track_id

                if best_track_id is not None:
                    self.targets[best_track_id].update(box, scores[person_indices[i]], self.img_width, self.img_height)
                    matched_tracks.add(best_track_id)
                    matched_detections.add(i)
                else:
                    # 创建新目标
                    new_target = TrackedTarget(self.next_id, box, scores[person_indices[i]], self.img_width, self.img_height)
                    self.targets[self.next_id] = new_target
                    matched_tracks.add(self.next_id)
                    matched_detections.add(i)
                    self.next_id += 1

        # 处理未匹配的检测框（新目标）
        for i in range(len(person_boxes)):
            if i not in matched_detections:
                new_target = TrackedTarget(self.next_id, person_boxes[i], scores[person_indices[i]], self.img_width, self.img_height)
                self.targets[self.next_id] = new_target
                self.next_id += 1

        # 处理未匹配的跟踪目标（目标消失）
        for track_id in list(self.targets.keys()):
            if track_id not in matched_tracks:
                self.targets[track_id].lost_count += 1
                if self.targets[track_id].lost_count > 10:
                    del self.targets[track_id]

    def get_primary_target(self):
        """获取主要目标（置信度最高的）"""
        if not self.targets:
            return None

        # 返回置信度最高的目标
        primary_target = max(self.targets.values(), key=lambda t: t.score)
        return primary_target

    def get_all_targets(self):
        """获取所有目标"""
        return list(self.targets.values())


def draw_tracking(frame, tracker, show_all=True):
    """绘制跟踪结果"""
    if show_all:
        # 显示所有被跟踪的目标
        for target in tracker.get_all_targets():
            x1, y1, x2, y2 = map(int, target.box)

            # 颜色根据 ID 变化
            color = (
                int((target.id * 50) % 255),
                int((target.id * 100) % 255),
                int((target.id * 150) % 255)
            )

            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 绘制 ID 和置信度
            label = f'ID:{target.id} {target.score:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 绘制平滑后的中心点
            smooth_x, smooth_y = target.get_smooth_center()
            cv2.circle(frame, (int(smooth_x), int(smooth_y)), 5, color, -1)

            # 绘制相对位置
            rel_x, rel_y = target.get_relative_position()
            info = f'({rel_x:.2f}, {rel_y:.2f})'
            cv2.putText(frame, info, (x1, y2 + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    else:
        # 只显示主要目标
        primary = tracker.get_primary_target()
        if primary:
            x1, y1, x2, y2 = map(int, primary.box)

            # 主目标用绿色
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # 绘制平滑中心点
            smooth_x, smooth_y = primary.get_smooth_center()
            cv2.circle(frame, (int(smooth_x), int(smooth_y)), 8, (0, 255, 0), -1)
            cv2.circle(frame, (int(smooth_x), int(smooth_y)), 15, (0, 255, 0), 2)

            # 显示信息
            label = f'Target {primary.id}: {primary.score:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 显示相对坐标
            rel_x, rel_y = primary.get_relative_position()
            coord_info = f'Pos: ({rel_x:.2f}, {rel_y:.2f})'
            cv2.putText(frame, coord_info, (x1, y2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 绘制十字线
            h, w = frame.shape[:2]
            cv2.line(frame, (int(smooth_x), 0), (int(smooth_x), h), (0, 255, 0), 1)
            cv2.line(frame, (0, int(smooth_y)), (w, int(smooth_y)), (0, 255, 0), 1)

    return frame


class MJPEGStreamer:
    """MJPEG 流服务器"""
    def __init__(self, port=8080):
        self.port = port
        self.frame = None
        self.running = False
        self.fps = 30  # 推流帧率

    def start(self):
        self.running = True

        class StreamHandler(BaseHTTPRequestHandler):
            def __init__(self, *args, streamer=None, **kwargs):
                self.streamer = streamer
                super().__init__(*args, **kwargs)

            def do_GET(self):
                if self.path == '/' or self.path == '/stream':
                    self.send_response(200)
                    self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--frame')
                    self.end_headers()

                    try:
                        while self.streamer.running:
                            if self.streamer.frame is not None:
                                ret, buffer = cv2.imencode('.jpg', self.streamer.frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                                frame_bytes = buffer.tobytes()

                                self.wfile.write(b'--frame\r\n')
                                self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                                self.wfile.write(frame_bytes)
                                self.wfile.write(b'\r\n\r\n')

                                time.sleep(1.0 / self.streamer.fps)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                else:
                    self.send_error(404)

            def log_message(self, format, *args):
                pass  # 禁用日志

        def handler(*args, **kwargs):
            StreamHandler(*args, streamer=self, **kwargs)

        server = HTTPServer(('0.0.0.0', self.port), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

    def update_frame(self, frame):
        self.frame = frame

    def stop(self):
        self.running = False


def get_local_ip():
    """获取本机 IP"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "192.168.132.166"


def main():
    parser = argparse.ArgumentParser(description='RK3576 YOLOv8 人体跟踪 + MJPEG 推流')
    parser.add_argument('--source', type=str, default='36',
                       help='摄像头设备号')
    parser.add_argument('--model', type=str, default='yolov8.rknn',
                       help='RKNN 模型路径')
    parser.add_argument('--width', type=int, default=1280,
                       help='摄像头宽度')
    parser.add_argument('--height', type=int, default=720,
                       help='摄像头高度')
    parser.add_argument('--port', type=int, default=8080,
                       help='HTTP 端口')
    parser.add_argument('--stream-fps', type=int, default=30,
                       help='推流帧率（默认30）')
    parser.add_argument('--detect-fps', type=int, default=5,
                       help='检测帧率（默认5，适合电机跟踪）')
    parser.add_argument('--show-all', action='store_true',
                       help='显示所有跟踪目标（默认只显示主目标）')
    parser.add_argument('--output-coords', type=str, default='/tmp/tracker_coords.json',
                       help='输出坐标到文件（默认：/tmp/tracker_coords.json）')

    args = parser.parse_args()

    local_ip = get_local_ip()
    stream_url = f'http://{local_ip}:{args.port}/stream'

    print('='*60)
    print('RK3576 YOLOv8 人体跟踪系统')
    print('='*60)
    print(f'摄像头: /dev/video{args.source}')
    print(f'分辨率: {args.width}x{args.height}')
    print(f'推流帧率: {args.stream_fps} fps')
    print(f'检测帧率: {args.detect_fps} fps ⭐ 适合电机控制')
    print(f'HTTP 地址: {stream_url}')
    print(f'坐标输出: {args.output_coords}')
    print('='*60)
    print('\n📺 在浏览器或 PotPlayer 中查看:')
    print(f'   {stream_url}')
    print(f'\n📝 电机控制程序可读取:')
    print(f'   {args.output_coords}\n')

    # 加载检测器
    try:
        detector = YOLOv8Detector(args.model)
    except RuntimeError as e:
        print(f'❌ Error: {e}')
        return

    # 创建跟踪器
    tracker = PersonTracker(max_distance=100, iou_threshold=0.3)

    # 打开摄像头
    print(f'--> Opening camera /dev/video{args.source}...')
    cap = cv2.VideoCapture(int(args.source))

    if not cap.isOpened():
        print(f'❌ Failed to open camera')
        detector.release()
        return

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f'    Camera opened: {actual_w}x{actual_h}')
    tracker.set_image_size(actual_w, actual_h)

    # 启动 MJPEG 服务器
    print('--> 启动 HTTP MJPEG 服务器...')
    streamer = MJPEGStreamer(port=args.port)
    streamer.fps = args.stream_fps
    streamer.start()
    print(f'✅ MJPEG 服务器已启动')
    print('='*60)
    print('跟踪运行中... (按 Ctrl+C 停止)')
    print('='*60)

    frame_count = 0
    detect_count = 0
    start_time = time.time()
    last_detect_time = time.time()
    detect_interval = 1.0 / args.detect_fps  # 检测间隔

    last_coords = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            current_time = time.time()

            # 检测（控制频率）
            should_detect = (current_time - last_detect_time) >= detect_interval

            if should_detect:
                detect_count += 1
                last_detect_time = current_time

                # 执行检测
                boxes, classes, scores = detector.detect_frame(frame)

                # 更新跟踪器
                tracker.update(boxes, classes, scores)

                # 输出坐标到文件（默认自动保存）
                if True:  # 自动保存坐标
                    all_targets = tracker.get_all_targets()

                    # 构建坐标数据
                    coords_data = {
                        'system': {
                            'timestamp': time.time(),
                            'frame_count': frame_count,
                            'detect_count': detect_count,
                            'image': {
                                'width': actual_w,
                                'height': actual_h,
                                'center_x': actual_w / 2,
                                'center_y': actual_h / 2
                            }
                        },
                        'targets': []
                    }

                    # 添加所有跟踪目标
                    for target in all_targets:
                        coords_data['targets'].append(target.to_dict(actual_w, actual_h))

                    # 主目标信息
                    primary = tracker.get_primary_target()
                    if primary:
                        coords_data['primary_target'] = {
                            'id': primary.id,
                            'confidence': float(primary.score),
                            'position': {
                                'center': {
                                    'x': float(primary.smooth_x),
                                    'y': float(primary.smooth_y)
                                },
                                'relative': {
                                    'x': float(primary.rel_x),
                                    'y': float(primary.rel_y)
                                }
                            },
                            'motion': {
                                'velocity': {
                                    'vx': float(primary.vx),
                                    'vy': float(primary.vy),
                                    'speed': float(primary.speed)
                                },
                                'direction': {
                                    'angle': float(primary.angle),
                                    'text': primary.direction_str
                                }
                            }
                        }
                    else:
                        coords_data['primary_target'] = None

                    # 添加统计信息
                    coords_data['statistics'] = {
                        'total_targets': len(all_targets),
                        'avg_distance': float(np.mean([t.distance for t in all_targets])) if all_targets else 0,
                        'avg_speed': float(np.mean([t.speed for t in all_targets])) if all_targets else 0
                    }

                    try:
                        with open(args.output_coords, 'w') as f:
                            json.dump(coords_data, f, indent=2)
                    except Exception as e:
                        print(f'写入坐标文件失败: {e}')

            # 绘制跟踪结果
            frame = draw_tracking(frame, tracker, show_all=args.show_all)

            # 添加信息叠加
            primary = tracker.get_primary_target()
            info_lines = [
                f'Frame: {frame_count} | Detect: {detect_count}',
                f'Tracking: {len(tracker.get_all_targets())} person(s)',
            ]

            if primary:
                rel_x, rel_y = primary.get_relative_position()
                info_lines.append(f'Target {primary.id}: rel=({rel_x:.2f}, {rel_y:.2f})')

            # 绘制信息
            y = 30
            for line in info_lines:
                cv2.putText(frame, line, (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y += 25

            # 推流
            streamer.update_frame(frame)

            # 打印统计
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                stream_fps = frame_count / elapsed if elapsed > 0 else 0
                detect_fps = detect_count / elapsed if elapsed > 0 else 0
                print(f'[Frame {frame_count}] 推流: {stream_fps:.1f} fps | 检测: {detect_fps:.1f} fps | 目标: {len(tracker.get_all_targets())}')

    except KeyboardInterrupt:
        print('--> 用户中断')

    finally:
        print('='*60)
        print('跟踪结束')
        print(f'总帧数: {frame_count}')
        print(f'检测次数: {detect_count}')

        elapsed = time.time() - start_time
        if elapsed > 0:
            stream_fps = frame_count / elapsed
            detect_fps = detect_count / elapsed
            print(f'平均推流 FPS: {stream_fps:.2f}')
            print(f'平均检测 FPS: {detect_fps:.2f}')

        streamer.stop()
        cap.release()
        detector.release()

        print('='*60)


if __name__ == '__main__':
    main()
