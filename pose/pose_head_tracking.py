#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于YOLOv8 Pose的头部实时跟踪系统
- 从相机实时检测人体姿态
- 提取头部中心坐标
- 输出到JSON文件供电机控制使用
- 提供MJPEG视频流用于可视化
参考: https://github.com/lighttravel/rebothead-tracking
"""

import cv2
import numpy as np
import json
import time
import argparse
import threading
from io import BytesIO
from collections import deque
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import socket
from rknnlite.api import RKNNLite

# 参数设置
OBJ_THRESH = 0.5
NMS_THRESH = 0.4
IMG_SIZE = (640, 640)

# 姿态调色板
pose_palette = np.array([
    [255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
    [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
    [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
    [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]
], dtype=np.uint8)

kpt_color = pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]

skeleton = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
    [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
    [2, 4], [3, 5], [4, 6], [5, 7]
]

limb_color = pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax, keypoint):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.keypoint = keypoint


def process_output_branch(output, keypoints, kp_offset, model_w, model_h, stride):
    boxes = []
    xywh = output[:, :64, :]
    conf = sigmoid(output[:, 64:, :])

    xywh = xywh.reshape(1, 64, -1)
    conf = conf.reshape(1, -1)

    for i in range(model_h * model_w):
        score = conf[0, i]
        if score > OBJ_THRESH:
            xywh_i = xywh[0, :, i].reshape(1, 4, 16, 1)
            data = np.array([i for i in range(16)]).reshape(1, 1, 16, 1)
            xywh_i = softmax(xywh_i, 2)
            xywh_i = np.multiply(data, xywh_i)
            xywh_i = np.sum(xywh_i, axis=2, keepdims=True).reshape(-1)

            h = i // model_w
            w = i % model_w

            xywh_temp = xywh_i.copy()
            xywh_temp[0] = (w + 0.5) - xywh_i[0]
            xywh_temp[1] = (h + 0.5) - xywh_i[1]
            xywh_temp[2] = (w + 0.5) + xywh_i[2]
            xywh_temp[3] = (h + 0.5) + xywh_i[3]

            xywh_i[0] = ((xywh_temp[0] + xywh_temp[2]) / 2)
            xywh_i[1] = ((xywh_temp[1] + xywh_temp[3]) / 2)
            xywh_i[2] = (xywh_temp[2] - xywh_temp[0])
            xywh_i[3] = (xywh_temp[3] - xywh_temp[1])
            xywh_i = xywh_i * stride

            xmin = (xywh_i[0] - xywh_i[2] / 2)
            ymin = (xywh_i[1] - xywh_i[3] / 2)
            xmax = (xywh_i[0] + xywh_i[2] / 2)
            ymax = (xywh_i[1] + xywh_i[3] / 2)

            global_i = i + kp_offset
            keypoint = keypoints[:, :, :, global_i].reshape(17, 3)

            boxes.append(DetectBox(0, score, xmin, ymin, xmax, ymax, keypoint))

    return boxes


def iou(box1, box2):
    xmin = max(box1.xmin, box2.xmin)
    ymin = max(box1.ymin, box2.ymin)
    xmax = min(box1.xmax, box2.xmax)
    ymax = min(box1.ymax, box2.ymax)

    inner_width = xmax - xmin if xmax > xmin else 0
    inner_height = ymax - ymin if ymax > ymin else 0
    inner_area = inner_width * inner_height

    area1 = (box1.xmax - box1.xmin) * (box1.ymax - box1.ymin)
    area2 = (box2.xmax - box2.xmin) * (box2.ymax - box2.ymin)
    total = area1 + area2 - inner_area

    return inner_area / total if total > 0 else 0


def nms(boxes):
    keep_boxes = []
    sorted_boxes = sorted(boxes, key=lambda x: x.score, reverse=True)

    for i in range(len(sorted_boxes)):
        box1 = sorted_boxes[i]
        if box1.classId != -1:
            keep_boxes.append(box1)
            for j in range(i + 1, len(sorted_boxes)):
                box2 = sorted_boxes[j]
                if box1.classId == box2.classId:
                    if iou(box1, box2) > NMS_THRESH:
                        box2.classId = -1

    return keep_boxes


def post_process(outputs):
    all_boxes = []
    keypoints = outputs[3]

    strides = [8, 16, 32]
    kp_offsets = [0, 6400, 8000]

    for i, output in enumerate(outputs[:3]):
        model_h, model_w = output.shape[2], output.shape[3]
        boxes = process_output_branch(
            output, keypoints, kp_offsets[i], model_w, model_h, strides[i]
        )
        all_boxes.extend(boxes)

    final_boxes = nms(all_boxes)
    return final_boxes


def draw_pose(img, keypoints):
    for k, keypoint in enumerate(keypoints):
        x, y, conf = keypoint
        color_k = [int(x) for x in kpt_color[k]]
        if x != 0 and y != 0 and conf > 0.3:
            cv2.circle(img, (int(x), int(y)), 5, color_k, -1, lineType=cv2.LINE_AA)

    for k, sk in enumerate(skeleton):
        pos1 = (int(keypoints[(sk[0] - 1), 0]), int(keypoints[(sk[0] - 1), 1]))
        pos2 = (int(keypoints[(sk[1] - 1), 0]), int(keypoints[(sk[1] - 1), 1]))

        conf1 = keypoints[(sk[0] - 1), 2]
        conf2 = keypoints[(sk[1] - 1), 2]

        if pos1[0] == 0 or pos1[1] == 0 or pos2[0] == 0 or pos2[1] == 0:
            continue

        if conf1 > 0.3 and conf2 > 0.3:
            color = [int(x) for x in limb_color[k]]
            cv2.line(img, pos1, pos2, color, thickness=2, lineType=cv2.LINE_AA)


def detect_head_circle(keypoints, bbox):
    """检测头部位置"""
    head_kpt_indices = [0, 1, 2, 3, 4]  # nose, eyes, ears
    visible_head_pts = []

    for idx in head_kpt_indices:
        x, y, conf = keypoints[idx]
        if x > 0 and y > 0 and conf > 0.3:
            visible_head_pts.append((x, y))

    if len(visible_head_pts) == 0:
        return None, None

    head_center_x = sum([pt[0] for pt in visible_head_pts]) / len(visible_head_pts)
    head_center_y = sum([pt[1] for pt in visible_head_pts]) / len(visible_head_pts)

    left_ear = keypoints[3]
    right_ear = keypoints[4]
    left_eye = keypoints[1]
    right_eye = keypoints[2]

    if left_ear[0] > 0 and right_ear[0] > 0 and left_ear[2] > 0.3 and right_ear[2] > 0.3:
        ear_dist = abs(right_ear[0] - left_ear[0])
        head_radius = int(ear_dist * 0.9)
    elif left_eye[0] > 0 and right_eye[0] > 0 and left_eye[2] > 0.3 and right_eye[2] > 0.3:
        eye_dist = abs(right_eye[0] - left_eye[0])
        head_radius = int(eye_dist * 2.2)
    else:
        bbox_width = bbox[2] - bbox[0]
        head_radius = int(bbox_width * 0.18)

    head_radius = max(25, min(head_radius, 150))

    return (head_center_x, head_center_y), head_radius


def draw_head_circle(img, head_center, head_radius, color=(255, 0, 0)):
    """绘制头部圆圈"""
    if head_center is not None and head_radius is not None:
        cv2.circle(img, (int(head_center[0]), int(head_center[1])),
                  head_radius, color, 3, lineType=cv2.LINE_AA)
        cv2.circle(img, (int(head_center[0]), int(head_center[1])),
                  3, (0, 255, 0), -1, lineType=cv2.LINE_AA)


class YOLOv8PoseDetector:
    """YOLOv8姿态估计检测器"""

    def __init__(self, rknn_model='yolov8_pose.rknn', target_size=640):
        self.target_size = target_size
        self.img_size = (target_size, target_size)

        print(f'--> Loading RKNN model: {rknn_model}')
        self.rknn_lite = RKNNLite()
        ret = self.rknn_lite.load_rknn(rknn_model)
        if ret != 0:
            print('Load RKNN model failed!')
            exit(-1)

        print('--> Init runtime environment')
        ret = self.rknn_lite.init_runtime()
        if ret != 0:
            print('Init runtime failed!')
            exit(-1)

        print('Model loaded successfully!')

    def release(self):
        """释放 RKNN 资源"""
        if hasattr(self, 'rknn_lite'):
            self.rknn_lite.release()

    def letterbox(self, img):
        """Letterbox预处理"""
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

        return img_padded, scale, dw, dh

    def detect_frame(self, frame):
        """单帧检测"""
        img_h, img_w = frame.shape[:2]

        # 预处理
        input_data, scale, dw, dh = self.letterbox(frame)
        img_rgb = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(img_rgb, axis=0)

        # 推理
        outputs = self.rknn_lite.inference(inputs=[input_data])

        # 后处理
        boxes = post_process(outputs)

        if not boxes:
            return None, frame

        results = []
        img_draw = frame.copy()

        for box in boxes:
            # 坐标映射
            box.xmin = (box.xmin - dw) / scale
            box.ymin = (box.ymin - dh) / scale
            box.xmax = (box.xmax - dw) / scale
            box.ymax = (box.ymax - dh) / scale

            box.xmin = max(0, min(box.xmin, img_w))
            box.ymin = max(0, min(box.ymin, img_h))
            box.xmax = max(0, min(box.xmax, img_w))
            box.ymax = max(0, min(box.ymax, img_h))

            # 关键点映射
            kp = box.keypoint.reshape(-1, 3)
            kp[..., 0] = (kp[..., 0] - dw) / scale
            kp[..., 1] = (kp[..., 1] - dh) / scale
            kp[..., 0] = np.clip(kp[..., 0], 0, img_w)
            kp[..., 1] = np.clip(kp[..., 1], 0, img_h)

            # 绘制
            cv2.rectangle(img_draw, (int(box.xmin), int(box.ymin)),
                         (int(box.xmax), int(box.ymax)), (0, 255, 0), 2)
            cv2.putText(img_draw, f'person {box.score:.2f}',
                       (int(box.xmin), int(box.ymin) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            draw_pose(img_draw, kp)

            # 检测并绘制头部圆圈
            bbox = [box.xmin, box.ymin, box.xmax, box.ymax]
            head_center, head_radius = detect_head_circle(kp, bbox)

            if head_center is not None:
                draw_head_circle(img_draw, head_center, head_radius)

            results.append({
                'confidence': float(box.score),
                'bbox': [float(box.xmin), float(box.ymin), float(box.xmax), float(box.ymax)],
                'keypoints': kp.tolist(),
                'head_center': [float(head_center[0]), float(head_center[1])] if head_center else None,
                'head_radius': head_radius
            })

        return results, img_draw


class TrackedHeadTarget:
    """被跟踪的头部目标"""
    def __init__(self, track_id, head_center, head_radius, confidence, img_width, img_height):
        self.id = track_id
        self.confidence = confidence
        self.head_radius = head_radius

        # 头部中心点
        self.center_x = head_center[0]
        self.center_y = head_center[1]

        # 平滑滤波（5帧移动平均）
        self.history_x = deque(maxlen=5)
        self.history_y = deque(maxlen=5)
        self.history_x.append(self.center_x)
        self.history_y.append(self.center_y)

        # 平滑后的中心点
        self.smooth_x = self.center_x
        self.smooth_y = self.center_y

        # 相对位置（-1到1，用于电机控制）
        self.rel_x = (self.center_x - img_width / 2) / (img_width / 2)
        self.rel_y = (self.center_y - img_height / 2) / (img_height / 2)

        # 距离图像中心的像素距离
        self.dist_x = self.center_x - img_width / 2
        self.dist_y = self.center_y - img_height / 2
        self.distance = np.sqrt(self.dist_x ** 2 + self.dist_y ** 2)

        # 丢失计数
        self.lost_count = 0

        # 首次和最后检测时间
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.frame_count = 1

    def update(self, head_center, confidence, img_width, img_height):
        """更新目标位置"""
        self.center_x = head_center[0]
        self.center_y = head_center[1]
        self.confidence = confidence

        # 添加到历史记录
        self.history_x.append(self.center_x)
        self.history_y.append(self.center_y)

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

        # 更新时间
        self.last_seen = time.time()
        self.frame_count += 1
        self.lost_count = 0


class HeadTracker:
    """头部跟踪器"""

    def __init__(self, max_distance=100, smooth_frames=5):
        self.max_distance = max_distance
        self.smooth_frames = smooth_frames
        self.next_id = 1
        self.targets = {}  # {track_id: TrackedHeadTarget}
        self.img_width = 1280
        self.img_height = 720

    def set_image_size(self, width, height):
        """设置图像尺寸"""
        self.img_width = width
        self.img_height = height

    def calculate_distance(self, center1, center2):
        """计算中心点距离"""
        return np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

    def update(self, detections):
        """
        更新跟踪目标

        Args:
            detections: 检测结果列表
        """
        # 筛选出有头部检测的目标
        valid_detections = []
        for det in detections:
            if det['head_center'] is not None:
                valid_detections.append({
                    'center': tuple(det['head_center']),
                    'confidence': det['confidence'],
                    'radius': det['head_radius']
                })

        if not valid_detections:
            # 没有检测到头部，增加所有目标的丢失计数
            for track_id in list(self.targets.keys()):
                self.targets[track_id].lost_count += 1
                if self.targets[track_id].lost_count > 10:
                    del self.targets[track_id]
            return

        # 为每个检测框找到匹配的跟踪目标
        matched_tracks = set()
        matched_detections = set()

        # 使用距离匹配
        for i, det in enumerate(valid_detections):
            best_dist = float('inf')
            best_track_id = None

            for track_id, target in self.targets.items():
                if track_id in matched_tracks:
                    continue
                dist = self.calculate_distance(det['center'], (target.smooth_x, target.smooth_y))
                if dist < best_dist and dist < self.max_distance:
                    best_dist = dist
                    best_track_id = track_id

            if best_track_id is not None:
                # 找到匹配，更新目标
                self.targets[best_track_id].update(
                    det['center'],
                    det['confidence'],
                    self.img_width,
                    self.img_height
                )
                matched_tracks.add(best_track_id)
                matched_detections.add(i)
            else:
                # 创建新目标
                new_target = TrackedHeadTarget(
                    self.next_id,
                    det['center'],
                    det['radius'],
                    det['confidence'],
                    self.img_width,
                    self.img_height
                )
                self.targets[self.next_id] = new_target
                matched_tracks.add(self.next_id)
                matched_detections.add(i)
                self.next_id += 1

        # 处理未匹配的检测框（新目标）
        for i in range(len(valid_detections)):
            if i not in matched_detections:
                new_target = TrackedHeadTarget(
                    self.next_id,
                    valid_detections[i]['center'],
                    valid_detections[i]['radius'],
                    valid_detections[i]['confidence'],
                    self.img_width,
                    self.img_height
                )
                self.targets[self.next_id] = new_target
                self.next_id += 1

        # 处理未匹配的跟踪目标（目标消失）
        for track_id in list(self.targets.keys()):
            if track_id not in matched_tracks:
                self.targets[track_id].lost_count += 1
                if self.targets[track_id].lost_count > 10:
                    del self.targets[track_id]

    def get_primary_target(self):
        """获取主要目标（距离中心最近的）"""
        if not self.targets:
            return None

        # 返回距离中心最近的目标
        primary_target = min(self.targets.values(), key=lambda t: t.distance)
        return primary_target

    def get_all_targets(self):
        """获取所有目标"""
        return list(self.targets.values())


class MJPEGStreamer:
    """MJPEG流服务器 - 修复版"""

    def __init__(self, port=8080):
        self.port = port
        self.frame = None
        self.running = False
        self.fps = 30

    def start(self):
        """启动流服务器"""
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
                                ret, buffer = cv2.imencode('.jpg', self.streamer.frame,
                                                           [cv2.IMWRITE_JPEG_QUALITY, 80])
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
        """设置当前帧"""
        self.frame = frame

    def stop(self):
        """停止流服务器"""
        self.running = False


def get_local_ip():
    """获取本机IP"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "localhost"


def main():
    parser = argparse.ArgumentParser(description='基于YOLOv8 Pose的头部跟踪系统')
    parser.add_argument('--source', type=str, default='36',
                       help='视频源（设备号/文件路径/RTSP URL）')
    parser.add_argument('--model', type=str, default='yolov8_pose.rknn',
                       help='RKNN模型路径')
    parser.add_argument('--output-coords', type=str, default='tracker_coords.json',
                       help='输出坐标JSON文件路径')
    parser.add_argument('--detect-fps', type=int, default=8,
                       help='检测帧率（默认：8）')
    parser.add_argument('--stream-fps', type=int, default=30,
                       help='推流帧率（默认：30）')
    parser.add_argument('--port', type=int, default=8080,
                       help='HTTP端口（默认：8080）')
    parser.add_argument('--width', type=int, default=1280,
                       help='摄像头宽度（默认：1280）')
    parser.add_argument('--height', type=int, default=720,
                       help='摄像头高度（默认：720）')

    args = parser.parse_args()

    local_ip = get_local_ip()
    stream_url = f'http://{local_ip}:{args.port}/stream'

    print("=" * 60)
    print("基于YOLOv8 Pose的头部跟踪系统")
    print("=" * 60)
    print(f"摄像头: /dev/video{args.source}")
    print(f"分辨率: {args.width}x{args.height}")
    print(f"推流帧率: {args.stream_fps} fps")
    print(f"检测帧率: {args.detect_fps} fps")
    print(f"HTTP 地址: {stream_url}")
    print(f"坐标输出: {args.output_coords}")
    print("=" * 60)
    print(f'\n📺 在浏览器或 PotPlayer 中查看:')
    print(f' {stream_url}')
    print(f'\n📝 电机控制程序可读取:')
    print(f' {args.output_coords}\n')

    # 初始化检测器
    detector = YOLOv8PoseDetector(args.model)

    # 初始化跟踪器
    tracker = HeadTracker(max_distance=100, smooth_frames=5)

    # 初始化视频源
    source = args.source
    if source.isdigit():
        source = int(source)
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        print(f'--> Opening camera /dev/video{source}...')
    else:
        cap = cv2.VideoCapture(source)
        print(f'--> Opening video/file: {source}')

    if not cap.isOpened():
        print('❌ Failed to open video source!')
        detector.release()
        return

    # 获取实际分辨率
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f' Camera opened: {actual_w}x{actual_h}')

    tracker.set_image_size(actual_w, actual_h)

    # 启动 MJPEG 服务器
    print('--> 启动 HTTP MJPEG 服务器...')
    streamer = MJPEGStreamer(port=args.port)
    streamer.fps = args.stream_fps
    streamer.start()
    print('✅ MJPEG 服务器已启动')
    print("=" * 60)
    print('跟踪运行中... (按 Ctrl+C 停止)')
    print("=" * 60)

    # 帧率控制
    detect_interval = 1.0 / args.detect_fps
    last_detect_time = 0

    frame_count = 0
    detect_count = 0
    start_time = time.time()

    # 保存最新的检测结果用于显示
    latest_detections_frame = None
    latest_detections = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('--> End of video stream')
                break

            frame_count += 1
            current_time = time.time()

            # 检测（低频）
            if current_time - last_detect_time >= detect_interval:
                last_detect_time = current_time
                detect_count += 1

                # 执行检测
                detections, detected_frame = detector.detect_frame(frame)

                # 保存检测结果用于显示
                latest_detections = detections
                latest_detections_frame = detected_frame.copy() if detected_frame is not None else None

                # 更新跟踪器（只在没有错误时更新）
                if detections is not None:
                    tracker.update(detections)

                # 获取primary target（可能在上面update后改变）
                primary = tracker.get_primary_target()

                # 写入JSON（即使没有primary target也要写入，以便触发抬头搜索）
                output_data = {
                    'frame_info': {
                        'width': actual_w,
                        'height': actual_h,
                        'frame_count': frame_count,
                        'detect_count': detect_count,
                        'timestamp': datetime.now().isoformat()
                    },
                    'all_detections': [
                        {
                            'confidence': det['confidence'],
                            'bbox': det['bbox'],
                            'head_center': det['head_center'],
                            'head_radius': det['head_radius']
                        }
                        for det in (detections or [])
                    ]
                }

                # 如果有primary target，添加到输出
                if primary:
                    output_data['primary_target'] = {
                        'id': primary.id,
                        'position': {
                            'center': [primary.smooth_x, primary.smooth_y],
                            'relative': {
                                'x': primary.rel_x,
                                'y': primary.rel_y
                            }
                        },
                        'head_radius': primary.head_radius,
                        'confidence': float(primary.confidence),
                        'lost_frames': primary.lost_count,
                        'timestamp': datetime.now().isoformat()
                    }

                    # 打印状态
                    print(f"🎯 目标{primary.id} | "
                          f"rel=({primary.rel_x:.2f}, {primary.rel_y:.2f}) | "
                          f"conf={primary.confidence:.2f}")
                else:
                    # 没有primary target（可能检测到人但没头）
                    output_data['primary_target'] = None
                    # 检查是否有身体检测但没头部
                    has_body = detections and len(detections) > 0
                    has_any_head = any(d['head_center'] is not None for d in (detections or []))
                    if has_body and not has_any_head:
                        print(f"⚠️  检测到 {len(detections)} 人但没有头部 - 可能需要抬头搜索")
                    elif has_body:
                        print(f"✅ 检测到 {len(detections)} 人，其中 {sum(1 for d in detections if d['head_center'])} 人有头部")

                # 写入JSON文件
                with open(args.output_coords, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)

            # 在帧上绘制检测信息
            # 优先使用包含检测结果绘制的帧（骨架、边界框、头部圆圈）
            if latest_detections_frame is not None:
                display_frame = latest_detections_frame.copy()
            else:
                display_frame = frame.copy()

            # 绘制中心区域（绿色矩形）
            center_x1 = int(actual_w * 0.375)
            center_x2 = int(actual_w * 0.625)
            center_y1 = int(actual_h * 0.375)
            center_y2 = int(actual_h * 0.625)

            cv2.rectangle(display_frame, (center_x1, center_y1),
                        (center_x2, center_y2), (0, 255, 0), 2)

            # 如果有跟踪数据，绘制目标位置
            primary = tracker.get_primary_target()
            if primary:
                pos = (int(primary.smooth_x), int(primary.smooth_y))
                cv2.circle(display_frame, pos, 10, (0, 0, 255), -1)
                cv2.putText(display_frame, f"Target {primary.id}",
                           (pos[0] + 15, pos[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # 绘制十字线
                cv2.line(display_frame, (pos[0], 0), (pos[0], actual_h), (0, 0, 255), 1)
                cv2.line(display_frame, (0, pos[1]), (actual_w, pos[1]), (0, 0, 255), 1)

            # 绘制FPS信息
            elapsed = current_time - start_time
            stream_fps = frame_count / elapsed if elapsed > 0 else 0
            detect_fps = detect_count / elapsed if elapsed > 0 else 0

            info_lines = [
                f'Frame: {frame_count} | Detect: {detect_count}',
                f'Stream: {stream_fps:.1f} fps | Detect: {detect_fps:.1f} fps',
                f'Tracking: {len(tracker.get_all_targets())} person(s)',
            ]

            y = 30
            for line in info_lines:
                cv2.putText(display_frame, line, (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y += 25

            # 设置流帧
            streamer.update_frame(display_frame)

            # 打印统计
            if frame_count % 30 == 0:
                print(f'[Frame {frame_count}] 推流: {stream_fps:.1f} fps | '
                      f'检测: {detect_fps:.1f} fps | 目标: {len(tracker.get_all_targets())}')

    except KeyboardInterrupt:
        print('\n--> 用户中断')
    finally:
        # 清理
        streamer.stop()
        cap.release()
        detector.release()

        print()
        print("=" * 60)
        print(f'总帧数: {frame_count}')
        print(f'检测次数: {detect_count}')
        elapsed = time.time() - start_time
        if elapsed > 0:
            stream_fps = frame_count / elapsed
            detect_fps = detect_count / elapsed
            print(f'平均推流 FPS: {stream_fps:.2f}')
            print(f'平均检测 FPS: {detect_fps:.2f}')
        print("=" * 60)


if __name__ == '__main__':
    main()
