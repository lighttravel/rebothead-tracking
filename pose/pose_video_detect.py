"""
RK3576 YOLOv8 Pose 姿态估计视频流检测
支持 USB 摄像头、视频文件、RTSP 流
参考 video_detect.py 和 pose_infer_fixed.py
"""

import cv2
import numpy as np
import argparse
import time
from rknnlite.api import RKNNLite

# 参数设置
OBJ_THRESH = 0.5
NMS_THRESH = 0.4
IMG_SIZE = (640, 640)

# COCO 17个关键点名称
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

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
    """
    检测头部位置

    Args:
        keypoints: 关键点数组 (17, 3)
        bbox: 边界框 [xmin, ymin, xmax, ymax]

    Returns:
        head_center: 头部中心坐标 (x, y) 或 None
        head_radius: 头部半径 或 None
    """
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
        head_radius = int(ear_dist * 0.9)  # 增大到90%
    elif left_eye[0] > 0 and right_eye[0] > 0 and left_eye[2] > 0.3 and right_eye[2] > 0.3:
        eye_dist = abs(right_eye[0] - left_eye[0])
        head_radius = int(eye_dist * 2.2)  # 增大到220%
    else:
        bbox_width = bbox[2] - bbox[0]
        head_radius = int(bbox_width * 0.18)  # 增大到18%

    head_radius = max(25, min(head_radius, 150))  # 最小25，最大150

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
        """
        单帧检测

        Args:
            frame: numpy array (H, W, 3) BGR

        Returns:
            results: 检测结果列表
            img_draw: 绘制后的图像
        """
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


class FPSCalculator:
    """FPS计算器"""

    def __init__(self, window_size=30):
        self.window_size = window_size
        self.inference_times = []

    def update(self, inference_time_ms):
        self.inference_times.append(inference_time_ms)
        if len(self.inference_times) > self.window_size:
            self.inference_times.pop(0)

    def get_fps(self):
        if not self.inference_times:
            return 0.0
        avg_time = sum(self.inference_times) / len(self.inference_times)
        return 1000.0 / avg_time if avg_time > 0 else 0.0

    def get_avg_time(self):
        if not self.inference_times:
            return 0.0
        return sum(self.inference_times) / len(self.inference_times)


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Pose 视频流检测')
    parser.add_argument('--source', type=str, default='0',
                       help='视频源（设备号/文件路径/RTSP URL）')
    parser.add_argument('--model', type=str, default='yolov8_pose.rknn',
                       help='RKNN模型路径')
    parser.add_argument('--output', type=str, default=None,
                       help='输出视频路径（可选）')
    parser.add_argument('--no-display', action='store_true',
                       help='禁用显示窗口')
    parser.add_argument('--input-size', type=int, default=640,
                       help='模型输入尺寸')
    parser.add_argument('--conf-thresh', type=float, default=0.5,
                       help='置信度阈值')

    args = parser.parse_args()

    # 更新全局阈值
    global OBJ_THRESH
    OBJ_THRESH = args.conf_thresh

    # 初始化检测器
    detector = YOLOv8PoseDetector(args.model, args.input_size)

    # 初始化视频源
    source = args.source
    if source.isdigit():
        source = int(source)
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        print(f'--> Opening camera {source}')
    else:
        cap = cv2.VideoCapture(source)
        print(f'--> Opening video/file: {source}')

    if not cap.isOpened():
        print('❌ Failed to open video source!')
        return

    # 视频写入器
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))
        print(f'--> Output will be saved to: {args.output}')

    # FPS计算器
    fps_calc = FPSCalculator()
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print('--> Starting detection...')
    print('Press "q" to quit')

    while True:
        ret, frame = cap.read()
        if not ret:
            print('--> End of video stream')
            break

        frame_count += 1
        start_time = time.time()

        # 检测
        results, img_draw = detector.detect_frame(frame)

        inference_time = (time.time() - start_time) * 1000
        fps_calc.update(inference_time)

        # 绘制FPS信息
        fps = fps_calc.get_fps()
        avg_time = fps_calc.get_avg_time()

        cv2.putText(img_draw, f'FPS: {fps:.1f}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img_draw, f'Inference: {avg_time:.1f}ms', (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if results:
            cv2.putText(img_draw, f'Persons: {len(results)}', (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 写入视频
        if writer:
            writer.write(img_draw)

        # 显示
        if not args.no_display:
            cv2.imshow('YOLOv8 Pose Detection', img_draw)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('--> Quitting...')
                break

        if frame_count % 30 == 0:
            print(f'Processed {frame_count}/{total_frames} frames, FPS: {fps:.1f}')

    # 清理
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print(f'\n{"="*60}')
    print(f'Total frames processed: {frame_count}')
    print(f'Average FPS: {fps:.1f}')
    print(f'Average inference time: {avg_time:.1f}ms')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
