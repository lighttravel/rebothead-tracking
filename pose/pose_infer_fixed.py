"""
RK3576 YOLOv8 Pose 姿态估计推理脚本
参考: https://github.com/airockchip/rknn_model_zoo/tree/master/examples/yolov8_pose
基于 rk3576_infer_fixed.py 改造
"""

import cv2
import numpy as np
from rknnlite.api import RKNNLite

# 参数设置
OBJ_THRESH = 0.5  # 姿态估计置信度阈值
NMS_THRESH = 0.4  # NMS阈值
IMG_SIZE = (640, 640)

# COCO姿态估计类别（只有person）
CLASSES = ('person',)

# COCO 17个关键点名称
KEYPOINT_NAMES = [
    'nose',           # 0
    'left_eye',       # 1
    'right_eye',      # 2
    'left_ear',       # 3
    'right_ear',      # 4
    'left_shoulder',  # 5
    'right_shoulder', # 6
    'left_elbow',     # 7
    'right_elbow',    # 8
    'left_wrist',     # 9
    'right_wrist',    # 10
    'left_hip',       # 11
    'right_hip',      # 12
    'left_knee',      # 13
    'right_knee',     # 14
    'left_ankle',     # 15
    'right_ankle'     # 16
]

# 姿态调色板
pose_palette = np.array([
    [255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
    [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
    [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
    [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]
], dtype=np.uint8)

# 关键点颜色
kpt_color = pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]

# 骨架连接（关键点索引对）
skeleton = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
    [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
    [2, 4], [3, 5], [4, 6], [5, 7]
]

# 骨架颜色
limb_color = pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]


def sigmoid(x):
    """Sigmoid激活函数"""
    return 1 / (1 + np.exp(-x))


def softmax(x, axis=-1):
    """Softmax函数"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def process_output_branch(output, keypoints, kp_offset, model_w, model_h, stride):
    """
    处理单个输出分支

    Args:
        output: 单个分支输出 (1, 65, H, W)
        keypoints: 所有关键点数据 (1, 17, 3, 8400)
        kp_offset: 关键点索引偏移 (0/6400/8000)
        model_w, model_h: 特征图尺寸
        stride: 步长

    Returns:
        list of DetectBox
    """
    boxes = []

    # 分离边界框和置信度
    xywh = output[:, :64, :]  # (1, 64, H*W)
    conf = sigmoid(output[:, 64:, :])  # (1, 1, H*W)

    # 展平特征图
    xywh = xywh.reshape(1, 64, -1)  # (1, 64, H*W)
    conf = conf.reshape(1, -1)  # (1, H*W)

    # 遍历所有网格点
    for i in range(model_h * model_w):
        score = conf[0, i]

        if score > OBJ_THRESH:
            # DFL解码
            xywh_i = xywh[0, :, i].reshape(1, 4, 16, 1)
            data = np.array([i for i in range(16)]).reshape(1, 1, 16, 1)

            # Softmax + 期望值
            xywh_i = softmax(xywh_i, 2)
            xywh_i = np.multiply(data, xywh_i)
            xywh_i = np.sum(xywh_i, axis=2, keepdims=True).reshape(-1)

            # 计算网格坐标
            h = i // model_w
            w = i % model_w

            # 计算边界框（中心点格式）
            xywh_temp = xywh_i.copy()
            xywh_temp[0] = (w + 0.5) - xywh_i[0]
            xywh_temp[1] = (h + 0.5) - xywh_i[1]
            xywh_temp[2] = (w + 0.5) + xywh_i[2]
            xywh_temp[3] = (h + 0.5) + xywh_i[3]

            # 转换为xywh格式
            xywh_i[0] = ((xywh_temp[0] + xywh_temp[2]) / 2)
            xywh_i[1] = ((xywh_temp[1] + xywh_temp[3]) / 2)
            xywh_i[2] = (xywh_temp[2] - xywh_temp[0])
            xywh_i[3] = (xywh_temp[3] - xywh_temp[1])

            # 应用步长
            xywh_i = xywh_i * stride

            # 转换为xyxy格式
            xmin = (xywh_i[0] - xywh_i[2] / 2)
            ymin = (xywh_i[1] - xywh_i[3] / 2)
            xmax = (xywh_i[0] + xywh_i[2] / 2)
            ymax = (xywh_i[1] + xywh_i[3] / 2)

            # 提取关键点：keypoints shape is (1, 17, 3, 8400)
            # 计算全局索引
            global_i = i + kp_offset
            # 提取该网格点的所有17个关键点
            # keypoints[:, :, :, global_i] gives (1, 17, 3)
            keypoint = keypoints[:, :, :, global_i].reshape(17, 3)

            boxes.append(DetectBox(0, score, xmin, ymin, xmax, ymax, keypoint))

    return boxes


class DetectBox:
    """检测结果类"""
    def __init__(self, classId, score, xmin, ymin, xmax, ymax, keypoint):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.keypoint = keypoint


def iou(box1, box2):
    """计算IoU"""
    xmin = max(box1.xmin, box2.xmin)
    ymin = max(box1.ymin, box2.ymin)
    xmax = min(box1.xmax, box2.xmax)
    ymax = min(box1.ymax, box2.ymax)

    inner_width = xmax - xmin
    inner_height = ymax - ymin

    inner_width = inner_width if inner_width > 0 else 0
    inner_height = inner_height if inner_height > 0 else 0

    inner_area = inner_width * inner_height

    area1 = (box1.xmax - box1.xmin) * (box1.ymax - box1.ymin)
    area2 = (box2.xmax - box2.xmin) * (box2.ymax - box2.ymin)

    total = area1 + area2 - inner_area

    return inner_area / total if total > 0 else 0


def nms(boxes):
    """非极大值抑制"""
    keep_boxes = []

    # 按置信度排序
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
    """
    后处理函数

    Args:
        outputs: 模型输出列表
            outputs[0-2]: 3个尺度的边界框预测 (1, 65, H, W)
            outputs[3]: 关键点数据 (1, 17, 3, 8400)

    Returns:
        list of DetectBox
    """
    all_boxes = []
    keypoints = outputs[3]  # (1, 17, 3, 8400)

    # 处理3个尺度（P3, P4, P5）
    # outputs[0]: P3 (80x80, stride=8)
    # outputs[1]: P4 (40x40, stride=16)
    # outputs[2]: P5 (20x20, stride=32)

    # 计算每个分支的关键点索引偏移
    # 8400 = 80*80 + 40*40 + 20*20 = 6400 + 1600 + 400
    # P3 offset = 0
    # P4 offset = 6400
    # P5 offset = 6400 + 1600 = 8000
    strides = [8, 16, 32]
    kp_offsets = [0, 6400, 8000]

    for i, output in enumerate(outputs[:3]):
        model_h, model_w = output.shape[2], output.shape[3]
        boxes = process_output_branch(
            output, keypoints, kp_offsets[i], model_w, model_h, strides[i]
        )
        all_boxes.extend(boxes)

    # NMS
    final_boxes = nms(all_boxes)

    return final_boxes


def draw_pose(img, keypoints, skeleton_color=None):
    """
    在图像上绘制姿态骨架和头部圆圈

    Args:
        img: 图像
        keypoints: 关键点数组 (17, 3) - [x, y, confidence]
        skeleton_color: 骨架颜色，如果为None则使用默认颜色
    """
    # 绘制关键点
    for k, keypoint in enumerate(keypoints):
        x, y, conf = keypoint
        color_k = [int(x) for x in kpt_color[k]]

        if x != 0 and y != 0 and conf > 0.3:  # 置信度阈值
            cv2.circle(img, (int(x), int(y)), 5, color_k, -1, lineType=cv2.LINE_AA)

    # 绘制骨架
    for k, sk in enumerate(skeleton):
        pos1 = (int(keypoints[(sk[0] - 1), 0]), int(keypoints[(sk[0] - 1), 1]))
        pos2 = (int(keypoints[(sk[1] - 1), 0]), int(keypoints[(sk[1] - 1), 1]))

        conf1 = keypoints[(sk[0] - 1), 2]
        conf2 = keypoints[(sk[1] - 1), 2]

        if pos1[0] == 0 or pos1[1] == 0 or pos2[0] == 0 or pos2[1] == 0:
            continue

        if conf1 > 0.3 and conf2 > 0.3:
            if skeleton_color is None:
                color = [int(x) for x in limb_color[k]]
            else:
                color = skeleton_color
            cv2.line(img, pos1, pos2, color, thickness=2, lineType=cv2.LINE_AA)


def detect_head_circle(keypoints, bbox):
    """
    检测头部位置并绘制圆圈

    Args:
        keypoints: 关键点数组 (17, 3)
        bbox: 边界框 [xmin, ymin, xmax, ymax]

    Returns:
        head_center: 头部中心坐标 (x, y) 或 None
        head_radius: 头部半径 或 None
    """
    # 头部相关关键点索引：0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear
    head_kpt_indices = [0, 1, 2, 3, 4]

    # 收集可见的头部关键点
    visible_head_pts = []
    for idx in head_kpt_indices:
        x, y, conf = keypoints[idx]
        if x > 0 and y > 0 and conf > 0.3:  # 置信度阈值
            visible_head_pts.append((x, y))

    if len(visible_head_pts) == 0:
        return None, None

    # 方法1：基于可见关键点计算头部中心
    head_center_x = sum([pt[0] for pt in visible_head_pts]) / len(visible_head_pts)
    head_center_y = sum([pt[1] for pt in visible_head_pts]) / len(visible_head_pts)

    # 方法2：估算头部半径
    # 如果有耳朵，使用耳朵间距；否则使用眼睛间距
    left_ear = keypoints[3]
    right_ear = keypoints[4]
    left_eye = keypoints[1]
    right_eye = keypoints[2]

    if left_ear[0] > 0 and right_ear[0] > 0 and left_ear[2] > 0.3 and right_ear[2] > 0.3:
        # 使用耳朵间距，头部宽度约为耳朵间距的1.2倍
        ear_dist = abs(right_ear[0] - left_ear[0])
        head_radius = int(ear_dist * 0.9)  # 半径约为耳朵间距的90% (增大)
    elif left_eye[0] > 0 and right_eye[0] > 0 and left_eye[2] > 0.3 and right_eye[2] > 0.3:
        # 使用眼睛间距，头部宽度约为眼睛间距的3倍
        eye_dist = abs(right_eye[0] - left_eye[0])
        head_radius = int(eye_dist * 2.2)  # 半径约为眼睛间距的220% (增大)
    else:
        # 基于边界框估算（头部通常占人脸宽度的约15-20%）
        bbox_width = bbox[2] - bbox[0]
        head_radius = int(bbox_width * 0.18)  # 增大系数到18%

    # 确保半径在合理范围内
    head_radius = max(25, min(head_radius, 150))  # 最小25，最大150

    return (head_center_x, head_center_y), head_radius


def draw_head_circle(img, head_center, head_radius, color=(255, 0, 0), thickness=3):
    """
    在图像上绘制头部圆圈

    Args:
        img: 图像
        head_center: 头部中心 (x, y)
        head_radius: 头部半径
        color: 圆圈颜色 (B, G, R)
        thickness: 线条粗细
    """
    if head_center is not None and head_radius is not None:
        cv2.circle(img, (int(head_center[0]), int(head_center[1])),
                  head_radius, color, thickness, lineType=cv2.LINE_AA)
        # 绘制中心点
        cv2.circle(img, (int(head_center[0]), int(head_center[1])),
                  3, (0, 255, 0), -1, lineType=cv2.LINE_AA)


class YOLOv8PoseRK3576:
    """YOLOv8姿态估计 RK3576推理类"""

    def __init__(self, rknn_model='yolov8_pose.rknn'):
        """初始化 RKNN 模型"""
        self.rknn_lite = RKNNLite()
        print(f'--> Loading RKNN model: {rknn_model}')
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

    def infer(self, img_path):
        """
        推理主函数

        Args:
            img_path: 输入图像路径

        Returns:
            检测结果列表
        """
        img = cv2.imread(img_path)
        if img is None:
            print(f'Failed to load image: {img_path}')
            return None

        img_h, img_w = img.shape[:2]

        print(f"\n{'='*60}")
        print(f"Image: {img_path}")
        print(f"Size: {img_w} x {img_h} pixels")
        print(f"{'='*60}")

        # Letterbox预处理（与检测模型相同）
        shape = (img_h, img_w)
        scale = min(IMG_SIZE[0] / shape[0], IMG_SIZE[1] / shape[1])
        new_unpad = (int(round(shape[1] * scale)), int(round(shape[0] * scale)))
        dw_total, dh_total = IMG_SIZE[1] - new_unpad[0], IMG_SIZE[0] - new_unpad[1]
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

        # BGR -> RGB
        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)

        # 转换为RKNN格式
        input_data = np.expand_dims(img_rgb, axis=0)

        print(f'输入 shape: {input_data.shape}, dtype={input_data.dtype}')
        print('--> Running inference...')

        # 推理
        outputs = self.rknn_lite.inference(inputs=[input_data])

        print('--> Post-processing...')
        boxes = post_process(outputs)

        if not boxes:
            print("\n❌ No persons detected!")
            return None

        print(f"\n✅ Detected {len(boxes)} person(s):\n")

        # 坐标映射回原图
        results = []
        for i, box in enumerate(boxes):
            # 映射边界框
            box.xmin = (box.xmin - dw) / scale
            box.ymin = (box.ymin - dh) / scale
            box.xmax = (box.xmax - dw) / scale
            box.ymax = (box.ymax - dh) / scale

            # 限制在图像范围内
            box.xmin = max(0, min(box.xmin, img_w))
            box.ymin = max(0, min(box.ymin, img_h))
            box.xmax = max(0, min(box.xmax, img_w))
            box.ymax = max(0, min(box.ymax, img_h))

            # 映射关键点
            kp = box.keypoint.reshape(-1, 3)  # (17, 3)
            kp[..., 0] = (kp[..., 0] - dw) / scale  # x坐标
            kp[..., 1] = (kp[..., 1] - dh) / scale  # y坐标
            kp[..., 0] = np.clip(kp[..., 0], 0, img_w)
            kp[..., 1] = np.clip(kp[..., 1], 0, img_h)

            print(f"Person {i+1}:")
            print(f"  Confidence: {box.score:.2f}")
            print(f"  BBox: ({box.xmin:.1f}, {box.ymin:.1f}) to ({box.xmax:.1f}, {box.ymax:.1f})")

            # 统计可见关键点
            visible_kpts = np.sum(kp[:, 2] > 0.3)
            print(f"  Visible keypoints: {visible_kpts}/17")

            # 绘制边界框
            cv2.rectangle(img, (int(box.xmin), int(box.ymin)),
                         (int(box.xmax), int(box.ymax)), (0, 255, 0), 2)
            cv2.putText(img, f'person {box.score:.2f}',
                       (int(box.xmin), int(box.ymin) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # 绘制姿态
            draw_pose(img, kp)

            # 检测并绘制头部圆圈
            bbox = [box.xmin, box.ymin, box.xmax, box.ymax]
            head_center, head_radius = detect_head_circle(kp, bbox)

            if head_center is not None:
                # 绘制头部圆圈（红色）
                draw_head_circle(img, head_center, head_radius, color=(255, 0, 0), thickness=3)

                # 输出头部信息
                print(f"  📍 Head center: ({head_center[0]:.1f}, {head_center[1]:.1f}), radius: {head_radius}")
            else:
                print(f"  ⚠️  Head not detected (insufficient keypoints)")

            results.append({
                'confidence': float(box.score),
                'bbox': [float(box.xmin), float(box.ymin), float(box.xmax), float(box.ymax)],
                'keypoints': kp.tolist(),  # (17, 3) -> list
                'head_center': [float(head_center[0]), float(head_center[1])] if head_center else None,
                'head_radius': head_radius
            })

            print()

        # 保存结果
        output_path = 'result_pose_rk3576.jpg'
        cv2.imwrite(output_path, img)
        print(f"Result saved to: {output_path}")

        return results

    def __del__(self):
        if hasattr(self, 'rknn_lite'):
            self.rknn_lite.release()


def main():
    import sys

    model = YOLOv8PoseRK3576('yolov8_pose.rknn')

    # 支持命令行参数
    img_path = sys.argv[1] if len(sys.argv) > 1 else 'bus.jpg'

    model.infer(img_path)
    del model


if __name__ == '__main__':
    main()
