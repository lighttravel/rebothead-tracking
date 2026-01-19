"""
RK3576 YOLOv8 Pose 姿态估计 FPS 性能测试脚本
简单测试处理器每秒能处理多少帧
"""

import cv2
import numpy as np
import time
import sys
from rknnlite.api import RKNNLite

# 参数设置
OBJ_THRESH = 0.5
NMS_THRESH = 0.4
IMG_SIZE = (640, 640)


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


def preprocess_image(img):
    """预处理图像（Letterbox）"""
    img_h, img_w = img.shape[:2]

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

    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(img_rgb, axis=0)

    return input_data, scale, dw, dh


def benchmark_fps(rknn_lite, input_data, num_iterations=100, warmup=10):
    """
    测试 FPS
    """
    print(f'\n{"="*70}')
    print(f'RK3576 YOLOv8 Pose 姿态估计 FPS 性能测试')
    print(f'{"="*70}')
    print(f'测试模式: {num_iterations} 次迭代')
    print(f'预热次数: {warmup}')
    print(f'输入尺寸: {IMG_SIZE[0]} x {IMG_SIZE[1]}')
    print(f'{"="*70}\n')

    # 预热
    print(f'🔥 预热中 ({warmup} 次)...')
    for i in range(warmup):
        outputs = rknn_lite.inference(inputs=[input_data])
    print('   ✓ 预热完成\n')

    # 性能测试
    print(f'🚀 开始测试...')

    inference_times = []
    total_start = time.time()

    for i in range(num_iterations):
        start = time.time()
        outputs = rknn_lite.inference(inputs=[input_data])
        end = time.time()

        inference_time = (end - start) * 1000
        inference_times.append(inference_time)

        if (i + 1) % 20 == 0 or i == 0:
            print(f'   进度: {i+1}/{num_iterations} - {inference_time:.2f}ms')

    total_end = time.time()
    total_time = total_end - total_start

    # 统计结果
    inference_times = np.array(inference_times)
    avg_time = np.mean(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)

    fps_avg = 1000 / avg_time
    fps_actual = num_iterations / total_time

    print(f'\n{"="*70}')
    print(f'📊 测试结果')
    print(f'{"="*70}\n')

    print(f'⏱️  推理时间:')
    print(f'   平均: {avg_time:.2f} ms/frame')
    print(f'   最快: {min_time:.2f} ms/frame')
    print(f'   最慢: {max_time:.2f} ms/frame')
    print(f'   波动: {max_time - min_time:.2f} ms')

    print(f'\n🎯 FPS (帧率):')
    print(f'   理论 FPS: {fps_avg:.2f} fps (1秒处理 {fps_avg:.0f} 帧)')
    print(f'   实际 FPS: {fps_actual:.2f} fps (包含所有开销)')

    print(f'\n📈 性能总结:')
    print(f'   总迭代次数: {num_iterations}')
    print(f'   总耗时: {total_time:.2f} 秒')
    print(f'   处理器能力: 1秒可处理 {fps_avg:.1f} 帧')

    # 验证检测结果
    boxes = post_process(outputs)
    if boxes:
        print(f'\n✓ 检测正常: 检测到 {len(boxes)} 个人')
    else:
        print(f'\n⚠ 未检测到人')

    print(f'\n{"="*70}')
    print(f'✅ 测试完成！')
    print(f'{"="*70}\n')

    return fps_avg, fps_actual


def main():
    # 参数
    model_path = 'yolov8_pose.rknn'
    img_path = 'bus.jpg'
    num_iterations = 100

    # 命令行参数
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    if len(sys.argv) > 2:
        num_iterations = int(sys.argv[2])

    print(f'\nRK3576 YOLOv8 Pose FPS 测试工具')
    print(f'='*50)

    # 初始化模型
    print(f'\n加载模型: {model_path}')
    rknn_lite = RKNNLite()
    ret = rknn_lite.load_rknn(model_path)
    if ret != 0:
        print('❌ 加载模型失败!')
        return

    ret = rknn_lite.init_runtime()
    if ret != 0:
        print('❌ 初始化运行时失败!')
        return

    print(f'✓ 模型加载成功')

    # 加载并预处理图片
    print(f'\n加载图片: {img_path}')
    img = cv2.imread(img_path)
    if img is None:
        print(f'❌ 无法加载图片: {img_path}')
        return

    img_h, img_w = img.shape[:2]
    print(f'✓ 图片尺寸: {img_w} x {img_h}')

    input_data, scale, dw, dh = preprocess_image(img)
    print(f'✓ 预处理完成')

    # 开始测试
    fps_avg, fps_actual = benchmark_fps(rknn_lite, input_data, num_iterations)

    # 清理
    rknn_lite.release()


if __name__ == '__main__':
    main()
