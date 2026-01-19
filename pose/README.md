# YOLOv8 Pose 姿态估计 RK3576 部署
 终端 1 - 姿态检测：
  python3 pose_head_tracking.py --source 36 --detect-fps 10

终端 2 - 电机控制：
  python3 bidirectional_tracking.py --coords tracker_coords.json --serial /dev/ttyS8 --kp 0.8 --max-delta 0.2 --interval 0.08 
本文件夹包含 YOLOv8 姿态估计模型在 RK3576 上的部署程序，支持人体姿态识别和**头部检测画圆**。

## 文件说明

| 文件 | 用途 |
|------|------|
| `yolov8_pose.rknn` | RKNN 姿态估计模型文件 |
| `pose_infer_fixed.py` | 单张图片姿态估计推理 + **头部圆圈识别** ⭐ |
| `pose_video_detect.py` | 视频流/摄像头实时姿态检测 + **头部圆圈识别** ⭐ |
| `pose_benchmark.py` | 性能测试工具 |
| `pose_head_tracking.py` | **实时头部跟踪 + JSON输出 + MJPEG流** 🎥新增 |
| `start_pose_tracking.py` | **头部跟踪系统启动脚本** 🚀新增 |
| `bidirectional_tracking.py` | 双向电机控制算法（水平+垂直） |
| `motor_controller.py` | 串口电机控制器 |

## 功能特性

### ✅ 姿态估计
- 检测17个COCO人体关键点
- 绘制完整的人体骨架（20条连接线）
- 支持多人同时检测

### ✅ 头部识别 ⭐新增
- 自动识别头部位置并画圆圈标记
- 基于鼻子、眼睛、耳朵关键点计算头部中心
- 智能估算头部半径（自适应）
- 红色圆圈 + 绿色中心点标注

## 快速开始

### 1. 单图片推理（带头部识别）

```bash
cd pose

# 使用默认图片测试
python3 pose_infer_fixed.py bus.jpg

# 指定图片
python3 pose_infer_fixed.py /path/to/image.jpg
```

**输出**:
- 终端显示检测到的人数、关键点数量
- **头部中心坐标和半径**
- 保存可视化结果到 `result_pose_rk3576.jpg`
- **图像中包含：边界框、骨架、头部圆圈**

### 2. 摄像头实时检测（带头部识别）

```bash
# USB 摄像头（设备号36）
python3 pose_video_detect.py --source 36

# 视频文件检测
python3 pose_video_detect.py --source video.mp4

# 保存检测结果
python3 pose_video_detect.py --source 36 --output result.mp4
```

**按键**:
- 按 `q` 键退出

### 3. 性能测试

```bash
# 测试100次迭代
python3 pose_benchmark.py bus.jpg 100
```

### 4. 实时头部跟踪（带电机控制）🎥新增

**快速启动（一键启动检测+电机控制）**:

```bash
# 使用默认配置
python3 start_pose_tracking.py

# 自定义配置
CAMERA_SOURCE=36 DETECT_FPS=8 python3 start_pose_tracking.py
```

**手动启动（分步运行）**:

```bash
# 终端1: 启动姿态检测和头部跟踪
python3 pose_head_tracking.py \
    --source 36 \
    --detect-fps 8 \
    --stream-fps 30 \
    --port 8080 \
    --output-coords tracker_coords.json

# 终端2: 启动电机控制
python3 bidirectional_tracking.py \
    --coords tracker_coords.json \
    --serial /dev/ttyS8 \
    --baudrate 115200 \
    --max-lost-frames 10
```

**功能**:
- ✅ 实时检测人体姿态并提取头部位置
- ✅ 输出头部坐标到JSON文件（供电机控制使用）
- ✅ MJPEG视频流实时查看检测结果
- ✅ 自动选择距离图像中心最近的人头作为跟踪目标
- ✅ 位置平滑处理，减少抖动
- ✅ 可配置检测/推流帧率

## 头部识别算法 ⭐

### 原理
基于COCO 17个关键点中的头部相关点：
- **索引0**: nose (鼻子)
- **索引1-2**: left_eye, right_eye (左右眼)
- **索引3-4**: left_ear, right_ear (左右耳)

### 头部中心计算
1. 收集所有可见的头部关键点（置信度>0.3）
2. 计算这些点的平均位置作为头部中心

### 头部半径估算
1. **优先级1**: 使用耳朵间距（如果有）
   - `radius = ear_distance × 0.9`
2. **优先级2**: 使用眼睛间距（如果有）
   - `radius = eye_distance × 2.2`
3. **优先级3**: 基于边界框宽度估算
   - `radius = bbox_width × 0.18`
4. 限制范围：25-150像素

### 可视化
- **红色圆圈**: 标记头部区域（3像素粗）
- **绿色圆点**: 标记头部中心点

## COCO 17个关键点

| 索引 | 名称 | 索引 | 名称 |
|------|------|------|------|
| 0 | nose ⭐ | 9 | right_wrist |
| 1 | left_eye ⭐ | 10 | left_hip |
| 2 | right_eye ⭐ | 11 | right_hip |
| 3 | left_ear ⭐ | 12 | left_knee |
| 4 | right_ear ⭐ | 13 | right_knee |
| 5 | left_shoulder | 14 | left_ankle |
| 6 | right_shoulder | 15 | right_ankle |
| 7 | left_elbow | 16 | (unused) |
| 8 | right_elbow | 17 | (unused) |

**头部相关**: 0-4 (nose, eyes, ears) - 用于头部识别

## 性能参数

- **输入尺寸**: 640 × 640
- **置信度阈值**: 0.5（可修改代码调整）
- **NMS阈值**: 0.4
- **关键点置信度阈值**: 0.3
- **头部检测最小关键点**: 至少1个头部关键点可见

## 可视化说明

姿态估计的可视化包括:
- **边界框**: 绿色矩形框
- **置信度**: 红色文字
- **关键点**: 彩色圆点（17个）
- **骨架**: 连接线（20条）
- **头部圆圈**: 红色圆圈 + 绿色中心点 ⭐新增

## 输出数据格式

每个检测结果包含:
```python
{
    "confidence": 0.85,
    "bbox": [x1, y1, x2, y2],
    "keypoints": [
        [x, y, conf],  # 17个关键点
        ...
    ],
    "head_center": [x, y],  # 头部中心坐标 ⭐新增
    "head_radius": 45       # 头部半径 ⭐新增
}
```

## 头部跟踪系统 JSON 输出格式 🎥

`pose_head_tracking.py` 输出的 JSON 文件格式（用于电机控制）:

```json
{
  "primary_target": {
    "id": 0,
    "position": {
      "center": [640.5, 360.2],
      "relative": {
        "x": -0.02,
        "y": 0.05
      }
    },
    "head_radius": 85,
    "confidence": 0.92,
    "lost_frames": 0,
    "timestamp": "2026-01-19T12:34:56.789"
  },
  "all_detections": [
    {
      "confidence": 0.92,
      "bbox": [100, 150, 300, 500],
      "head_center": [200, 250],
      "head_radius": 85
    }
  ],
  "frame_info": {
    "width": 1280,
    "height": 720,
    "frame_count": 1234,
    "detect_count": 98,
    "timestamp": "2026-01-19T12:34:56.789"
  }
}
```

**坐标说明**:
- `center`: 头部中心的像素坐标 [x, y]
- `relative`: 相对位置（-1到1，0为图像中心）
  - `x`: 负数=左侧，正数=右侧
  - `y`: 负数=上方，正数=下方
- `lost_frames`: 连续丢失帧数（超过阈值则回到零位）

## 常见问题

### Q: 头部圆圈不准确怎么办？
- 确保人物正面或侧面朝向摄像头
- 头部被遮挡时可能导致圆圈偏移
- 可以调整 `detect_head_circle()` 中的置信度阈值

### Q: 如何调整检测阈值？
修改脚本中的 `OBJ_THRESH` 变量（默认0.5）

### Q: 头部圆圈太大/太小？
可以修改 `detect_head_circle()` 函数中的半径计算系数

### Q: 如何只检测头部？
修改脚本，在检测后只绘制头部圆圈，不绘制骨架

### Q: 如何调整电机跟踪参数？
修改 `bidirectional_tracking.py` 中的参数：
- `--center-start / --center-end`: 中心区域边界（默认0.375-0.625）
- `--deadzone`: 死区大小（默认0.02）
- `--max-delta`: 单次最大转动角度（默认0.15弧度）
- `--kp`: 比例系数（默认0.5，越大响应越快）
- `--max-lost-frames`: 目标丢失多少帧后回到零位（默认10）

### Q: 串口连接失败怎么办？
1. 检查硬件连接：TX/RX/GND 是否正确连接
2. 尝试其他串口：`/dev/ttyS1`, `/dev/ttyS3`, `/dev/ttyS4`
3. 确认STM32固件已烧录并运行
4. 检查波特率设置（默认115200）

## 与目标检测的区别

| 特性 | 目标检测 (YOLOv8) | 姿态估计 (YOLOv8 Pose) |
|------|------------------|----------------------|
| 输出类别 | 80个类别 | 只有person |
| 输出内容 | 边界框 + 类别 | 边界框 + 17个关键点 + **头部圆圈** |
| 模型输出 | 9个张量 | 4个张量 |
| 后处理 | DFL + NMS | DFL + NMS + 关键点解码 + **头部检测** |

## 电机控制系统架构 🎥

### 系统组成

1. **pose_head_tracking.py**: 姿态检测 + 头部跟踪
   - 实时运行YOLOv8 Pose检测
   - 提取头部中心坐标
   - 写入JSON文件
   - 提供MJPEG视频流

2. **bidirectional_tracking.py**: 电机控制算法
   - 读取JSON坐标文件
   - 计算电机转动角度（水平+垂直）
   - 发送增量命令到电机

3. **motor_controller.py**: 串口通信
   - 与STM32通信
   - 发送电机控制命令（0x10绝对位置，0x11增量）

4. **start_pose_tracking.py**: 启动脚本
   - 同时启动检测和电机控制
   - 管理进程生命周期
   - 日志记录

### 工作流程

```
相机输入 → YOLOv8 Pose检测 → 头部位置提取 → JSON文件
                                                    ↓
视频流显示 ← MJPEG服务器 ← 跟踪可视化          电机控制算法 ← JSON文件
                                                    ↓
                                              串口通信 → STM32
                                                    ↓
                                              电机21(偏航) + 电机22(俯仰)
```

### 电机控制策略

- **中心区域**: 37.5%-62.5%（水平和垂直）
- **死区**: ±2%（中心区域内不动作）
- **控制模式**: 增量控制（每次转动一小步）
- **响应速度**: 比例系数kp=0.5（可调）
- **丢失处理**: 连续丢失10帧后回到零位

## 注意事项

1. **模型文件**: 确保 `yolov8_pose.rknn` 在同一目录下
2. **测试图片**: 可以使用 `bus.jpg` 作为测试图片
3. **摄像头设备**: 默认 USB 摄像头是 `/dev/video36`
4. **RKNN Runtime**: 需要安装 `rknn-toolkit-lite2`
5. **串口连接**: 电机控制需要正确连接TX/RX/GND到STM32
6. **日志文件**: 检测和电机控制日志保存在 `logs/` 目录
7. **视频流访问**: 确保防火墙允许访问HTTP端口（默认8080）

## 参考

- 官方实现: `rknn_model_zoo/examples/yolov8_pose/python/yolov8_pose.py`
- COCO数据集: https://cocodataset.org/#keypoints-2020
