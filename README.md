# rebothead-tracking

基于 RK3576 NPU 和高擎 5047-36 伺服电机的 YOLOv8 头部跟踪系统。

## 1. 项目概述

本项目实现了一个实时人体头部跟踪系统，适用于机器人云台、摄像设备等应用场景。系统采用分层架构设计，上位机负责视觉检测和目标跟踪算法，电机控制器负责执行电机运动指令。

### 核心特性

- **实时检测**：RK3576 NPU 加速 YOLOv8 推理，检测帧率 8fps
- **平滑跟踪**：IOU 目标匹配 + 5帧移动平均滤波，减少抖动
- **串口通信**：自定义二进制协议，115200 波特率
- **双轴控制**：支持 Yaw（偏航）/ Pitch（俯仰）两轴联动
- **MJPEG 推流**：支持局域网视频流预览

### 适用场景

- 机器人云台头部跟随
- 智能摄像监控系统
- 直播自动跟拍设备
- 人机交互演示系统

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           系统架构图                                      │
└─────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐        USB        ┌──────────────────┐
  │   PC 上位机   │ ◄──────────────► │   ARM-Link       │
  │  (调试/烧录)   │                  │   CMSIS-DAP      │
  └──────────────┘                   └────────┬─────────┘
                                               │
                                               ▼
                                      ┌──────────────────┐
                                      │  STM32 H723      │
                                      │  电机控制器       │
                                      │  - PWM 输出      │
                                      │  - 串口解析       │
                                      │  - 电机驱动       │
                                      └────────┬─────────┘
                                               │
                                               ▼
                                      ┌──────────────────┐
                                      │  高擎 5047-36 ×2 │
                                      │  电机21: Yaw     │
                                      │  电机22: Pitch   │
                                      └──────────────────┘

  ┌──────────────┐        USB        ┌──────────────────┐
  │   摄像头       │ ◄──────────────► │  RK3576 开发板   │
  │  (USB/MIPI)   │                  │  - NPU 推理      │
  └──────────────┘                  │  - 跟踪算法       │
                                    │  - 串口发送        │
                                    │  - MJPEG 推流     │
                                    └──────────────────┘
```

## 3. 硬件清单

| 设备 | 型号/规格 | 用途 |
|------|----------|------|
| 主控板 | Industio RK3576 | 视觉检测、算法运行 |
| 伺服电机 | 高擎机电 5047-36 ×2 | Yaw/Pitch 双轴驱动 |
| 电机控制器 | 达妙 STM32 H723 | 接收串口指令、控制电机 |
| 下载器 | ARM-Link (CMSIS-DAP) | STM32 程序烧录 |
| 摄像头 | USB 或 MIPI CSI 接口 | 视频采集 |

### 串口连接

```
RK3576 开发板                    STM32 H723
  UART2_TX (Pin XX) ─────────► UART7_RX
  UART2_RX (Pin XX) ◄───────── UART7_TX
  GND ──────────────────────── GND
```

**注意**：串口波特率配置为 115200，8N1 格式。

## 4. 软件架构

### 4.1 核心模块

```
person_tracking.py          # 主程序：YOLOv8 检测 + 坐标输出
├── 初始化 RKNNLite 模型
├── 打开摄像头 /dev/video36
├── YOLOv8 推理 (8fps)
├── IOU 目标匹配
├── 5帧移动平均平滑
└── 输出坐标 → tracker_coords.json

horizontal_tracking.py      # 水平跟踪模块
├── 读取 tracker_coords.json
├── 计算目标区域 (左/中/右)
├── 计算偏航增量
└── 调用 motor_controller

bidirectional_tracking.py   # 双向跟踪模块
├── 水平跟踪 + 垂直跟踪
├── 计算 Yaw/Pitch 增量
└── 统一电机控制接口

motor_controller.py         # 串口通信模块
├── 串口初始化 /dev/ttyS8
├── 打包控制指令 (0x10/0x11)
├── 发送二进制数据包
└── 校验和计算
```

### 4.2 文件结构

```
.
├── person_tracking.py          # 主检测程序 (936 行)
├── horizontal_tracking.py      # 水平跟踪 (308 行)
├── bidirectional_tracking.py   # 双向跟踪 (392 行)
├── motor_controller.py         # 电机控制器 (185 行)
├── test_motor_rk3576_native.py # 电机测试程序 (137 行)
├── yolov8.rknn                # RKNN 模型文件
├── 电机21_22串口控制说明.md     # 串口协议文档
└── README.md                   # 本文档
```

### 4.3 串口协议

通信采用定长二进制帧，格式如下：

```
| 帧头 (1B) | 长度 (1B) | 命令 (1B) | 数据 (8B) | 校验 (1B) |
|   0xAA   |   0x0A   |  0x10/0x11| yaw+pitch|   SUM    |
```

**字段说明**：

| 字段 | 值 | 说明 |
|-----|-----|------|
| 帧头 | 0xAA | 固定帧头 |
| 长度 | 0x0A | 10 = 1(命令) + 8(数据) + 1(校验) |
| 命令 | 0x10 | 绝对位置模式 |
| 命令 | 0x11 | 增量位置模式 |
| 数据 | 8字节 | float × 2 (小端序，弧度制) |
| 校验 | 1字节 | 命令 + 数据字节和 (模 256) |

**数据格式**：

- 浮点数：IEEE 754 单精度，小端序
- 角度单位：弧度 (rad)
- 范围：±12.566 rad ≈ ±720°

**示例**：

```python
# 发送 0.5 rad 偏航、0.3 rad 俯仰
yaw = 0.5
pitch = 0.3
yaw_bytes = struct.pack('<f', yaw)      # b'\x00\x00\x3F\x00'
pitch_bytes = struct.pack('<f', pitch)  # b'\xCD\xCC\x9C\x3E'
packet = bytes([0xAA, 0x0A, 0x10]) + yaw_bytes + pitch_bytes
checksum = (0x10 + sum(yaw_bytes + pitch_bytes)) % 256
packet += bytes([checksum])
```

## 5. 快速开始

### 5.1 环境准备

```bash
# 1. 安装依赖
pip install opencv-python numpy rknnlite2

# 2. 上传模型文件
adb push yolov8.rknn /userdata/

# 3. 连接串口
# RK3576 UART2 <-> STM32 UART7
# TX -> RX, RX -> TX, GND -> GND
```

### 5.2 运行程序

```bash
# 单独启动跟踪系统
python3 person_tracking.py

# 单独测试电机
python3 test_motor_rk3576_native.py

#跟踪运行（这步就能开启头部追踪了！！！安装依赖后可直接这一步）
python3 start_tracking.py

# 查看视频流
# 浏览器访问 http://<RK3576_IP>:8080/stream
```

### 5.3 调试串口

```bash
# 查看串口设备
ls /dev/ttyS*

# 测试串口通信
cat /dev/ttyS8

# 示波器抓包验证时序
# 波特率: 115200
# 数据位: 8
# 停止位: 1
# 无校验
```

## 6. 关键实现

### 6.1 坐标归一化

检测输出的归一化坐标范围 [-1, 1]，便于电机控制：

```python
# 图像中心为原点
center_x = (box[0] + box[2]) / 2 / img_width * 2 - 1  # -1 ~ 1
center_y = (box[1] + box[3]) / 2 / img_height * 2 - 1  # -1 ~ 1
```

### 6.2 平滑滤波

使用 5 帧移动平均减少抖动：

```python
from collections import deque

class MovingAverage:
    def __init__(self, window=5):
        self.queue = deque(maxlen=window)
    
    def update(self, value):
        self.queue.append(value)
        return sum(self.queue) / len(self.queue)
```

### 6.3 目标匹配

使用 IOU (Intersection over Union) 进行帧间目标关联：

```python
def iou_match(prev_box, curr_box):
    # 计算两个框的交并比
    # 大于阈值则认为是同一目标
    return iou > 0.5
```

## 7. 已知问题

1. **串口速率**：RK3576 与 STM32 通信波特率固定为 115200，高速运动时可能丢包
2. **坐标抖动**：目标边缘检测不稳定时，平滑滤波效果有限
3. **模型量化**：rknn 模型为 INT8 量化，精度略有损失

## 8. 许可证

MIT License

## 9. 参考资料

- [RKNN Lite Python SDK](https://github.com/rockchip-linux/rknn-toolkit2)
- [YOLOv8 官方文档](https://docs.ultralytics.com/)
- [高擎 5047-36 电机手册](./电机21_22串口控制说明.md)
- [达妙 STM32 开发板资料](http://www.damov.com/)