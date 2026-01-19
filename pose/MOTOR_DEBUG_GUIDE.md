# 电机控制调试和测试指南

## 问题诊断步骤

### 步骤 1: 检查 Python 依赖

在 RK3576 上运行：

```bash
# 检查 pyserial 是否已安装
python3 -c "import serial; print('pyserial 已安装')"

# 如果报错，安装 pyserial
pip3 install pyserial
# 或者
pip install pyserial
```

### 步骤 2: 测试串口连接

```bash
# 查看可用的串口设备
ls -la /dev/ttyS*

# 应该看到类似：
# crw-rw---- 1 root dialout 4, 64 Jan 19 09:00 ttyS0
# crw-rw---- 1 root dialout 4, 65 Jan 19 09:00 ttyS1
# ...
# crw-rw---- 1 root dialout 4, 72 Jan 19 09:00 ttyS8  <-- 串口8
```

### 步骤 3: 测试电机控制器

```bash
cd pose
python3 motor_controller.py
```

**期望输出**：
```
============================================================
电机控制测试
============================================================

✅ 串口已连接: /dev/ttyS8 @ 115200
[测试 1/6] 电机21 转动到 +0.5 弧度
已发送绝对位置: 电机21=+0.500rad, 电机22=+0.000rad
数据包: AA0A1000000000E03F0000000000000046
...
```

**如果失败**：
```
❌ 串口连接失败: [Errno 2] could not open port '/dev/ttyS8'
```

尝试其他串口：
```bash
# 编辑 motor_controller.py，修改串口
controller = MotorController(serial_port='/dev/ttyS1')  # 改为 ttyS1
```

### 步骤 4: 测试双向跟踪

```bash
# 先运行姿态检测（终端1）
python3 pose_head_tracking.py --source 36

# 再运行电机控制（终端2）
python3 bidirectional_tracking.py \
    --coords tracker_coords.json \
    --serial /dev/ttyS8 \
    --baudrate 115200
```

## 常见问题排查

### 问题 1: 没有安装 pyserial

**错误信息**：
```
ModuleNotFoundError: No module named 'serial'
```

**解决方法**：
```bash
pip3 install pyserial
```

### 问题 2: 串口权限不足

**错误信息**：
```
PermissionError: [Errno 13] Permission denied: '/dev/ttyS8'
```

**解决方法**：
```bash
# 方法1: 将用户添加到 dialout 组
sudo usermod -a -G dialout $USER
# 注销后重新登录生效

# 方法2: 临时更改权限
sudo chmod 666 /dev/ttyS8
```

### 问题 3: 串口设备不存在

**错误信息**：
```
serial.SerialException: [Errno 2] could not open port '/dev/ttyS8'
```

**解决方法**：
```bash
# 查看可用的串口
ls -la /dev/tty*

# 根据实际设备修改代码中的串口号
python3 bidirectional_tracking.py --serial /dev/ttyS1
```

### 问题 4: 串口打开但没有响应

**可能原因**：
1. STM32 固件未运行
2. TX/RX 接线错误
3. 波特率不匹配

**排查步骤**：

```bash
# 1. 检查串口是否被占用
sudo lsof | grep ttyS8

# 2. 使用 minicom 测试串口
sudo apt-get install minicom
sudo minicom -D /dev/ttyS8 -b 115200
# 按 Ctrl+A, Z, X 退出

# 3. 检查 STM32 固件
# 确保 STM32 已烧录固件并正在运行
```

### 问题 5: 电机不转动

**可能原因**：
1. 数据包格式错误
2. 校验和计算错误
3. STM32 未正确解析命令

**调试方法**：

在 `motor_controller.py` 中添加调试输出：
```python
def send_motor_delta(self, yaw_delta, pitch_delta):
    print(f"🔍 调试: 发送增量命令")
    print(f"   yaw_delta={yaw_delta}, pitch_delta={pitch_delta}")
    # ... 现有代码
```

在 STM32 端添加串口接收调试：
```c
// 在 STM32 代码中添加
printf("Received: %02X %02X %02X ...\n", buf[0], buf[1], buf[2]);
```

## 快速测试脚本

创建 `test_motor_simple.py`：

```python
#!/usr/bin/env python3
"""简单的电机测试脚本"""
import struct
import serial
import time

try:
    # 打开串口
    ser = serial.Serial(
        port='/dev/ttyS8',
        baudrate=115200,
        timeout=1,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE
    )
    print(f"✅ 串口已打开: {ser.name}")

    # 测试1: 发送零位命令
    yaw = 0.0
    pitch = 0.0
    yaw_bytes = struct.pack('<f', yaw)
    pitch_bytes = struct.pack('<f', pitch)

    header = 0xAA
    cmd = 0x10  # 绝对位置
    payload = yaw_bytes + pitch_bytes
    checksum = (cmd + sum(payload)) % 256
    packet = bytes([header, 0x0A, cmd]) + payload + bytes([checksum])

    ser.write(packet)
    print(f"✅ 已发送零位命令")
    print(f"   数据包: {packet.hex().upper()}")

    time.sleep(1)
    ser.close()
    print("✅ 测试完成")

except Exception as e:
    print(f"❌ 错误: {e}")
```

运行测试：
```bash
python3 test_motor_simple.py
```

## 硬件连接检查

### RK3576 与 STM32 连接

```
RK3576                    STM32
-------                  ------
UART2_TX (GPIO)  ──────▶  UART7_RX
UART2_RX (GPIO)  ◀─────  UART7_TX
GND             ────────  GND
```

### 使用万用表测试

1. **测试 TX/RX 连通性**：
   - 万用表设置：蜂鸣档
   - 一端接 RK3576 TX，另一端接 STM32 RX
   - 应该听到蜂鸣声

2. **测试 GND 连通性**：
   - 万用表设置：蜂鸣档
   - 一端接 RK3576 GND，另一端接 STM32 GND
   - 应该听到蜂鸣声

## 完整的系统启动流程

### 1. 安装依赖

```bash
# 在 RK3576 上
pip3 install opencv-python numpy pyserial rknnlite2
```

### 2. 测试串口和电机

```bash
cd pose
python3 motor_controller.py
```

### 3. 运行完整系统

```bash
# 方式1: 使用启动脚本（推荐）
python3 start_pose_tracking.py

# 方式2: 手动启动（两个终端）
# 终端1
python3 pose_head_tracking.py --source 36

# 终端2
python3 bidirectional_tracking.py \
    --coords tracker_coords.json \
    --serial /dev/ttyS8 \
    --baudrate 115200
```

## 监控和调试

### 查看串口数据

```bash
# 使用 cat 查看串口输出（需要权限）
sudo cat /dev/ttyS8

# 或者使用 minicom
sudo minicom -D /dev/ttyS8 -b 115200
```

### 查看系统日志

```bash
# 查看内核日志
dmesg | grep tty

# 查看 serial 相关信息
dmesg | grep serial
```

### 网络抓包（如果 STM32 支持回传）

```bash
# 如果 STM32 会回传数据，可以监控
cat /dev/ttyS8 | hexdump -C
```

## 常用的串口测试命令

```bash
# 1. 列出所有串口
ls -la /dev/tty*

# 2. 查看串口属性
stty -F /dev/ttyS8 -a

# 3. 发送测试数据
echo "test" > /dev/ttyS8

# 4. 读取串口数据
cat /dev/ttyS8

# 5. 使用 Python 测试串口
python3 -c "import serial; s=serial.Serial('/dev/ttyS8', 115200); print(s.name); s.close()"
```

## 成功的标志

当一切正常工作时，你应该看到：

### 姿态检测输出：
```
🎯 目标1 | rel=(0.01, -0.72) | conf=0.88
```

### 电机控制输出：
```
✅ 串口已连接: /dev/ttyS8 @ 115200
🎯 目标1 | rel=(0.01, -0.72) | 向右转0.050rad, 向下0.025rad
已发送增量: 电机21=-0.050rad, 电机22=-0.025rad
数据包: AA0B1100000000C9BF00000000C9BF3D
```

### STM32 端（如果支持调试输出）：
```
Received command: 0x11
Yaw: -0.050 rad
Pitch: -0.025 rad
Motor 21 updated
Motor 22 updated
```

## 联系支持

如果以上步骤都无法解决问题，请提供以下信息：

1. 错误信息的完整输出
2. `ls -la /dev/ttyS*` 的结果
3. `dmesg | grep tty` 的结果
4. STM32 固件是否正常运行
5. 硬件连接照片（如果有）
