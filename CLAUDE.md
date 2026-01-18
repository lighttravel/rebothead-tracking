# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a robot head tracking system (rebothead-tracking) based on:
- **Hardware**: RK3576 development board + Gaoqing 5047_36 motors
- **Vision**: YOLOv8 person detection via RKNN runtime
- **Communication**: Serial (UART) from RK3576 to STM32 motor controller

The system tracks people using a camera and controls motors 21 (yaw) and 22 (pitch) to follow the target.

## Architecture

### Python Components

1. **`person_tracking.py`** - Main tracking application
   - Runs YOLOv8 detection on RK3576 using RKNNLite
   - Tracks persons using IOU/distance matching algorithm
   - Outputs MJPEG video stream (default port 8080)
   - Writes tracking coordinates to JSON file for motor control
   - Implements configurable detection frequency (default 5 fps for motor control)

2. **`test_motor_rk3576_native.py`** - Motor control testing
   - Tests serial communication from RK3576 to STM32
   - Implements both absolute and delta position commands

### Data Flow

```
Camera → YOLOv8 Detection → PersonTracker → JSON Output
                                                    ↓
                                            Motor Control Program
                                                    ↓
                                            Serial Port (UART)
                                                    ↓
                                            STM32 Motor Controller
```

## Serial Protocol

Motor control uses UART communication with the following packet format:

```
[Header][Length][Command][Data][Checksum]
  1B      1B      1B      8B      1B
```

- **Header**: `0xAA`
- **Length**: `0x0A` (command + data + checksum)
- **Command**: `0x10` (absolute angle) or `0x11` (delta angle)
- **Data**: 8 bytes total
  - Motor 21 angle: IEEE 754 float (4 bytes, little-endian)
  - Motor 22 angle: IEEE 754 float (4 bytes, little-endian)
- **Checksum**: Sum of command + data bytes modulo 256

**Angles**: In radians, range ±12.566 rad (±720°)

## Common Commands

### Run Person Tracking
```bash
# Default: camera /dev/video36, 1280x720, 5 fps detection, 30 fps streaming
python3 person_tracking.py

# With options
python3 person_tracking.py --source 36 --width 1280 --height 720 \
    --detect-fps 5 --stream-fps 30 --port 8080 \
    --output-coords /tmp/tracker_coords.json
```

### Test Motor Control
```bash
# Tests motor movement using /dev/ttyS8 (RK3576 J33 port)
python3 test_motor_rk3576_native.py
```

### View Video Stream
- URL: `http://<RK3576_IP>:8080/stream`
- Open in browser or PotPlayer/VLC

## Key Implementation Details

### Coordinate Output Format
The tracker writes coordinates to `/tmp/tracker_coords.json` with:
- `primary_target.position.relative`: x, y values from -1 to 1 (normalized position from center)
- `primary_target.position.center`: pixel coordinates
- `primary_target.motion.velocity`: velocity in pixels/sec

These relative coordinates are designed for direct motor control input.

### Tracking Algorithm
- Uses IOU (Intersection over Union) for primary matching
- Falls back to center-point distance matching (max 100 pixels)
- Maintains tracking history with 5-frame moving average for smoothing
- Targets are removed after 10 consecutive missed detections

### Motor Control Notes
- Motor 21: Head yaw (left-right rotation)
- Motor 22: Head pitch (up-down rotation)
- STM32 functions: `somnia_set_head_motors(yaw, pitch)` and `somnia_set_motor_direct(index, position)`
- Motors automatically use PID config from current mode (RELAX/ACTION/HOLD/SOFTHOLD)

## Hardware Configuration

- **Camera**: `/dev/video36` (USB camera)
- **Serial**: `/dev/ttyS8` (UART2 on RK3576, connected to STM32 UART7)
- **Baudrate**: 115200
- **Connection**:
  - RK3576 UART2_TX → STM32 UART7_RX
  - RK3576 UART2_RX → STM32 UART7_TX
  - GND → GND

## Dependencies

- `rknnlite` - RKNN runtime for RK3576
- `opencv-python` (cv2) - Camera and image processing
- `numpy` - Numerical operations
- `pyserial` - Serial communication

## Important Files

- `person_tracking.py:1-936` - Main tracking system
- `test_motor_rk3576_native.py:1-119` - Motor control test utility
- `电机21_22串口控制说明.md` - Serial protocol documentation (Chinese)
