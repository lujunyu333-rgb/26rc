> 生成时间：2026-07-10T21:10:58+08:00
# MCU ↔ ROS 串口通信协议文档

> **Auto-generated** — 由 `scripts/codegen.py` 根据 `config/protocol.yaml` 生成，请勿手动修改。

---

## 全局参数

| 参数 | 值 |
| :--- | :--- |
| 波特率 | `460800` |
| 帧头字节 1 | `0x5a` |
| 帧头字节 2 | `0xa5` |
| 校验算法 | `CRC8` |
| 强制握手 | `否` |
| 协议哈希（握手用）| `0xC1FA7AFE` |
| 严格心跳模式 | `否` |
| 心跳超时时间 | `0 ms` |
| 可靠传输重试间隔 | `100 ms` |
| 可靠传输最大重试 | `3 次` |

---

## 帧格式

每帧结构如下（小端序）：

| 字节位置 | 字段 | 说明 |
| :------: | :--- | :--- |
| 0 | Header1 | 固定 `0x5a` |
| 1 | Header2 | 固定 `0xa5` |
| 2 | ID | 消息 ID，见下表 |
| 3 | Len | 数据段字节数 |
| 4 … 4+Len-1 | Data | 各字段按结构体内存布局排列 |
| 4+Len | Checksum | CRC8，覆盖 ID + Len + Data，多项式 `0x31` |

---

## 电控 → ROS（电控主动发送）

### `Ack` — ID `0xfd`

- **ROS 话题**：`/task/ack`
- **ROS 消息类型**：`std_msgs/msg/Int32MultiArray`
- **数据段字节数（Len）**：`2`
- **注意事项**：通用 ACK。acked_id 为被确认包的原始 Packet ID，ack_seq 为发送方附加的序列号。

| 字节偏移 | 字段名 | C 类型 | 字节数 |
| :------: | :----- | :----- | :----: |
| 0 | `acked_id` | `uint8_t` | 1 |
| 1 | `ack_seq` | `uint8_t` | 1 |
| **2** | *(CRC8)* | `uint8_t` | 1 |

### `Heartbeat` — ID `0xfe`

- **ROS 话题**：`/task/heartbeat`
- **ROS 消息类型**：`std_msgs/msg/UInt32`
- **数据段字节数（Len）**：`4`
- **注意事项**：心跳包。当前 enable_heartbeat=false，调试阶段不会主动使用。

| 字节偏移 | 字段名 | C 类型 | 字节数 |
| :------: | :----- | :----- | :----: |
| 0 | `count` | `uint32_t` | 4 |
| **4** | *(CRC8)* | `uint8_t` | 1 |

### `Handshake` — ID `0xff`

- **ROS 话题**：`/task/handshake`
- **ROS 消息类型**：`std_msgs/msg/UInt32`
- **数据段字节数（Len）**：`4`
- **注意事项**：握手包。当前 require_handshake=false，调试阶段不会强制握手。

| 字节偏移 | 字段名 | C 类型 | 字节数 |
| :------: | :----- | :----- | :----: |
| 0 | `protocol_hash` | `uint32_t` | 4 |
| **4** | *(CRC8)* | `uint8_t` | 1 |

---

## ROS → 电控（电控被动接收）

### `Ack` — ID `0xfd`

- **ROS 话题**：`/task/ack`
- **ROS 消息类型**：`std_msgs/msg/Int32MultiArray`
- **数据段字节数（Len）**：`2`
- **注意事项**：通用 ACK。acked_id 为被确认包的原始 Packet ID，ack_seq 为发送方附加的序列号。

| 字节偏移 | 字段名 | C 类型 | 字节数 |
| :------: | :----- | :----- | :----: |
| 0 | `acked_id` | `uint8_t` | 1 |
| 1 | `ack_seq` | `uint8_t` | 1 |
| **2** | *(CRC8)* | `uint8_t` | 1 |

### `Heartbeat` — ID `0xfe`

- **ROS 话题**：`/task/heartbeat`
- **ROS 消息类型**：`std_msgs/msg/UInt32`
- **数据段字节数（Len）**：`4`
- **注意事项**：心跳包。当前 enable_heartbeat=false，调试阶段不会主动使用。

| 字节偏移 | 字段名 | C 类型 | 字节数 |
| :------: | :----- | :----- | :----: |
| 0 | `count` | `uint32_t` | 4 |
| **4** | *(CRC8)* | `uint8_t` | 1 |

### `Handshake` — ID `0xff`

- **ROS 话题**：`/task/handshake`
- **ROS 消息类型**：`std_msgs/msg/UInt32`
- **数据段字节数（Len）**：`4`
- **注意事项**：握手包。当前 require_handshake=false，调试阶段不会强制握手。

| 字节偏移 | 字段名 | C 类型 | 字节数 |
| :------: | :----- | :----- | :----: |
| 0 | `protocol_hash` | `uint32_t` | 4 |
| **4** | *(CRC8)* | `uint8_t` | 1 |

### `PoseRef` — ID `0x12`

- **ROS 话题**：`/serial/pose_ref`
- **ROS 消息类型**：`std_msgs/msg/Float32MultiArray`
- **数据段字节数（Len）**：`12`
- **注意事项**：map->base_link 位姿，20Hz 定时发送，用于下位机 IMU 漂移校正

| 字节偏移 | 字段名 | C 类型 | 字节数 |
| :------: | :----- | :----- | :----: |
| 0 | `x_map` | `float` | 4 |
| 4 | `y_map` | `float` | 4 |
| 8 | `yaw_map` | `float` | 4 |
| **12** | *(CRC8)* | `uint8_t` | 1 |

### `CmdVel` — ID `0x01`

- **ROS 话题**：`/cmd_vel`
- **ROS 消息类型**：`geometry_msgs/msg/Twist`
- **数据段字节数（Len）**：`12`
- **注意事项**：麦轮底盘速度控制：linear.x 前后，linear.y 左右，angular.z 自转角速度。

| 字节偏移 | 字段名 | C 类型 | 字节数 |
| :------: | :----- | :----- | :----: |
| 0 | `linear_x` | `float` | 4 |
| 4 | `linear_y` | `float` | 4 |
| 8 | `angular_z` | `float` | 4 |
| **12** | *(CRC8)* | `uint8_t` | 1 |

### `CamCmd` — ID `0x02`

- **ROS 话题**：`/camera/view_cmd`
- **ROS 消息类型**：`std_msgs/msg/UInt8`
- **数据段字节数（Len）**：`1`
- **注意事项**：camera_cmd

| 字节偏移 | 字段名 | C 类型 | 字节数 |
| :------: | :----- | :----- | :----: |
| 0 | `camaction` | `uint8_t` | 1 |
| **1** | *(CRC8)* | `uint8_t` | 1 |

### `StairActionCmd` — ID `0x03`

- **ROS 话题**：`/stair_action_cmd`
- **ROS 消息类型**：`std_msgs/msg/UInt8`
- **数据段字节数（Len）**：`1`
- **注意事项**： 1=40cm, 2=20cm, 3=九宫格执行动作, 4=STAIR_DOWN_END

| 字节偏移 | 字段名 | C 类型 | 字节数 |
| :------: | :----- | :----- | :----: |
| 0 | `action` | `uint8_t` | 1 |
| **1** | *(CRC8)* | `uint8_t` | 1 |

### `CamSig` — ID `0x04`

- **ROS 话题**：`/camera/view_sog`
- **ROS 消息类型**：`std_msgs/msg/UInt8`
- **数据段字节数（Len）**：`1`
- **注意事项**：camera_sig

| 字节偏移 | 字段名 | C 类型 | 字节数 |
| :------: | :----- | :----- | :----: |
| 0 | `camsignal` | `uint8_t` | 1 |
| **1** | *(CRC8)* | `uint8_t` | 1 |

---

*文档由构建系统自动生成，版本以协议哈希为准。*
