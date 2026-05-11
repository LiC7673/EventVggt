# Event Stream Visualization Script

## 概述
`visualize_event_stream.py` 是一个事件流可视化脚本，用于将 `.h5` 格式的事件数据转换为图像。脚本将事件流按时间分成多个 bins，并为每个 bin 生成可视化图像。

## 功能特点

- **时间分箱**: 按时间均匀分割事件流为 N 个 frames，每 M 个 frame 为一个 bin
- **极性编码**:
  - 正极性事件 (polarity = 1) → **红色**
  - 负极性事件 (polarity = 0 或 -1) → **蓝色**
  - 背景 → **黑色**
- **分辨率自动推断**: 从事件坐标自动推断图像分辨率
- **批量处理**: 支持自定义输出目录

## 安装依赖

```bash
pip install numpy h5py pillow matplotlib
```

## 使用方法

### 基本用法
```bash
python visualize_event_stream.py <event_h5_path>
```

### 示例
```bash
# 使用默认参数（120帧，每5个frame为一个bin）
python visualize_event_stream.py "F:\TreeOBJ\reflective_raw\Actaeon_Anodized_Red\esmi_event\event.h5"

# 自定义参数
python visualize_event_stream.py \
    "F:\TreeOBJ\reflective_raw\Actaeon_Anodized_Red\esmi_event\event.h5" \
    --num-frames 120 \
    --bin-size 5 \
    --output-dir ./vis_event
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `event_h5_path` | 必需 | 输入的 event.h5 文件路径 |
| `--num-frames` | 120 | 总帧数（事件流被分割成的帧数） |
| `--bin-size` | 5 | 每个 bin 包含的帧数（num_frames / bin_size = 生成的图像数） |
| `--output-dir` | ./vis_event | 输出目录 |

## 输出

- 所有可视化图像保存在 `--output-dir` 指定的目录中
- 文件命名格式: `event_bin_NNN.png`（NNN 为 bin 索引，从 000 开始）
- 生成的图像数 = num_frames / bin_size
  - 默认参数下：120 / 5 = **24 张图像**

## 示例输出结构

```
./vis_event/
├── event_bin_000.png
├── event_bin_001.png
├── event_bin_002.png
├── ...
└── event_bin_023.png
```

## Git 忽略配置

已在 `.gitignore` 中添加 `vis_event/*`，确保可视化输出不会被提交到版本控制系统。

## event.h5 文件格式

事件文件应包含一个名为 `events` 的 HDF5 数据集，格式为：

| 列 | 数据类型 | 说明 |
|----|---------|------|
| 0 | float64 | 事件时间戳 |
| 1 | int | 事件 x 坐标 |
| 2 | int | 事件 y 坐标 |
| 3 | int/float | 事件极性 (1 为正, 0 或 -1 为负) |

## 常见问题

### Q: 如何调整可视化的时间粒度？
**A:** 使用 `--bin-size` 参数。较小的值会生成更多图像，粒度更细。
```bash
python visualize_event_stream.py event.h5 --num-frames 120 --bin-size 1  # 120张图像
python visualize_event_stream.py event.h5 --num-frames 120 --bin-size 10 # 12张图像
```

### Q: 输出目录已存在会怎样？
**A:** 脚本会直接在已存在的目录中保存文件，若存在同名文件则覆盖。

### Q: 事件信息显示什么？
**A:** 日志显示每个 bin 的事件总数、正极性事件数和负极性事件数。

## 性能提示

- 大规模事件数据（>1亿事件）可能需要较长处理时间
- 使用较大的 `--bin-size` 可以减少输出文件数量，加快处理速度
