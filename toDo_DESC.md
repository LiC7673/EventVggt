你现在需要为当前 EventVggt 项目实现 DSEC real-world evaluation pipeline。
数据路径：/data1/lzh/dataset/DESC/DSEC_EV_VGGT
目标：
实现一个可以一键执行的 DSEC 测试脚本，用于评估已经训练好的 Event-VGGT checkpoint。

不要重新设计模型。
不要修改训练逻辑。
优先复用当前项目已有：
- MyEventDataset
- evaluation pipeline
- depth/normal metric
- visualization 工具
- checkpoint loading


==============================
当前 DSEC 数据结构
==============================

数据根目录：

DSEC_EV_VGGT/

当前 split:

test/
    zurich_city_00_a
    zurich_city_01_d

train/
    thun_00_a
    zurich_city_02_a
    zurich_city_04_e
    zurich_city_09_b
    zurich_city_09_e


注意：
这里的 train/test 名字只是下载时目录，不代表训练划分。

本次真实测试使用：

test sequences:

    thun_00_a
    zurich_city_04_e
    zurich_city_09_e


如果目录位置不同，通过参数传入。


==============================
目标功能
==============================

新增：

scripts/eval_dsec.sh


要求：

执行：

bash scripts/eval_dsec.sh


即可完成：

1. 加载 checkpoint
2. 构建 DSEC dataset
3. 读取 RGB image
4. 读取 event stream
5. 根据 image timestamp 对齐 event
6. 构造 event voxel
7. 输入 Event-VGGT
8. 输出 depth prediction
9. 与 GT depth 比较
10. 保存 visualization


==============================
新增文件
==============================


新增：

datasets/dsec_dataset.py

负责：

DSEC sequence -> 当前项目 sample 格式。


sample 格式必须兼容已有 MyEventDataset：

{
    "rgb": Tensor,
    "event": Tensor,
    "intrinsics": Tensor,
    "depth": Tensor,
    "pose": optional
}


不要引入新的数据格式。


==============================
DSEC读取要求
==============================


RGB:

读取：

images_rectified_left


event:

读取：

events_left


根据：

image timestamp

建立：

frame-event correspondence。


event voxel:

按照两个连续 RGB frame 时间窗口：

[t_i, t_(i+1)]

累积 event。


event voxel:

保持和当前 synthetic pipeline 相同：

[T,H,W]

或者当前项目使用格式。


==============================
Depth GT
==============================


DSEC提供：

disparity_image


需要转换：

depth = fx * baseline / disparity


读取：

calibration

获得：

fx
baseline


无效 disparity:

设为 invalid mask。


==============================
Evaluation
==============================


新增：

evaluate_dsec.py


参数：

--data-root
--checkpoint
--split
--output


例如：

python evaluate_dsec.py \
    --data-root /data/DSEC \
    --checkpoint ckpt/model.pt \
    --output results/dsec


==============================
测试sequence
==============================


默认：

[
"thun_00_a",
"zurich_city_04_e",
"zurich_city_09_e"
]


逐sequence测试。


输出：

results/dsec/

├── metrics.json

├── thun_00_a/
│   ├── depth_pred/
│   ├── depth_gt/
│   └── vis/

├── zurich_city_04_e/

└── zurich_city_09_e/


metrics.json:

保存：

average:

AbsRel
RMSE
delta1
delta2
delta3


每个sequence单独保存。


==============================
Visualization
==============================


每个sequence保存固定数量样例。


每个样例包含：

1. RGB input
2. event projection
3. predicted depth
4. GT depth
5. error map


如果模型存在：

prediction["contribution"]

额外保存：

6. contribution map
7. selected event projection


用于论文展示。


==============================
要求
==============================


1. 先检查当前代码结构。

2. 不要假设不存在的函数。

3. 如果已有evaluation函数，直接调用。

4. 保持和synthetic dataset输出完全一致。

5. 给出：
- 新增文件列表
- 修改文件列表
- 最终运行命令


最终目标：

我希望只执行：

bash scripts/eval_dsec.sh

即可完成DSEC真实数据测试。