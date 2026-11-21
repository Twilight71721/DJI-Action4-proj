# BoxDreamer 设计核对与 Action4 项目改进依据

日期：2026-09-26。范围：阅读官方论文、补充材料和实现，核对我们最新紧裁剪实验；执行两个独立函数的 CPU 诊断。本次没有运行 BoxDreamer 预训练模型，也没有测得它在 Action4 上的准确率。

## 1. 核心结论

我们目前实现的是“单张目标裁剪 → ResNet50-FPN → 八角点热图”的实例级基线，不能称为完整复现 BoxDreamer。BoxDreamer 的关键是：**参考图像及参考角点提供几何提示，预训练 DINOv2 提供图像特征，Transformer 在参考图与待测图之间传递信息，最后用 PnP 求位姿。** 它并不是仅把 ResNet 换成 ViT。

当前最值得借鉴的是两件事：把“保留哪些图像像素”与“允许角点落在哪个坐标范围”分开；用有明确角点身份的参考视图降低新视角下的歧义。前者可以在保留现有 YOLO 的条件下改进，后者会改变模型输入条件，需要单独作为实验报告。

不能从论文推导出“我们的失败已经确定主要由背景造成”或“换 ViT 一定成功”。我们的人工八角外接框诊断仍有很大误差，说明裁剪之外的问题依然存在。

## 2. 来源与版本

- [官方项目页](https://zju3dv.github.io/boxdreamer/)
- [论文 v2](https://arxiv.org/html/2504.07955v2)
- [补充材料](https://zju3dv.github.io/boxdreamer/files/supp.pdf)
- [官方仓库](https://github.com/zju3dv/BoxDreamer)
- 本地阅读版本：`e7200e06bbb5e6aa11b8d7015b4ea7783bdd6930`，位于 `third_party/BoxDreamer`。
- 远端 HEAD：`106f3c16c746770cbad80720ce4a557eb31f30f5`。通过官方 GitHub compare API 核对，远端比本地多两个提交，只改变 `LICENSE` 和 `README.md`；本次研究的网络、数据处理和解码代码一致。没有更新本地第三方目录。

下文严格区分论文描述、公开默认配置、代码行为和我们自己的建议。默认配置不等同于论文所有实验实际使用的完整配置。

## 3. 它实际在做什么

### 3.1 推理有参考图，不是单图孤立预测

流程可以理解为：

```text
同一物体的若干参考 RGB 图
  → 检测/分割目标 → 稀疏重建
  → 参考位姿、内参估计、物体三维包围盒
  → 在每张参考图投影八点，生成八通道参考热图

参考 RGB + 参考热图 ─┐
                     ├→ DINOv2 特征与角点提示融合 → Transformer
待测 RGB + 未知标记 ─┘
  → 待测图八通道热图 → 二维八点 → PnP → 位姿 → 重投影三维框
```

参考图告诉网络的不只是“这是相机”，还包括“这个面、这个位置对应三维框的哪一个编号”。待测图的真实八点没有作为网络角点提示输入：`BETR.forward` 会用可学习的未知标记替换待测图的角点特征。

公开默认设置为 5 张参考图 + 1 张待测图，`length=6`。同一个参考集合可以用于多张待测图。参考条件推理不等同于在真实帧上微调权重，但同样使用了真实参考信息，汇报时应明确。

源码：[参考特征融合及 query 替换](https://github.com/zju3dv/BoxDreamer/blob/e7200e06bbb5e6aa11b8d7015b4ea7783bdd6930/src/models/modules/backbone/betr.py#L249)、[默认演示配置](https://github.com/zju3dv/BoxDreamer/blob/e7200e06bbb5e6aa11b8d7015b4ea7783bdd6930/src/demo/configs/data.yaml)。

### 3.2 骨干和输出

| 环节 | 官方默认实现 | 我们最新紧裁剪版本 |
|---|---|---|
| 输入条件 | 参考 RGB、参考八点热图、待测 RGB | 单张 RGB 裁剪 |
| 骨干 | 冻结的 `dinov2_vitb14_reg` | ImageNet 预训练 ResNet50，继续训练 |
| 跨图融合 | 12 层 Transformer，768 维，8 个注意力头 | FPN 融合同一张图的多尺度特征 |
| 输入尺寸 | 224×224 | 256×256 |
| 热图尺寸 | 8×224×224 | 8×64×64 |
| 角点身份提示 | 每次推理提供参考角点热图 | 依赖训练中学到的固定编号关系 |
| 输出后处理 | 八点 → PnP → 三维框重投影 | 最新评估直接连接预测八点 |

224 输入按 14×14 patch 切成 16×16 个图像 token；线性层预测每个 patch 内的 14×14×8 个热图值，再拼回全分辨率热图。因此“全分辨率热图”不代表原始图像细节被额外恢复。它的输入比我们的 256 更小，也说明不能只用输入分辨率解释我们的大幅失效。

源码：[网络配置](https://github.com/zju3dv/BoxDreamer/blob/e7200e06bbb5e6aa11b8d7015b4ea7783bdd6930/configs/model/transformer.yaml)、[冻结的 DINOv2 编码器](https://github.com/zju3dv/BoxDreamer/blob/e7200e06bbb5e6aa11b8d7015b4ea7783bdd6930/src/models/modules/encoder/dinov2.py)、[输出重组](https://github.com/zju3dv/BoxDreamer/blob/e7200e06bbb5e6aa11b8d7015b4ea7783bdd6930/src/models/modules/backbone/betr.py#L403)。

我们的配置依据：服务器回传 config.json（`outputs/server_return_tight_cuboid_20260926/tight_cuboid_fixed_yolo_30ep/training/config.json`）、[ResNet50-FPN 实现](../bop_bb8/net/resnet_fpn.py)。

## 4. 对当前裁剪问题最有价值的发现

### 4.1 机身范围与八角投影范围分开

三维包围盒的八个顶点是几何构造，不一定在机身表面，也不一定落在可见轮廓内。因而 YOLO 机身框正确，并不能保证它包住八个虚拟角点。

BoxDreamer 的普通训练/基准数据处理路径在有三维盒和位姿时：

1. 保存原始二维目标范围为 `org_box`。
2. 投影三维盒的八点，以它们的 min/max 得到新的裁剪范围。
3. 把裁剪范围补成正方形，`square_bbox` 默认边长再乘 1.1。
4. 在 `use_bbox=True`、`mask_bg=True` 且 `org_box` 存在时，用 `org_box` 决定保留哪些 RGB 像素，外面填黑。
5. 按新的裁剪范围缩放，并同步调整内参，再投影/生成训练角点热图。

这里的背景屏蔽是**二维矩形屏蔽**，不是默认精确沿相机轮廓分割。矩形内部仍可能有背景。`use_mask=False` 也不等于 `mask_bg=False`，两者是不同开关。

它允许角点落在纯色区域内，同时不需要保留该区域原来的背景纹理。老师提出的“尽量少看背景”和我们的“八点不能超出输出范围”可以同时满足。

源码：[两种范围的准备和调用](https://github.com/zju3dv/BoxDreamer/blob/e7200e06bbb5e6aa11b8d7015b4ea7783bdd6930/src/datasets/base.py#L455)、[矩形之外填黑](https://github.com/zju3dv/BoxDreamer/blob/e7200e06bbb5e6aa11b8d7015b4ea7783bdd6930/src/datasets/utils/preprocess.py#L218)。

### 4.2 不能把基准路径当成真实部署路径

`base.py` 明确排除了 `split == "demo"` 的待测图：真实推理不知道待测位姿，无法预先获得真实八角外接框；该分支 `org_box=None`。默认演示的 `use_mask=False`，因此不能说每张演示待测图都执行了上述矩形填黑。

此外，`CustomDataset.read_boxes` 从 mask 取二维范围后还按 `padding=0.3` 扩大，再进入方形裁剪。该实现先更新左上再计算右下，实际扩大也不是严格对称的“左右各加原宽度 30%”。这证明官方演示并不是始终零余量紧贴目标。

所以，不能宣称“BoxDreamer 已经提供了训练和部署完全一致的紧裁剪方案”。可借鉴的是把内容范围与坐标范围分开的思想，而非原样复制所有裁剪常数。

源码：[demo 例外条件](https://github.com/zju3dv/BoxDreamer/blob/e7200e06bbb5e6aa11b8d7015b4ea7783bdd6930/src/datasets/base.py#L457)、[演示框扩展](https://github.com/zju3dv/BoxDreamer/blob/e7200e06bbb5e6aa11b8d7015b4ea7783bdd6930/src/datasets/custom.py#L195)。

### 4.3 我们当前还差在哪里

我们的 `content_box` 在训练时就是合成八角 min/max，在真实预测时是 YOLO 机身框。它们同时决定真实像素保留范围和坐标画布大小。灰色补边已经减少背景，但尚未做到上述两种范围的独立控制，而且训练/预测的范围定义不同。

上一轮已核对的结果：9 个真实参考实例中，紧 YOLO 裁剪有 9/72 个参考点落出输入；换成人工八点外接框后，同一评估坐标中的平均误差只从 75.24 降至 69.54 px，仍明显失败。这里 72 点包含 28 个可见参考点和 44 个隐藏/虚拟点的人工推定，不是精密真值，也不是独立视频盲测。

因此，框定义不一致是已发现的实际问题，但不是已证明的唯一主因。依据：上一轮本地核对（`outputs/server_return_tight_cuboid_20260926/ANALYSIS.md:25`）、[现有裁剪实现](../bop_bb8/cropping.py)、[训练裁剪调用](../bop_bb8/dataset.py#L257)。

## 5. 热图、损失、解码：应该成套比较

### 5.1 热图不是同一种监督

官方 `make_bbox_features` 使用距离指数衰减：

```text
d_i = 第 i 个角点到八点二维平均中心的距离
s_i = (d_i / 10)^2
H_i(x, y) = exp(-distance((x, y), corner_i) / s_i)
```

它根据角点与中心的距离调整衰减范围，随后归一化并映射到 [-1,1]；网络输出也经过 `2*sigmoid(logits)-1`。我们使用固定 sigma=2 的高斯热图、[0,1] 目标、64×64 输出。即使都叫“heatmap”，监督的形状、尺度、数值范围并不相同。

公开默认损失是热图 SmoothL1；坐标 SmoothL1 配置被注释。论文描述了热图与坐标的组合监督，但不能据此说公开默认配置已启用两者。我们当前则是 focal + 10 倍可微坐标损失。

因此，不应只把我们的 focal 改成 SmoothL1 就称为复现。需要把目标形状、归一化、输出激活、输出分辨率和损失一起核对，再做受控对照。

源码：[热图生成](https://github.com/zju3dv/BoxDreamer/blob/e7200e06bbb5e6aa11b8d7015b4ea7783bdd6930/src/datasets/utils/base/bbox_utils.py#L262)、[默认损失](https://github.com/zju3dv/BoxDreamer/blob/e7200e06bbb5e6aa11b8d7015b4ea7783bdd6930/configs/model/loss/default.yaml)、[我们的损失](../bop_bb8/net/loss.py)。论文相关描述见[第 3.3 节](https://arxiv.org/html/2504.07955v2#S3.SS3)。

### 5.2 官方解码没有保证局部峰或八点几何合法

`recover_bb8_corners` 在每个通道全局取最高 20 个像素，直接平均它们的坐标，没有概率加权，也没有强制这些位置属于同一个局部峰。如果出现相隔很远的两个峰，均值可能落在中间的低响应区域。

本次调用未修改的官方函数，构造两个各 10 个像素的高响应簇：输出落在 `(13.0, 11.5)`，该位置附近的原始响应为 0。这验证了解码的多峰局限，但不能据此断言我们的每个真实失败都由解码引起。

同一检查还确认输出坐标 `requires_grad=False`：官方热图解码只使用 top-k 的离散索引，坐标路径不可微。因此，直接取消默认配置中坐标损失的注释，也不能自动得到有效的坐标训练梯度。这个发现限定于所审版本的热图分支，不推断作者全部训练历史。

源码：[top-20 解码](https://github.com/zju3dv/BoxDreamer/blob/e7200e06bbb5e6aa11b8d7015b4ea7783bdd6930/src/models/utils/box_utils.py#L73)、[训练中坐标的计算](https://github.com/zju3dv/BoxDreamer/blob/e7200e06bbb5e6aa11b8d7015b4ea7783bdd6930/src/models/utils/prediction_utils.py#L124)。

### 5.3 官方视频中的规则线框经过 PnP

官方演示读取预测位姿，再将同一个刚性三维盒投影回画面。它显示的是几何后处理后的框，不能直接与我们未经约束的二维八点连线比较“像不像盒子”。八点都预测错时，PnP 也可能给出错误但外形规则的框。

还需注意，默认单组八点的 PnP 实现调用 RANSAC 后紧接着设 `success=False`，强制走普通迭代 PnP；不能仅凭函数里出现 `solvePnPRansac` 就说该路径有效利用了 RANSAC 剔除错点。另一套 dense 分支不同。

我们以前已经试过刚体拟合与联合解码。BoxDreamer 并不构成“再次加几何约束就能成功”的新证据；前端需要先获得可信的角点信息。

源码：[默认 PnP 路径](https://github.com/zju3dv/BoxDreamer/blob/e7200e06bbb5e6aa11b8d7015b4ea7783bdd6930/src/models/utils/box_utils.py#L156)、[演示重投影绘制](https://github.com/zju3dv/BoxDreamer/blob/e7200e06bbb5e6aa11b8d7015b4ea7783bdd6930/src/demo/gradio_demo.py#L808)。

## 6. 它怎样应对真实域、新视角与内参缺失

论文训练集包含 4.5 万以上合成物体、50 个真实物体和 290 万以上图像；表 7 中加入真实训练数据后，LINEMOD 的 ADD(s)-0.1d 从 40.6 提高到 51.3，Proj2D@5px 从 25.4 提高到 43.8。这些是作者数据集上的结果，不能当成我们 Action4 的预计收益。依据：[论文第 3.4 节与表 7](https://arxiv.org/html/2504.07955v2)。

这意味着其迁移能力同时依赖通用图像预训练、跨物体姿态训练、参考信息以及真实训练数据。我们的背景、模糊和连续姿态增强覆盖了其中一部分，但增加单个模型的合成图片数量，不等价于增加物体与真实成像的多样性。

公开代码还有一种需避免误抄的增强：对同一组视图的位姿共同右乘随机旋转，从而改变虚拟三维框与外观的对应关系，而不重新渲染 RGB。参考图的角点提示随之更新，待测图学习跟随这组几何定义。单图固定角点编号模型如果只改标签不改 RGB、也不提供参考提示，会引入输入无法区分的标签变化。我们已有的连续姿态重新渲染是另一件事，应继续保持 RGB 与标签一致。

源码：[共同旋转](https://github.com/zju3dv/BoxDreamer/blob/e7200e06bbb5e6aa11b8d7015b4ea7783bdd6930/src/datasets/utils/aug.py#L298)、[调用位置](https://github.com/zju3dv/BoxDreamer/blob/e7200e06bbb5e6aa11b8d7015b4ea7783bdd6930/src/datasets/base.py#L418)。默认 local 配置里该开关为 False，不能把论文增强描述直接当成默认运行行为。

对没有测量内参的问题，官方演示使用 DUSt3R 从参考图估计内参、位姿和粗几何，并把估计内参传入后续推理；不是完全无需内参。我们可探索这条路线，但估计焦距、反光屏幕、两台相似相机与运动模糊都需要实际验证，不能视为已经完成标定。若采用重建坐标系，还需和我们 GLB 的尺度、朝向及八点编号建立对应，才能比较同一任务的结果。

源码：[估计内参与数据传递](https://github.com/zju3dv/BoxDreamer/blob/e7200e06bbb5e6aa11b8d7015b4ea7783bdd6930/src/demo/gradio_demo.py#L633)、[演示说明](https://github.com/zju3dv/BoxDreamer/blob/e7200e06bbb5e6aa11b8d7015b4ea7783bdd6930/src/demo/gradio_demo.py#L2130)。

## 7. 推荐的实验顺序

以下是建议，不是本次已经执行的训练。保留现有 YOLO；继续遵循用户“不增加人工遮挡”的决定。

| 顺序 | 实验 | 主要回答的问题 | 必须固定/记录的条件 |
|---|---|---|---|
| 1 | 保留机身内容，独立设置纯色坐标画布 | 训练/推理框定义和越界是否在损害八点预测 | 用同一个冻结 YOLO 或一致的机身框定义处理训练与预测；画布规则只由训练/验证数据确定，待测时不能使用真实八点 |
| 2 | 冻结 DINOv2 + 轻量八点头，与 ResNet 对照 | 通用特征迁移是否比当前骨干更合适 | 相同数据、划分、裁剪、输出网格、损失、解码和优化步数；报告为骨干方案对照，不称完整 BoxDreamer |
| 3 | 官方预训练 BoxDreamer + 覆盖主要视角的参考图，小规模推理验证 | 显式参考几何提示能否缓解角点身份/新视角歧义 | 记录参考来源、是否用真实参考、重建质量、内参来源；参考图与测试图分开，避免相邻帧过于相似；不把此成绩混入单图基线 |
| 4 | 按真实响应分布比较热图监督与解码 | 错误是峰没学到、编号混淆，还是多峰平均造成 | 先保存八张热图；区分原始点与 PnP 后结果；避免同时换骨干、损失和数据 |

步骤 1 不是把图无限放大，也不是恢复原来的大背景裁剪：保留相机像素之外的部分可以是纯色。需要先用合成三维标签统计八点相对机身框的覆盖范围，并检查真实参考点越界率。仅在旧权重上改变输入可做诊断，正式对照仍应让训练与推理都遵循新规则。

步骤 3 应优先利用公开预训练模型验证可行性，而非在单张 4090 上重做论文级预训练。论文级训练和本项目的 30 轮小规模训练不是同一计算任务。是否值得搭建该参考分支，要看小批实际推理结果。

评价应同时保留：固定真实帧上的可见参考点误差、推定点误差、PCK、点越界率、编号是否一致、明显失败实例；如有 PnP，还应单列其原始点和重投影。当前少量、反复查看过的真实实例适合诊断，不足以支持新的泛化结论。

## 8. 本次验证与产物

仅新增本报告及独立诊断脚本/结果，未修改现有训练、预测代码或第三方代码。

- 诊断脚本（`outputs/boxdreamer_design_audit_20260926/check_official_primitives.py`）
- 诊断结果与源文件 SHA256（`outputs/boxdreamer_design_audit_20260926/primitive_checks.json`）

运行方式：

```bash
cd /home/twilight/University/njukym/cs/cv/exam-project
.venv-hccepose/bin/python outputs/boxdreamer_design_audit_20260926/check_official_primitives.py
```

检查结果：矩形屏蔽可在不改变 40×40 坐标画布的情况下，只保留 153 个原始 RGB 像素；top-20 解码的双峰样例落入两峰之间，且坐标输出没有梯度。它们验证实现行为，不是模型识别准确率实验。
