# 项目目录与存储管理

当前项目位置：`/home/twilight/University/njukym/cs/cv/ZJU3DV/exam-project`。

## 目录用途

| 目录 | 用途 | 清理原则 |
| --- | --- | --- |
| `bop_bb8/`、`scripts/`、`tests/`、`configs/` | 算法、数据处理、训练及测试代码 | 保留，包括未提交修改 |
| `dataset/` | 原始与派生训练数据、真实标注、划分记录 | 保留；多个目录可能共享硬链接或符号链接，不能仅按目录名判断冗余 |
| `outputs/` | 模型权重、实验日志、预测、可视化和视频 | 保留基线及新模型，保留可复核实验结果 |
| `exports/` | 服务器传输包和完整打包目录 | 已验证的重复完整压缩包可按需重新生成；保留增量包及其依赖目录 |
| `.cache/` | 三维生成等模型缓存 | 含约5 GiB预训练权重，不是全部可随意删除的临时文件 |
| `.venv-hccepose/`、`.venv-hunyuan3d/` | 运行环境 | 保留 |
| `third_party/`、`assets/`、`3D Model/` | 渲染依赖、模型、贴图等资产 | 保留 |
| `.git/` | 版本历史及对象 | 保留，未执行历史改写或垃圾回收 |
| `docs/`、`prompts/` | 实验说明、服务器启动步骤 | 保留 |

## 2026-09-28 清理

完整审计、删除清单和空间统计见 `outputs/storage_cleanup_20260928/`。

1. 逐成员比对内容及SHA256后，删除 `exports/resnet3305_4090.tar.gz` 与 `exports/resnet5000_cleanblur_4090.tar.gz`。各自完整打包目录保留；原容器校验和与 `.removed.md` 说明保留。
2. `outputs/delta5000_restore_check_20260923/dataset` 的12,240个文件与 `exports/resnet5000_cleanblur_4090/dataset` 核对一致，改为相对符号链接，保留原访问路径。不要删除该链接所指的完整打包目录。
3. 未改变正式训练数据、模型权重、旧版回退权重、视频、实验报告或项目源码。

## 重新生成旧完整传输包

在当前项目根目录执行需要的命令。压缩过程可能需要数分钟及约6或18 GiB额外空间。新容器的SHA256可能与历史文件不同，应同步重建校验文件。

```bash
tar -czf exports/resnet3305_4090.tar.gz -C exports resnet3305_4090
sha256sum exports/resnet3305_4090.tar.gz > exports/resnet3305_4090.tar.gz.sha256
```

```bash
tar -czf exports/resnet5000_cleanblur_4090.tar.gz -C exports resnet5000_cleanblur_4090
sha256sum exports/resnet5000_cleanblur_4090.tar.gz > exports/resnet5000_cleanblur_4090.tar.gz.sha256
```

## 统计方法

用 `du -s -B1 .` 比較项目清理前后实际分配空间。多个目录存在硬链接时，分别执行du后相加会重复计数；同一次du对子目录分配占用时又受到扫描顺序影响。不要把这些数字当成可删除空间。
