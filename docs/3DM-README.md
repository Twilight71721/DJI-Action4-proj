### 本地运行 Hunyuan3D-2mv

本项目使用独立的 `.venv-hunyuan3d` 环境和官方
`tencent/Hunyuan3D-2mv/hunyuan3d-dit-v2-mv-turbo` 几何模型。为了适配 8 GB
显存，纹理生成默认关闭。

启动本地网页：

```bash
./scripts/run_hunyuan3d_mv.sh
```

浏览器访问 `http://127.0.0.1:8080`，按照 `images/README.md` 中的对应关系上传
四张图片。模型和生成结果分别缓存在 `.cache/huggingface/` 与
`outputs/hunyuan3d/`。

当前部署已在 RTX PRO 2000 Blackwell Laptop GPU（8 GB）上完成真实推理测试。测试
使用四视图、Turbo 5 步和 196 八叉树解析度，生成约 23.4 万个三角面，耗时约 6 秒。
测试产物为 `outputs/hunyuan3d/dji_action4_smoke_test.glb`。

如需从头重建虚拟环境，可运行：

```bash
./scripts/setup_hunyuan3d.sh
```

也可以直接使用预设参数生成：

```bash
HF_HOME="$PWD/.cache/huggingface" \
HY3DGEN_MODELS="$PWD/.cache/hy3dgen" \
    .venv-hunyuan3d/bin/python scripts/generate_action4.py
```
