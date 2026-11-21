#!/usr/bin/env python3
from pathlib import Path
import os
import time

import torch
from PIL import Image

from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline


PROJECT_DIR = Path(__file__).resolve().parent.parent
IMAGE_DIR = PROJECT_DIR / "images"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "hunyuan3d"


def main() -> None:
    os.environ.setdefault(
        "HY3DGEN_MODELS", str(PROJECT_DIR / ".cache" / "hy3dgen")
    )
    images = {
        view: Image.open(IMAGE_DIR / f"{view}.png").convert("RGBA")
        for view in ("front", "back", "left", "right")
    }

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2mv",
        subfolder="hunyuan3d-dit-v2-mv-turbo",
        use_safetensors=True,
        device="cuda",
    )
    pipeline.enable_flashvdm()

    start_time = time.time()
    mesh = pipeline(
        image=images,
        num_inference_steps=5,
        guidance_scale=5.0,
        octree_resolution=256,
        num_chunks=8000,
        generator=torch.manual_seed(1234),
        output_type="trimesh",
    )[0]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "dji_action4_mv_turbo.glb"
    mesh.export(output_path)
    print(f"Saved: {output_path}")
    print(f"Elapsed: {time.time() - start_time:.1f} seconds")


if __name__ == "__main__":
    main()
