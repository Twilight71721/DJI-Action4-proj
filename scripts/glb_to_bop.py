#!/usr/bin/env python3
"""Convert a GLB model (e.g. Hunyuan3D output) into a BOP-format dataset.

将 GLB 模型转换为 BOP 数据集所需的 PLY（顶点色 + 法线，单位 mm，几何中心在原点），
并生成 models_info.json 与 camera.json。

Pipeline / 流程:
    1. trimesh:        GLB -> 烘焙纹理到顶点色 -> 中间 PLY（缓存在 .cache/glb2bop/）
    2. pymeshlab:      合并 UV 接缝重复顶点 -> 抽稀 -> 缩放至真实尺寸(mm) -> 居中
                       -> 计算法线 -> 输出非二进制 PLY
    3. 生成 models/models_info.json 与 camera.json

Usage:
    .venv-hunyuan3d/bin/python scripts/glb_to_bop.py \
        --input "3D Model/DJI Action 4 model.glb" \
        --output-dir dataset/dji-action4

Requires / 依赖: trimesh, pymeshlab, numpy, scipy, Pillow
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent

DEFAULT_CAMERA = {
    "cx": 325.2611083984375,
    "cy": 242.04899588216654,
    "depth_scale": 0.1,
    "fx": 572.411363389757,
    "fy": 573.5704328585578,
    "height": 480,
    "width": 640,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert GLB to BOP-format PLY dataset")
    parser.add_argument("--input", required=True, type=Path, help="input GLB/GLTF file")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="BOP dataset directory, e.g. dataset/dji-action4")
    parser.add_argument("--obj-id", type=int, default=1, help="BOP object id (default: 1)")
    parser.add_argument("--max-dim-mm", type=float, default=70.5,
                        help="real-world size of the longest bounding-box edge in mm (default: 70.5)")
    parser.add_argument("--target-faces", type=int, default=30000,
                        help="target face count after decimation (default: 30000)")
    parser.add_argument("--no-decimate", action="store_true", help="keep the original resolution")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_DIR / ".cache" / "glb2bop",
                        help="directory for intermediate files")
    return parser.parse_args()


def export_colored_ply(glb_path: Path, cache_dir: Path) -> Path:
    """Load GLB, bake its texture into per-vertex colors, export an intermediate PLY."""
    import trimesh

    scene = trimesh.load(glb_path)
    mesh = scene.to_geometry() if isinstance(scene, trimesh.Scene) else scene
    visual = mesh.visual
    if type(visual).__name__ == "TextureVisuals":
        mesh.visual = visual.to_color()
    elif type(visual).__name__ == "ColorVisuals" and visual.vertex_colors is not None:
        pass
    else:
        raise RuntimeError(f"GLB has neither texture nor vertex colors: {glb_path}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_ply = cache_dir / f"{glb_path.stem}.colored.ply"
    mesh.export(tmp_ply, file_type="ply", encoding="binary")
    n_colors = len(mesh.visual.vertex_colors)
    print(f"[1/3] GLB -> colored PLY: {len(mesh.vertices)} vertices, "
          f"{len(mesh.faces)} faces, {n_colors} colors -> {tmp_ply}")
    return tmp_ply


def convert_to_bop_ply(colored_ply: Path, out_ply: Path,
                       max_dim_mm: float, target_faces: int, decimate: bool) -> None:
    """Merge seam vertices, decimate, scale to mm, center at origin and save the BOP PLY."""
    import pymeshlab as ml

    ms = ml.MeshSet()
    ms.load_new_mesh(str(colored_ply))

    ms.meshing_merge_close_vertices(threshold=ml.PercentageValue(1e-4))
    merged_vn = ms.current_mesh().vertex_number()

    ext_before = np.ptp(ms.current_mesh().vertex_matrix(), axis=0)
    if decimate and ms.current_mesh().face_number() > target_faces:
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=target_faces,
            optimalplacement=False,
            planarquadric=False,
            autoclean=True,
        )
    ext_after = np.ptp(ms.current_mesh().vertex_matrix(), axis=0)
    if not np.allclose(ext_before, ext_after, rtol=1e-2, atol=1e-6):
        raise RuntimeError(f"Decimation changed the bounding box: {ext_before} -> {ext_after}")

    scale = max_dim_mm / float(ext_after.max())
    ms.compute_matrix_from_scaling_or_normalization(
        axisx=scale, axisy=scale, axisz=scale, uniformflag=True, scalecenter="origin", freeze=True)
    verts = ms.current_mesh().vertex_matrix()
    center = (verts.min(axis=0) + verts.max(axis=0)) / 2.0
    ms.compute_matrix_from_translation_rotation_scale(
        translationx=float(-center[0]), translationy=float(-center[1]), translationz=float(-center[2]))
    ms.compute_normal_per_vertex()

    out_ply.parent.mkdir(parents=True, exist_ok=True)
    ms.save_current_mesh(str(out_ply), binary=False, save_vertex_coord=True,
                         save_vertex_normal=True, save_vertex_color=True)
    print(f"[2/3] merge vertices: {merged_vn} -> PLY {out_ply.name}: "
          f"{ms.current_mesh().vertex_number()} vertices, {ms.current_mesh().face_number()} faces, "
          f"scale {scale:.4f}")


def compute_models_info(ply_path: Path) -> dict:
    """Compute BOP models_info fields (mm) including the exact mesh diameter."""
    import trimesh

    mesh = trimesh.load(ply_path)
    verts = np.asarray(mesh.vertices)
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    size = hi - lo
    hull = trimesh.convex.convex_hull(verts)
    hv = np.asarray(hull.vertices)
    diameter = float(np.linalg.norm(hv[:, None, :] - hv[None, :, :], axis=-1).max())
    return {
        "diameter": diameter,
        "min_x": float(lo[0]), "min_y": float(lo[1]), "min_z": float(lo[2]),
        "max_x": float(hi[0]), "max_y": float(hi[1]), "max_z": float(hi[2]),
        "size_x": float(size[0]), "size_y": float(size[1]), "size_z": float(size[2]),
    }


def main() -> None:
    args = parse_args()
    glb_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    if not glb_path.exists():
        raise FileNotFoundError(glb_path)

    warnings.filterwarnings("ignore")
    out_ply = output_dir / "models" / f"obj_{args.obj_id:06d}.ply"
    colored_ply = export_colored_ply(glb_path, args.cache_dir.resolve())
    convert_to_bop_ply(colored_ply, out_ply, args.max_dim_mm,
                       args.target_faces, not args.no_decimate)

    models_info_path = output_dir / "models" / "models_info.json"
    models_info = {str(args.obj_id): compute_models_info(out_ply)}
    with open(models_info_path, "w") as f:
        json.dump(models_info, f, indent=2, sort_keys=True)
        f.write("\n")

    camera_path = output_dir / "camera.json"
    camera_existed = camera_path.exists()
    if not camera_existed:
        with open(camera_path, "w") as f:
            json.dump(DEFAULT_CAMERA, f, indent=2, sort_keys=True)
            f.write("\n")

    import trimesh
    check = trimesh.load(out_ply)
    dims = np.asarray(check.extents)
    center = (np.asarray(check.bounds[0]) + np.asarray(check.bounds[1])) / 2.0
    print(f"[3/3] dataset ready: {output_dir}")
    print(f"      {out_ply.relative_to(output_dir)}: {len(check.vertices)} vertices, "
          f"{len(check.faces)} faces, dims {np.round(dims, 2)} mm, center {np.round(center, 4)}")
    print(f"      diameter {models_info[str(args.obj_id)]['diameter']:.2f} mm, "
          f"camera.json {'kept' if camera_existed else 'written'}")


if __name__ == "__main__":
    main()
