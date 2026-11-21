"""Render action/lighting coverage diagnostics without overwriting the old 200."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ACTIONS = ("right_sleeve_fold", "button", "left_sleeve_fold", "sleeve_grasp",
           "hem_fold", "press", "turnover", "spread")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-limit", type=int, default=8)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=args.resume)
    (root / "models").mkdir(exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    for name in ("obj_000001.glb", "models_info.json", "corners_3d.json"):
        if not (root / "models" / name).exists():
            shutil.copy2(ROOT / "dataset/action4-v6-domain-pilot-200/models" / name, root / "models" / name)
    for batch, action in enumerate(ACTIONS[:args.batch_limit]):
        if (root / "render_metadata" / f"batch_{batch:03d}" / "metadata.json").exists():
            continue
        command = [sys.executable, str(ROOT / "scripts/render_action4_wrist_hccepose.py"),
                   "--output", str(root), "--model", str(root / "models/obj_000001.glb"),
                   "--batch", str(batch), "--frames", "4", "--scene", "tabletop", "--action", action,
                   "--seed", str(202609230 + 137*batch), "--width", "1218", "--height", "924",
                   "--samples", "48"]
        print("START", batch, action, flush=True)
        with (root / "logs" / f"batch_{batch:03d}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        print("DONE", (batch+1)*4, flush=True)
    (root / "generation_config.json").write_text(json.dumps({
        "purpose": "32-image action/lighting diagnostics; not final training acceptance",
        "actions": ACTIONS, "frames_per_action": 4, "reference": "four supplied screenshots",
        "sampling": "balanced diagnostic coverage, not empirical event frequencies",
        "resolution": [1218, 924], "samples": 48}, indent=2))


if __name__ == "__main__":
    main()
