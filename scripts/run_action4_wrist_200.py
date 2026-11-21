"""Generate exactly 200 reference-guided HCCEPose images in isolated batches."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [("raised", p) for p in ("open", "grasp", "open", "fist")]
CONFIGS += [("tabletop", p) for p in ("pinch", "grasp", "grasp", "open", "pinch", "fist",
                                     "pinch", "grasp", "pinch", "open", "grasp", "pinch", "grasp", "fist")]
CONFIGS += [("raised", "thumbs_up"), ("raised", "open")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-limit", type=int, default=20)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    source = ROOT / "dataset/action4-v6-domain-pilot-200/models"
    (output / "models").mkdir(exist_ok=True)
    for filename in ("obj_000001.glb", "models_info.json", "corners_3d.json"):
        destination = output / "models" / filename
        if not destination.exists():
            shutil.copy2(source / filename, destination)
    (output / "logs").mkdir(exist_ok=True)
    start = time.monotonic()
    for batch, (scene, pose) in enumerate(CONFIGS[:args.batch_limit]):
        done = output / "render_metadata" / f"batch_{batch:03d}" / "metadata.json"
        if done.exists():
            print("Already completed", batch, flush=True)
            continue
        command = [sys.executable, str(ROOT / "scripts/render_action4_wrist_hccepose.py"),
                   "--output", str(output), "--model", str(output / "models/obj_000001.glb"),
                   "--batch", str(batch), "--frames", "10", "--seed", str(20260922 + batch * 137),
                   "--scene", scene, "--pose", pose, "--width", "1624", "--height", "1232", "--samples", "64"]
        if batch in (0, 4, 18):
            command.append("--save-blend")
        print(f"START batch={batch} scene={scene} pose={pose}", flush=True)
        with (output / "logs" / f"batch_{batch:03d}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"DONE images={(batch + 1) * 10}/200 elapsed={time.monotonic() - start:.1f}s", flush=True)
    (output / "generation_config.json").write_text(json.dumps({
        "requested_frames": 200, "resolution": [1624, 1232], "samples": 64,
        "configs": CONFIGS, "batch_size": 10, "seed_base": 20260922,
        "seed_step": 137, "elapsed_seconds_this_run": time.monotonic() - start}, indent=2))


if __name__ == "__main__":
    main()
