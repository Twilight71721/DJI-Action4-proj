# Action4 5000-frame background rendering

User approved this run on 2026-09-21 after reviewing the generation plan.
Output: `dataset/action4-reference-5000-20260921` (new directory; previous 200 preserved).

## Configuration

- Exact approved GLB, front/rear LCD assets, rear tag and top label.
- HCCEPose vendored BlenderProc, Cycles OPTIX, 1624x1232, 48 samples.
- 3500 reference-guided, 1000 nearby, 500 difficult candidate frames.
- Four lighting families, 1250 frames each, with within-family variation.
- First 100 frames: five per existing reference family, across 20 families.
- Subsequent jobs: at most 20 frames per process, to bound memory and failure scope.
- Eight additional real reference frames inspected: 400, 1075, 1325, 1550,
  1750, 1950, 2125, 2600. Nearby variants interpolate only between same-action
  anchors; their metadata explicitly marks the geometry as estimated.

## Validation and physical constraints

The existing own-arm/cuff, mount/body-bottom, camera/arm, cable/both-arms,
lens-to-work-area and tabletop handedness gates remain mandatory.
`action4_extended_checks.py` adds cross-arm assembly intersection and containment
checks, plus cable tube clearance against wrist fixtures and device envelopes.
Containment probes complement BVH surface overlap, which alone can miss embedded
objects or coplanar intersections. Three regression tests cover that distinction
and a cable passing through a cuff. Existing nine rendering geometry tests pass.

Each frame has up to 25 pose proposals before the batch stops. Each accepted
batch must also pass mask/depth/BOP independent-corner projection checks and
attachment report checks. Original native JSON is backed up before reconciling
pixel counts with actual delivered masks and correcting native COCO PNG paths.
Only accepted batches enter `train_pbr` and `completed/*.json`.

The initial five frames rendered before the containment regression correction
were moved into `superseded_checks/`; they do not count toward the 5000.
Source snapshots and hashes are retained per revision. Source changes stop the
queue, rather than silently mixing unrecorded renderer revisions.

## Split policy

4000 train / 500 validation / 500 test, provided as scene lists under `splits`.
To avoid neighbouring-reference leakage, the full button-handling segment
(reference frames 300–2200, including its flat-press reference) is validation;
the final raised-hand segment (2800–2849) is test. Training uses the initial
segment and garment-folding segment. This is deliberately a **non-IID action
segment holdout**, not a balanced random split. All interpolated descendants
inherit their parent segment's split. A separately recorded real video is still
needed for an unbiased real-world evaluation; this video guided asset design.

## Operation

`scripts/run_action4_reference_5000.py --output <root> --run` resumes completed
batches, preserving failed attempts. A nonblocking file lock prevents duplicate
queues. `status.json` records accepted frame count, current job and failures.
No monitoring automation is installed. The worker itself validates each batch,
writes a gallery and stops on error. Bulk jobs require the first 100 validated
frames and a recorded representative-image review in `VISUAL_APPROVED.json`.

The current scene still uses kinematic cloth, approximate hand contact and one
simulated lens image per device per short batch. The checks are not a full
cloth-contact solver and do not prove all possible physical interactions.

## Estimated duration

Previous 200: 10.95 seconds/frame averaged across ten-frame jobs, about 15.2h
when linearly extrapolated. Expanded five-frame checks measured roughly 86–90s
per batch including process/model initialization and annotation validation.
Conservative small-batch extrapolation is about 24h; larger later batches amortize
startup costs. Initial planning range: 16–24h of uninterrupted machine operation,
excluding any failures that require intervention. Progress estimates in status
are observations, not a completion guarantee.
