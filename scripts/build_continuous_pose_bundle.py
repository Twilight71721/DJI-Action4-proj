"""Stage and verify a portable renderer/training increment; large rendering stays on server."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
from server_background_assets import sha,dump

ROOT=Path(__file__).resolve().parents[1]
NAME='resnet_continuous_pose_background_20260924'


def stage(out):
    out.mkdir(parents=True,exist_ok=False)
    files=[p.relative_to(ROOT) for p in (ROOT/'bop_bb8').rglob('*.py')]
    files += [Path('scripts')/n for n in [
        'build_continuous_pose_bank.py','build_multiview_background_bank.py','render_pose_replacement_pilot.py',
        'blender_real_scene_preview.py','action4_dynamic_scene.py','action4_wrist_assets.py','review_pose_replacement_pilot.py',
        'prepare_real_background_extension.py','prepare_online_pose_backgrounds.py','run_online_pose_training.py',
        'verify_continuous_pose_inputs.py','train_heatmap.py','evaluate_online_pose_holdout.py','evaluate_background_model.py',
        'evaluate_cleanblur5000.py','evaluate_heatmap_checkpoint.py','infer_resnet_yolo_real.py','server_background_assets.py','prepare_camera_masks.py']]
    files += [Path('tests')/n for n in ['test_continuous_pose_data.py','test_online_background.py','test_background_augmentation.py','test_net.py']]
    files += [Path('configs')/n for n in ['continuous_pose_background_20260924.json','online_pose_background_20260924.json','background_data_hashes.json']]
    files += [Path('requirements.txt'),Path('dataset/code/action4_domain_v6.py'),Path('dataset/action4-v6-domain-pilot-200/models/obj_000001.glb')]
    texture=Path('dataset/action4-screen-cable-corrected32-20260921/render_metadata/batch_000')
    files += [texture/n for n in ['front_screen_left.png','front_screen_right.png','screen_202609230.png','screen_202609231.png']]
    files += [Path('third_party/Blender/HCCEPose/LICENSE')]
    for rel in files:
        dest=out/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
    shutil.copytree(ROOT/'third_party/Blender/vendor',out/'third_party/Blender/vendor',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    requirements=(ROOT/'third_party/Blender/requirements-render.txt').read_text().replace('opencv-python-headless','opencv-contrib-python-headless')
    (out/'requirements-continuous-render.txt').write_text(requirements+'\nPillow==11.3.0\n')
    plan_source=ROOT/'outputs/continuous_pose_plan_20260924';bank=out/'pose_assets';bank.mkdir()
    plan=json.loads((plan_source/'plan.json').read_text());plan['glb']='dataset/action4-v6-domain-pilot-200/models/obj_000001.glb'
    dump(bank/'plan.json',plan);shutil.copytree(plan_source/'models',bank/'models')
    for n in ['sampling_audit.json','coverage_audit.json']:shutil.copy2(plan_source/n,bank/n)
    # Orientation-only inputs are retained for reproducibility, without absolute preview paths.
    source=ROOT/'exports/full_video_pose_assets_20260924/manifest.json'
    anchors=[{k:r[k] for k in ['id','R','usage']} for r in json.loads(source.read_text())['samples'] if r['usage']=='foreground_candidate_for_server_compositing']
    dest=out/'exports/full_video_pose_assets_20260924';dest.mkdir(parents=True);dump(dest/'manifest.json',dict(samples=anchors,original_manifest_sha256=sha(source)))
    shutil.copytree(ROOT/'outputs/real_background_extension_20260924',out/'real_backgrounds')
    shutil.copytree(ROOT/'exports/resnet_pose_mix_no_occlusion_20260924/real',out/'real')
    shutil.copy2(ROOT/'docs/CONTINUOUS_POSE_BACKGROUND_SERVER_20260924.md',out/'OPENCODE_SERVER_GUIDE.md')
    shutil.copy2(ROOT/'prompts/opencode_continuous_pose_background_20260924.txt',out/'OPENCODE_PROMPT.txt')
    checks=out/'local_verification';checks.mkdir()
    pilot=ROOT/'outputs/continuous_pose_pilot_20260924'
    for n in ['manifest.json','plan.json','sampling_audit.json']:shutil.copy2(pilot/n,checks/('pilot_'+n))
    shutil.copytree(pilot/'review',checks/'render_preview')
    local=ROOT/'outputs/continuous_pose_local_check_20260924'
    shutil.copytree(local/'review',checks/'network_input_preview')
    smoke=checks/'gpu_smoke';smoke.mkdir()
    for n in ['metrics.csv','config.json','baseline.json']:shutil.copy2(local/'smoke_gpu'/n,smoke/n)
    shutil.copy2(local/'smoke_gpu.log',smoke/'train.log')
    (out/'START_HERE.md').write_text('# 连续姿态＋独立外观灯光＋大量真实随机背景\n\n先读OPENCODE_SERVER_GUIDE.md。包内是完整渲染计划及素材；正式4352个前景尚需服务器生成。local_verification为64张本地样本及短训练检查。本轮未正式训练。\n')
    print('STAGED',out,flush=True)


def pack(out):
    sums={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='SHA256SUMS' and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    (out/'SHA256SUMS').write_text(''.join(f'{v}  {k}\n' for k,v in sums.items()))
    archive=out.with_suffix('.tar.gz')
    if archive.exists():raise FileExistsError(archive)
    with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
        for rel in ['SHA256SUMS',*sums]:tar.add(out/rel,arcname=out.name+'/'+rel,recursive=False)
    with tarfile.open(archive) as tar:
        for rel,digest in sums.items():
            if hashlib.sha256(tar.extractfile(out.name+'/'+rel).read()).hexdigest()!=digest:raise ValueError('Archive checksum mismatch '+rel)
    archive.with_name(archive.name+'.sha256').write_text(sha(archive)+'  '+archive.name+'\n')
    print('ARCHIVE',archive,archive.stat().st_size,'VERIFIED',len(sums),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--action',choices=['stage','pack'],required=True);a=p.parse_args()
    out=ROOT/'exports'/NAME
    if a.action=='stage':stage(out)
    else:pack(out)
