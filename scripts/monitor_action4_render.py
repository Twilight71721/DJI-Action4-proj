"""Watch an existing render queue and notify the Linux desktop once per incident."""

import argparse
import fcntl
import html
import json
import logging
import os
from pathlib import Path
import subprocess
import time


def process_matches(pid, root):
    try:
        proc = Path('/proc') / str(int(pid))
        args = (proc / 'cmdline').read_bytes().decode().split('\0')
        state = (proc / 'stat').read_text().rsplit(')', 1)[1].split()[0]
        return (state != 'Z' and any(Path(a).name == 'run_action4_reference_5000.py'
                                    for a in args if a)
                and str(root) in args)
    except (OSError, ValueError, TypeError, IndexError):
        return False


def latest_activity(root):
    latest = 0.0
    # Queue status changes only between batches; include native-render outputs.
    for base in (root / 'logs', root / '.staging'):
        for directory, _, files in os.walk(base):
            for name in files:
                try:
                    latest = max(latest, (Path(directory) / name).stat().st_mtime)
                except FileNotFoundError:
                    pass
    return max(latest, (root / 'status.json').stat().st_mtime)


def diagnose(status, alive, idle_seconds, stall_seconds):
    state = status.get('state', '')
    if state == 'complete':
        return None
    if state == 'failed':
        return '渲染队列报告失败：' + str(status.get('error', '请查看渲染日志'))[:400]
    if state.startswith('paused') or state == 'pilot_complete_awaiting_visual_review':
        return '渲染队列已暂停，需要人工处理。'
    if not alive:
        return '渲染队列进程已退出，任务尚未完成。'
    limit = max(stall_seconds, 3600) if state == 'final_validation' else stall_seconds
    if idle_seconds >= limit:
        return f'连续 {int(idle_seconds // 60)} 分钟没有输出更新，疑似卡住，请检查。'
    return None


def notify(title, body, urgency='critical'):
    result = subprocess.run(
        ['notify-send', '--app-name=Action4 render monitor', f'--urgency={urgency}',
         '--expire-time=' + ('0' if urgency == 'critical' else '10000'),
         '--print-id', title, html.escape(body)],
        check=True, capture_output=True, text=True, timeout=15)
    return result.stdout.strip()


def completed_counts(root):
    markers = [json.loads(p.read_text()) for p in (root / 'completed').glob('*.json')]
    return len(markers), sum(int(marker['frames']) for marker in markers)


def progress_due(memory, now, interval):
    return now - memory.get('last_progress_notification', 0) >= interval


def update_incident(memory, reason, recovery_key):
    # A changed error message alone is not a new incident. Require recovery or
    # a new queue run / accepted batch before another successful notification.
    if reason is None or memory.get('recovery_key') != recovery_key:
        memory['notified'] = False
    memory['recovery_key'] = recovery_key
    return reason is not None and not memory.get('notified', False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=60)
    parser.add_argument('--stall-minutes', type=float, default=20)
    parser.add_argument('--progress-minutes', type=float, default=5)
    parser.add_argument('--test-notification', action='store_true')
    args = parser.parse_args()
    if args.interval <= 0 or args.stall_minutes <= 0 or args.progress_minutes <= 0:
        parser.error('interval, stall-minutes and progress-minutes must be positive')
    root = args.dataset_root.resolve(strict=True)
    if args.test_notification:
        print(notify('渲染监控测试', '这是一条测试通知，并非渲染故障。桌面提醒通道已连通。'))
        return
    monitor = root / 'monitor'
    monitor.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(monitor / 'monitor.log')])
    lock = (monitor / 'monitor.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('A monitor is already running for this dataset')
    (monitor / 'monitor.pid').write_text(str(os.getpid()))
    state_path = monitor / 'state.json'
    try:
        memory = json.loads(state_path.read_text())
    except (OSError, ValueError):
        memory = {}
    started = time.time()
    unreadable_since = None
    logging.info('Started: root=%s interval=%ss stall=%smin', root, args.interval, args.stall_minutes)
    while True:
        now = time.time()
        try:
            status = json.loads((root / 'status.json').read_text())
            if not isinstance(status, dict):
                raise ValueError('status must be an object')
            activity = latest_activity(root)
            unreadable_since = None
            alive = process_matches(status.get('pid'), root)
            # Give newly attached monitoring a full timeout before stall alerts.
            reason = diagnose(status, alive, now - max(activity, started), args.stall_minutes * 60)
            key = [status.get('pid'), status.get('completed_frames')]
        except (OSError, ValueError) as exc:
            unreadable_since = unreadable_since or now
            logging.warning('Cannot read progress: %s', exc)
            status = {}
            key = memory.get('recovery_key')
            reason = '连续多次无法读取渲染进度，请检查任务及磁盘。' if now - unreadable_since >= 120 else None
            if reason is None:
                time.sleep(args.interval)
                continue
        if update_incident(memory, reason, key):
            body = (f"已完成 {status.get('completed_frames', '?')}/{status.get('total', 5000)} 张。\n"
                    f'{reason}\n目录：{root}')
            try:
                notification_id = notify('Action4 渲染需要处理', body)
                memory.update(notified=True, notification_id=notification_id)
                logging.warning('Notified: %s', reason)
            except (OSError, subprocess.SubprocessError) as exc:
                logging.error('Notification failed; will retry: %s', exc)
        if status and progress_due(memory, now, args.progress_minutes * 60):
            try:
                groups, frames = completed_counts(root)
                state_text = reason or {
                    'rendering': '正在渲染', 'batch_complete': '本组已完成，继续渲染',
                    'final_validation': '正在最终校验', 'complete': '全部完成',
                }.get(status.get('state'), str(status.get('state')))
                body = f"已完成 {groups} 组，{frames}/{status.get('total', 5000)} 张。\n{state_text}"
                notification_id = notify('Action4 渲染进度', body, urgency='normal')
                memory.update(last_progress_notification=now, progress_notification_id=notification_id,
                              completed_groups=groups)
                logging.info('Progress notification: %s', body.replace('\n', ' '))
            except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
                logging.error('Progress notification failed; will retry: %s', exc)
        memory.update(checked_at=now, monitor_pid=os.getpid(),
                      render_state=status.get('state'), completed_frames=status.get('completed_frames'),
                      problem=reason)
        temporary = state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(memory, ensure_ascii=False, indent=2))
        temporary.replace(state_path)
        if status.get('state') == 'complete':
            logging.info('Rendering complete; monitor exits quietly')
            return
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
