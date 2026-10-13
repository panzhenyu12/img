"""Manage only the development processes launched by this checkout."""
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / '.run'
RUN.mkdir(exist_ok=True)
STATE = RUN / 'processes.json'


def load():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def stop():
    for name, info in load().items():
        pid = info['pid']
        if alive(pid):
            actual = subprocess.run(['ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True).stdout
            if info['marker'] in actual:
                os.killpg(pid, signal.SIGTERM)
                print(f'Stopped {name} (pid {pid})')
    STATE.unlink(missing_ok=True)


def start():
    processes = load()
    definitions = {
        'api': (8080, [str(RUN / 'api')], ROOT / 'services/api', str(RUN / 'api')),
        'web': (5173, ['node', str(ROOT / 'apps/web/node_modules/vite/bin/vite.js'), '--host', '127.0.0.1', '--strictPort'], ROOT / 'apps/web', '/vite/bin/vite.js'),
    }
    for name, (port, cmd, cwd, marker) in definitions.items():
        if name in processes and alive(processes[name]['pid']):
            print(f'{name} already running')
            continue
        with socket.socket() as sock:
            # Restarting a local server can leave accepted sockets in
            # TIME_WAIT even after its listener has gone away.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(('127.0.0.1', port))
            except OSError:
                raise SystemExit(f'Port {port} is occupied; stop that process before make dev.')
        with (RUN / f'{name}.log').open('ab') as log:
            env = dict(os.environ, IMG_RENDER_BINARY=str(ROOT / 'core/image-core/target/release/image-render'))
            child = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log, stderr=log, start_new_session=True)
        processes[name] = {'pid': child.pid, 'marker': marker}
        STATE.write_text(json.dumps(processes))
        time.sleep(0.3)
        if child.poll() is not None:
            raise SystemExit(f'{name} exited; see {RUN / (name + ".log")}')
        print(f'{name}: http://127.0.0.1:{port} (log {RUN / (name + ".log")})')


if __name__ == '__main__':
    action = sys.argv[1] if len(sys.argv) > 1 else 'start'
    if action == 'start':
        start()
    elif action == 'stop':
        stop()
    elif action == 'clean':
        for directory in ['apps/web/dist', 'apps/web/public/wasm', 'core/image-core/target', 'apps/web/node_modules/.tmp']:
            shutil.rmtree(ROOT / directory, ignore_errors=True)
    else:
        raise SystemExit('Use start, stop, or clean')
