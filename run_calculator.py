"""Create an isolated environment, install requirements once, open local app."""
import hashlib
from pathlib import Path
import subprocess
import sys
import venv


def main():
    root = Path(__file__).resolve().parent
    if sys.version_info < (3, 10):
        raise RuntimeError('Python 3.10 or newer required. Recommended: Python 3.12.')
    folder = root / '.venv'
    python = folder / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    if not python.exists():
        print('Creating local Python environment...', flush=True)
        venv.EnvBuilder(with_pip=True).create(folder)
    requirement_file = root / 'requirements.txt'
    digest = hashlib.sha256(requirement_file.read_bytes()).hexdigest()
    stamp = folder / '.calculator-requirements'
    if not stamp.exists() or stamp.read_text() != digest:
        print('Installing dependencies. Internet is required for the first launch...', flush=True)
        subprocess.run([str(python), '-m', 'pip', 'install', '-r', str(requirement_file)], cwd=root, check=True)
        stamp.write_text(digest)
    print('Starting calculator. Open the Local URL below in your browser. Ctrl+C stops the server.', flush=True)
    # Headless mode also disables Streamlit's first-run interactive email prompt.
    subprocess.run([str(python), '-m', 'streamlit', 'run', str(root / 'app.py'), '--server.headless=true'], cwd=root, check=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f'Launch failed: {error}\nCheck Python installation, internet access and folder permissions.', file=sys.stderr)
        sys.exit(1)
