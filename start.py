#!/usr/bin/env python3
"""
Insolare Safety System - Single Command Startup Script (Cross-platform)
Starts Frontend, Backend, and Flask Video Server with one command
"""

import os
import sys
import subprocess
import time
import signal
import socket
import shutil
from pathlib import Path

# Colors for terminal output
class Colors:
    GREEN = '\033[0;32m'
    BLUE = '\033[0;34m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    NC = '\033[0m'  # No Color

SCRIPT_DIR = Path(__file__).parent.absolute()
os.chdir(SCRIPT_DIR)

processes = []
FLASK_PORT = None

def cleanup(signum=None, frame=None):
    """Cleanup function to stop all processes"""
    print(f"\n{Colors.YELLOW}Shutting down all services...{Colors.NC}")
    for proc in processes:
        try:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    sys.exit(0 if signum is None else 130)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

def print_header():
    print(f"{Colors.BLUE}{'='*50}{Colors.NC}")
    print(f"{Colors.BLUE}  Insolare Safety System{Colors.NC}")
    print(f"{Colors.BLUE}  Starting all services...{Colors.NC}")
    print(f"{Colors.BLUE}{'='*50}{Colors.NC}\n")

def port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(('127.0.0.1', port)) == 0

def pick_flask_port():
    env_port = os.getenv('FLASK_PORT')
    if env_port:
        return int(env_port)
    for port in (5000, 5001, 5050):
        if not port_in_use(port):
            return port
    return 5001

def wait_for_service(name, port, proc, log_file, timeout=30):
    for _ in range(timeout):
        if proc.poll() is not None:
            print(f"{Colors.RED}✗ {name} failed (process exited). Last log lines:{Colors.NC}")
            if log_file.exists():
                print(log_file.read_text(errors='replace').splitlines()[-10:])
            return False
        if port_in_use(port):
            print(f"{Colors.GREEN}✓ {name} listening on http://localhost:{port} (PID: {proc.pid}){Colors.NC}")
            return True
        time.sleep(1)

    print(f"{Colors.RED}✗ {name} timed out waiting for port {port}. See {log_file}{Colors.NC}")
    if log_file.exists():
        print(log_file.read_text(errors='replace').splitlines()[-10:])
    return False

def check_dependencies():
    (SCRIPT_DIR / "logs").mkdir(exist_ok=True)
    (SCRIPT_DIR / "backend" / "uploads").mkdir(parents=True, exist_ok=True)

    if not (SCRIPT_DIR / "backend" / ".env").exists():
        print(f"{Colors.YELLOW}Warning: backend/.env file not found!{Colors.NC}")
        print(f"{Colors.YELLOW}Please create backend/.env with required environment variables.{Colors.NC}\n")

    global FLASK_PORT
    FLASK_PORT = pick_flask_port()
    os.environ['FLASK_PORT'] = str(FLASK_PORT)

    if port_in_use(5000) and FLASK_PORT != 5000:
        print(f"{Colors.YELLOW}Note: Port 5000 is in use (often macOS AirPlay Receiver).{Colors.NC}")
        print(f"{Colors.YELLOW}Flask will use port {FLASK_PORT} instead.{Colors.NC}")
        frontend_env = SCRIPT_DIR / "frontend" / ".env"
        expected = f"VITE_FLASK_URL=http://localhost:{FLASK_PORT}"
        if not frontend_env.exists() or expected not in frontend_env.read_text():
            print(f"{Colors.YELLOW}Add to frontend/.env: {expected}{Colors.NC}\n")

def start_backend():
    print(f"{Colors.GREEN}[1/3] Starting Backend (Node.js on port 3000)...{Colors.NC}")
    backend_dir = SCRIPT_DIR / "backend"

    if not (backend_dir / "node_modules").exists():
        print(f"{Colors.YELLOW}Installing backend dependencies...{Colors.NC}")
        subprocess.run(["npm", "install"], cwd=backend_dir, check=True)

    logs_dir = SCRIPT_DIR / "logs"
    log_file = logs_dir / "backend.log"
    backend_cmd = ["nodemon", "app.js"] if shutil.which("nodemon") else ["node", "app.js"]

    with open(log_file, "w") as f:
        proc = subprocess.Popen(
            backend_cmd,
            cwd=backend_dir,
            stdout=f,
            stderr=subprocess.STDOUT
        )
    processes.append(proc)
    return proc

def start_frontend():
    print(f"{Colors.GREEN}[2/3] Starting Frontend (React on port 5173)...{Colors.NC}")
    frontend_dir = SCRIPT_DIR / "frontend"

    if not (frontend_dir / "node_modules").exists():
        print(f"{Colors.YELLOW}Installing frontend dependencies...{Colors.NC}")
        subprocess.run(["npm", "install"], cwd=frontend_dir, check=True)

    log_file = SCRIPT_DIR / "logs" / "frontend.log"
    with open(log_file, "w") as f:
        proc = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=frontend_dir,
            stdout=f,
            stderr=subprocess.STDOUT
        )
    processes.append(proc)
    return proc

def start_flask():
    print(f"{Colors.GREEN}[3/3] Starting Flask Video Server (Python on port {FLASK_PORT})...{Colors.NC}")
    flask_dir = SCRIPT_DIR / "flaskServer"

    venv_path = None
    for venv_name in ["myenv", "venv", "env"]:
        venv_dir = flask_dir / venv_name
        if venv_dir.exists():
            venv_path = venv_dir
            break

    venv_created = False
    if venv_path is None:
        print(f"{Colors.YELLOW}Creating Python virtual environment...{Colors.NC}")
        venv_path = flask_dir / "myenv"
        subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
        venv_created = True
        time.sleep(2)

    python_exe = venv_path / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python_exe.exists():
        raise FileNotFoundError(f"Python executable not found: {python_exe}")

    deps_file = flask_dir / ".deps_installed"
    if not deps_file.exists() or venv_created:
        print(f"{Colors.YELLOW}Installing Python dependencies...{Colors.NC}")
        subprocess.run([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"], cwd=flask_dir, check=True)
        subprocess.run([str(python_exe), "-m", "pip", "install", "-r", "requirements.txt", "flask-cors"], cwd=flask_dir, check=True)
        deps_file.touch()

    log_file = SCRIPT_DIR / "logs" / "flask.log"
    env = os.environ.copy()
    env["FLASK_PORT"] = str(FLASK_PORT)
    with open(log_file, "w") as f:
        proc = subprocess.Popen(
            [str(python_exe), "videoServer.py"],
            cwd=flask_dir,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env
        )
    processes.append(proc)
    return proc

def main():
    print_header()
    check_dependencies()

    try:
        backend_proc = start_backend()
        frontend_proc = start_frontend()
        flask_proc = start_flask()

        print(f"\n{Colors.BLUE}Verifying services...{Colors.NC}\n")
        results = [
            wait_for_service("Backend", 3000, backend_proc, SCRIPT_DIR / "logs" / "backend.log", 30),
            wait_for_service("Frontend", 5173, frontend_proc, SCRIPT_DIR / "logs" / "frontend.log", 30),
            wait_for_service("Flask", FLASK_PORT, flask_proc, SCRIPT_DIR / "logs" / "flask.log", 120),
        ]

        print("")
        if not all(results):
            print(f"{Colors.RED}One or more services failed to start.{Colors.NC}")
            print(f"{Colors.YELLOW}Check logs in the logs/ directory for details.{Colors.NC}")
            cleanup()

        print(f"{Colors.BLUE}{'='*50}{Colors.NC}")
        print(f"{Colors.GREEN}All services started successfully!{Colors.NC}")
        print(f"{Colors.BLUE}{'='*50}{Colors.NC}")
        print(f"{Colors.GREEN}Frontend:{Colors.NC}  http://localhost:5173")
        print(f"{Colors.GREEN}Backend:{Colors.NC}   http://localhost:3000")
        print(f"{Colors.GREEN}Flask:{Colors.NC}     http://localhost:{FLASK_PORT}")
        print(f"{Colors.BLUE}{'='*50}{Colors.NC}\n")
        print(f"{Colors.YELLOW}Logs:{Colors.NC}")
        print("  - logs/backend.log")
        print("  - logs/frontend.log")
        print("  - logs/flask.log")
        print(f"\n{Colors.YELLOW}Press Ctrl+C to stop all services{Colors.NC}\n")

        while True:
            time.sleep(1)
            for i, proc in enumerate(processes):
                if proc.poll() is not None:
                    print(f"{Colors.RED}Process {i + 1} exited with code {proc.returncode}{Colors.NC}")
                    cleanup()

    except KeyboardInterrupt:
        cleanup()
    except Exception as e:
        print(f"{Colors.RED}Error: {e}{Colors.NC}")
        cleanup()

if __name__ == "__main__":
    main()
