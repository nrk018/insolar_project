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
from pathlib import Path

# Colors for terminal output
class Colors:
    GREEN = '\033[0;32m'
    BLUE = '\033[0;34m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    NC = '\033[0m'  # No Color

# Get script directory
SCRIPT_DIR = Path(__file__).parent.absolute()
os.chdir(SCRIPT_DIR)

# Process IDs to track
processes = []

def cleanup(signum=None, frame=None):
    """Cleanup function to stop all processes"""
    print(f"\n{Colors.YELLOW}Shutting down all services...{Colors.NC}")
    for proc in processes:
        try:
            if proc.poll() is None:  # Process is still running
                proc.terminate()
                proc.wait(timeout=5)
        except:
            try:
                proc.kill()
            except:
                pass
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

def print_header():
    """Print startup header"""
    print(f"{Colors.BLUE}{'='*50}{Colors.NC}")
    print(f"{Colors.BLUE}  Insolare Safety System{Colors.NC}")
    print(f"{Colors.BLUE}  Starting all services...{Colors.NC}")
    print(f"{Colors.BLUE}{'='*50}{Colors.NC}\n")

def check_dependencies():
    """Check if required files exist"""
    if not (SCRIPT_DIR / "backend" / ".env").exists():
        print(f"{Colors.YELLOW}Warning: backend/.env file not found!{Colors.NC}")
        print(f"{Colors.YELLOW}Please create backend/.env with required environment variables.{Colors.NC}\n")

def start_backend():
    """Start Node.js backend"""
    print(f"{Colors.GREEN}[1/3] Starting Backend (Node.js on port 3000)...{Colors.NC}")
    backend_dir = SCRIPT_DIR / "backend"
    
    # Check if node_modules exists
    if not (backend_dir / "node_modules").exists():
        print(f"{Colors.YELLOW}Installing backend dependencies...{Colors.NC}")
        subprocess.run(["npm", "install"], cwd=backend_dir, check=True)
    
    # Create logs directory
    logs_dir = SCRIPT_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Start backend with nodemon
    log_file = logs_dir / "backend.log"
    with open(log_file, "w") as f:
        proc = subprocess.Popen(
            ["nodemon", "app.js"],
            cwd=backend_dir,
            stdout=f,
            stderr=subprocess.STDOUT
        )
    processes.append(proc)
    print(f"{Colors.GREEN}✓ Backend started (PID: {proc.pid}){Colors.NC}\n")
    return proc

def start_frontend():
    """Start React frontend"""
    print(f"{Colors.GREEN}[2/3] Starting Frontend (React on port 5173)...{Colors.NC}")
    frontend_dir = SCRIPT_DIR / "frontend"
    
    # Check if node_modules exists
    if not (frontend_dir / "node_modules").exists():
        print(f"{Colors.YELLOW}Installing frontend dependencies...{Colors.NC}")
        subprocess.run(["npm", "install"], cwd=frontend_dir, check=True)
    
    # Start frontend
    logs_dir = SCRIPT_DIR / "logs"
    log_file = logs_dir / "frontend.log"
    with open(log_file, "w") as f:
        proc = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=frontend_dir,
            stdout=f,
            stderr=subprocess.STDOUT
        )
    processes.append(proc)
    print(f"{Colors.GREEN}✓ Frontend started (PID: {proc.pid}){Colors.NC}\n")
    return proc

def start_flask():
    """Start Flask video server"""
    print(f"{Colors.GREEN}[3/3] Starting Flask Video Server (Python on port 5000)...{Colors.NC}")
    flask_dir = SCRIPT_DIR / "flaskServer"
    
    # Find virtual environment
    venv_path = None
    for venv_name in ["myenv", "venv", "env"]:
        venv_dir = flask_dir / venv_name
        if venv_dir.exists():
            venv_path = venv_dir
            break
    
    # Create venv if it doesn't exist
    venv_created = False
    if venv_path is None:
        print(f"{Colors.YELLOW}Creating Python virtual environment...{Colors.NC}")
        venv_path = flask_dir / "myenv"
        subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
        venv_created = True
        # Wait a moment for venv to be fully created
        time.sleep(2)
    
    # Determine Python executable in venv
    if sys.platform == "win32":
        python_exe = venv_path / "Scripts" / "python.exe"
    else:
        python_exe = venv_path / "bin" / "python"
    
    # Verify Python executable exists
    if not python_exe.exists():
        print(f"{Colors.RED}Error: Python executable not found at {python_exe}{Colors.NC}")
        raise FileNotFoundError(f"Python executable not found: {python_exe}")
    
    # Install dependencies if needed
    deps_file = flask_dir / ".deps_installed"
    if not deps_file.exists() or venv_created:
        print(f"{Colors.YELLOW}Installing Python dependencies...{Colors.NC}")
        # Use python -m pip instead of direct pip path (more reliable)
        subprocess.run([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"], 
                      cwd=flask_dir, check=True)
        subprocess.run([str(python_exe), "-m", "pip", "install", "-r", "requirements.txt", "flask-cors"], 
                      cwd=flask_dir, check=True)
        deps_file.touch()
    
    # Start Flask server
    logs_dir = SCRIPT_DIR / "logs"
    log_file = logs_dir / "flask.log"
    with open(log_file, "w") as f:
        proc = subprocess.Popen(
            [str(python_exe), "videoServer.py"],
            cwd=flask_dir,
            stdout=f,
            stderr=subprocess.STDOUT
        )
    processes.append(proc)
    print(f"{Colors.GREEN}✓ Flask server started (PID: {proc.pid}){Colors.NC}\n")
    return proc

def main():
    """Main function"""
    print_header()
    check_dependencies()
    
    try:
        # Start all services
        backend_proc = start_backend()
        frontend_proc = start_frontend()
        flask_proc = start_flask()
        
        # Wait a bit for services to initialize
        time.sleep(3)
        
        # Print status
        print(f"{Colors.BLUE}{'='*50}{Colors.NC}")
        print(f"{Colors.GREEN}All services started successfully!{Colors.NC}")
        print(f"{Colors.BLUE}{'='*50}{Colors.NC}")
        print(f"{Colors.GREEN}Frontend:{Colors.NC}  http://localhost:5173")
        print(f"{Colors.GREEN}Backend:{Colors.NC}   http://localhost:3000")
        print(f"{Colors.GREEN}Flask:{Colors.NC}    http://localhost:5000")
        print(f"{Colors.BLUE}{'='*50}{Colors.NC}\n")
        print(f"{Colors.YELLOW}Logs are being written to:{Colors.NC}")
        print(f"  - logs/backend.log")
        print(f"  - logs/frontend.log")
        print(f"  - logs/flask.log")
        print(f"\n{Colors.YELLOW}Press Ctrl+C to stop all services{Colors.NC}\n")
        
        # Wait for processes (they run in background)
        # Check process status periodically
        while True:
            time.sleep(1)
            # Check if any process died
            for i, proc in enumerate(processes):
                if proc.poll() is not None:
                    print(f"{Colors.RED}Process {i+1} exited with code {proc.returncode}{Colors.NC}")
                    cleanup()
                    
    except KeyboardInterrupt:
        cleanup()
    except Exception as e:
        print(f"{Colors.RED}Error: {e}{Colors.NC}")
        cleanup()

if __name__ == "__main__":
    main()

