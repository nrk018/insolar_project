#!/bin/bash

# Insolare Safety System - Single Command Startup Script
# Starts Frontend, Backend, and Flask Video Server

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

BACKEND_PID=""
FRONTEND_PID=""
FLASK_PID=""
FLASK_PORT=""
FAILED=0

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Insolare Safety System${NC}"
echo -e "${BLUE}  Starting all services...${NC}"
echo -e "${BLUE}========================================${NC}\n"

cleanup() {
    echo -e "\n${YELLOW}Shutting down all services...${NC}"
    [ -n "$BACKEND_PID" ] && kill "$BACKEND_PID" 2>/dev/null || true
    [ -n "$FRONTEND_PID" ] && kill "$FRONTEND_PID" 2>/dev/null || true
    [ -n "$FLASK_PID" ] && kill "$FLASK_PID" 2>/dev/null || true
    exit "${1:-0}"
}

trap 'cleanup 130' INT TERM

port_in_use() {
    lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
}

pick_flask_port() {
    if [ -n "${FLASK_PORT:-}" ]; then
        echo "$FLASK_PORT"
        return
    fi
    local port
    for port in 5000 5001 5050; do
        if ! port_in_use "$port"; then
            echo "$port"
            return
        fi
    done
    echo 5001
}

wait_for_service() {
    local name="$1"
    local port="$2"
    local pid="$3"
    local log="$4"
    local timeout="${5:-30}"

    for _ in $(seq 1 "$timeout"); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo -e "${RED}✗ $name failed (process exited). Last log lines:${NC}"
            tail -10 "$log" 2>/dev/null || true
            return 1
        fi
        if port_in_use "$port"; then
            echo -e "${GREEN}✓ $name listening on http://localhost:$port (PID: $pid)${NC}"
            return 0
        fi
        sleep 1
    done

    echo -e "${RED}✗ $name timed out waiting for port $port. See $log${NC}"
    tail -10 "$log" 2>/dev/null || true
    return 1
}

mkdir -p logs backend/uploads

if [ ! -f "backend/.env" ]; then
    echo -e "${YELLOW}Warning: backend/.env file not found!${NC}"
    echo -e "${YELLOW}Please create backend/.env with required environment variables.${NC}\n"
fi

FLASK_PORT="$(pick_flask_port)"
export FLASK_PORT

if port_in_use 5000 && [ "$FLASK_PORT" != "5000" ]; then
    echo -e "${YELLOW}Note: Port 5000 is in use (often macOS AirPlay Receiver).${NC}"
    echo -e "${YELLOW}Flask will use port $FLASK_PORT instead.${NC}"
    if [ ! -f "frontend/.env" ] || ! grep -q "VITE_FLASK_URL=http://localhost:$FLASK_PORT" frontend/.env 2>/dev/null; then
        echo -e "${YELLOW}Add to frontend/.env: VITE_FLASK_URL=http://localhost:$FLASK_PORT${NC}\n"
    fi
fi

# 1. Start Backend (Node.js)
echo -e "${GREEN}[1/3] Starting Backend (Node.js on port 3000)...${NC}"
cd backend
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Installing backend dependencies...${NC}"
    npm install
fi
if command -v nodemon >/dev/null 2>&1; then
    nodemon app.js > ../logs/backend.log 2>&1 &
else
    node app.js > ../logs/backend.log 2>&1 &
fi
BACKEND_PID=$!
cd ..

# 2. Start Frontend (React)
echo -e "${GREEN}[2/3] Starting Frontend (React on port 5173)...${NC}"
cd frontend
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Installing frontend dependencies...${NC}"
    npm install
fi
npm run dev > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

# 3. Start Flask Video Server
echo -e "${GREEN}[3/3] Starting Flask Video Server (Python on port $FLASK_PORT)...${NC}"
cd flaskServer

VENV_CREATED=false
if [ ! -d "myenv" ] && [ ! -d "venv" ] && [ ! -d "env" ]; then
    echo -e "${YELLOW}Creating Python virtual environment...${NC}"
    python3 -m venv myenv
    VENV_CREATED=true
    sleep 2
fi

if [ -d "myenv" ]; then
    source myenv/bin/activate
    PYTHON_EXE="myenv/bin/python"
elif [ -d "venv" ]; then
    source venv/bin/activate
    PYTHON_EXE="venv/bin/python"
elif [ -d "env" ]; then
    source env/bin/activate
    PYTHON_EXE="env/bin/python"
fi

if [ ! -f ".deps_installed" ] || [ "$VENV_CREATED" = true ]; then
    echo -e "${YELLOW}Installing Python dependencies...${NC}"
    $PYTHON_EXE -m pip install --upgrade pip
    $PYTHON_EXE -m pip install -r requirements.txt flask-cors
    touch .deps_installed
fi

python videoServer.py > ../logs/flask.log 2>&1 &
FLASK_PID=$!
cd ..

echo -e "\n${BLUE}Verifying services...${NC}\n"
wait_for_service "Backend" 3000 "$BACKEND_PID" "logs/backend.log" 30 || FAILED=1
wait_for_service "Frontend" 5173 "$FRONTEND_PID" "logs/frontend.log" 30 || FAILED=1
wait_for_service "Flask" "$FLASK_PORT" "$FLASK_PID" "logs/flask.log" 120 || FAILED=1

echo ""
if [ "$FAILED" -ne 0 ]; then
    echo -e "${RED}One or more services failed to start.${NC}"
    echo -e "${YELLOW}Check logs in the logs/ directory for details.${NC}"
    cleanup 1
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}All services started successfully!${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Frontend:${NC}  http://localhost:5173"
echo -e "${GREEN}Backend:${NC}   http://localhost:3000"
echo -e "${GREEN}Flask:${NC}     http://localhost:$FLASK_PORT"
echo -e "${BLUE}========================================${NC}\n"
echo -e "${YELLOW}Logs:${NC}"
echo -e "  - logs/backend.log"
echo -e "  - logs/frontend.log"
echo -e "  - logs/flask.log"
echo -e "\n${YELLOW}Press Ctrl+C to stop all services${NC}\n"

wait
