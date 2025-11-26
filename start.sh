#!/bin/bash

# Insolare Safety System - Single Command Startup Script
# This script starts all services: Frontend, Backend, and Flask Video Server

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Insolare Safety System${NC}"
echo -e "${BLUE}  Starting all services...${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down all services...${NC}"
    kill $BACKEND_PID $FRONTEND_PID $FLASK_PID 2>/dev/null || true
    exit
}

# Trap Ctrl+C and cleanup
trap cleanup INT TERM

# Check if .env file exists
if [ ! -f "backend/.env" ]; then
    echo -e "${RED}Warning: backend/.env file not found!${NC}"
    echo -e "${YELLOW}Please create backend/.env with required environment variables.${NC}\n"
fi

# 1. Start Backend (Node.js)
echo -e "${GREEN}[1/3] Starting Backend (Node.js on port 3000)...${NC}"
cd backend
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Installing backend dependencies...${NC}"
    npm install
fi
nodemon app.js > ../logs/backend.log 2>&1 &
BACKEND_PID=$!
cd ..
echo -e "${GREEN}✓ Backend started (PID: $BACKEND_PID)${NC}\n"

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
echo -e "${GREEN}✓ Frontend started (PID: $FRONTEND_PID)${NC}\n"

# 3. Start Flask Video Server
echo -e "${GREEN}[3/3] Starting Flask Video Server (Python on port 5000)...${NC}"
cd flaskServer

# Check if virtual environment exists
VENV_CREATED=false
if [ ! -d "myenv" ] && [ ! -d "venv" ] && [ ! -d "env" ]; then
    echo -e "${YELLOW}Creating Python virtual environment...${NC}"
    python3 -m venv myenv
    VENV_CREATED=true
    # Wait for venv to be fully created
    sleep 2
fi

# Activate virtual environment
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

# Install Python dependencies if needed
if [ ! -f ".deps_installed" ] || [ "$VENV_CREATED" = true ]; then
    echo -e "${YELLOW}Installing Python dependencies...${NC}"
    $PYTHON_EXE -m pip install --upgrade pip
    $PYTHON_EXE -m pip install -r requirements.txt flask-cors
    touch .deps_installed
fi

# Start Flask server
python videoServer.py > ../logs/flask.log 2>&1 &
FLASK_PID=$!
cd ..
echo -e "${GREEN}✓ Flask server started (PID: $FLASK_PID)${NC}\n"

# Create logs directory if it doesn't exist
mkdir -p logs

# Wait a bit for services to start
sleep 3

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}All services started successfully!${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Frontend:${NC}  http://localhost:5173"
echo -e "${GREEN}Backend:${NC}   http://localhost:3000"
echo -e "${GREEN}Flask:${NC}    http://localhost:5000"
echo -e "${BLUE}========================================${NC}\n"
echo -e "${YELLOW}Logs are being written to:${NC}"
echo -e "  - logs/backend.log"
echo -e "  - logs/frontend.log"
echo -e "  - logs/flask.log"
echo -e "\n${YELLOW}Press Ctrl+C to stop all services${NC}\n"

# Wait for all background processes
wait

