@echo off
REM Insolare Safety System - Single Command Startup Script (Windows)
REM This script starts all services: Frontend, Backend, and Flask Video Server

echo ========================================
echo   Insolare Safety System
echo   Starting all services...
echo ========================================
echo.

cd /d "%~dp0"

REM Create logs directory
if not exist "logs" mkdir logs

REM Check if .env file exists
if not exist "backend\.env" (
    echo Warning: backend\.env file not found!
    echo Please create backend\.env with required environment variables.
    echo.
)

REM 1. Start Backend (Node.js)
echo [1/3] Starting Backend (Node.js on port 3000)...
cd backend
if not exist "node_modules" (
    echo Installing backend dependencies...
    call npm install
)
start "Backend Server" cmd /k "nodemon app.js > ..\logs\backend.log 2>&1"
cd ..
echo Backend started
echo.

REM 2. Start Frontend (React)
echo [2/3] Starting Frontend (React on port 5173)...
cd frontend
if not exist "node_modules" (
    echo Installing frontend dependencies...
    call npm install
)
start "Frontend Server" cmd /k "npm run dev > ..\logs\frontend.log 2>&1"
cd ..
echo Frontend started
echo.

REM 3. Start Flask Video Server
echo [3/3] Starting Flask Video Server (Python on port 5000)...
cd flaskServer

REM Check if virtual environment exists
if not exist "myenv" if not exist "venv" if not exist "env" (
    echo Creating Python virtual environment...
    python -m venv myenv
)

REM Activate virtual environment
if exist "myenv\Scripts\activate.bat" (
    call myenv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else if exist "env\Scripts\activate.bat" (
    call env\Scripts\activate.bat
)

REM Install Python dependencies if needed
if not exist ".deps_installed" (
    echo Installing Python dependencies...
    pip install -r requirements.txt flask-cors
    type nul > .deps_installed
)

REM Start Flask server
start "Flask Video Server" cmd /k "python videoServer.py > ..\logs\flask.log 2>&1"
cd ..

echo.
echo ========================================
echo All services started successfully!
echo ========================================
echo Frontend:  http://localhost:5173
echo Backend:   http://localhost:3000
echo Flask:     http://localhost:5000
echo ========================================
echo.
echo Logs are being written to:
echo   - logs\backend.log
echo   - logs\frontend.log
echo   - logs\flask.log
echo.
echo Close the command windows to stop services
echo.
pause

