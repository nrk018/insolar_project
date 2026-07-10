@echo off
REM Insolare Safety System - Single Command Startup Script (Windows)
REM Starts Frontend, Backend, and Flask Video Server

echo ========================================
echo   Insolare Safety System
echo   Starting all services...
echo ========================================
echo.

cd /d "%~dp0"

if not exist "logs" mkdir logs
if not exist "backend\uploads" mkdir backend\uploads

if not exist "backend\.env" (
    echo Warning: backend\.env file not found!
    echo Please create backend\.env with required environment variables.
    echo.
)

if not defined FLASK_PORT set FLASK_PORT=5000

echo Flask target port: %FLASK_PORT%
echo If port 5000 is busy, videoServer will auto-fallback to 5001 or 5050.
echo Set frontend/.env VITE_FLASK_URL to match the port shown in logs\flask.log
echo.

REM 1. Start Backend (Node.js)
echo [1/3] Starting Backend (Node.js on port 3000)...
cd backend
if not exist "node_modules" (
    echo Installing backend dependencies...
    call npm install
)
where nodemon >nul 2>&1
if %errorlevel%==0 (
    start "Backend Server" cmd /k "nodemon app.js > ..\logs\backend.log 2>&1"
) else (
    start "Backend Server" cmd /k "node app.js > ..\logs\backend.log 2>&1"
)
cd ..
echo Backend launch requested
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
echo Frontend launch requested
echo.

REM 3. Start Flask Video Server
echo [3/3] Starting Flask Video Server (Python on port %FLASK_PORT%)...
cd flaskServer

if not exist "myenv" if not exist "venv" if not exist "env" (
    echo Creating Python virtual environment...
    python -m venv myenv
)

if exist "myenv\Scripts\activate.bat" (
    call myenv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else if exist "env\Scripts\activate.bat" (
    call env\Scripts\activate.bat
)

if not exist ".deps_installed" (
    echo Installing Python dependencies...
    pip install -r requirements.txt flask-cors
    type nul > .deps_installed
)

set FLASK_PORT=%FLASK_PORT%
start "Flask Video Server" cmd /k "set FLASK_PORT=%FLASK_PORT% && python videoServer.py > ..\logs\flask.log 2>&1"
cd ..

echo.
echo Waiting for services to start...
timeout /t 5 /nobreak >nul

echo.
echo ========================================
echo Services launched
echo ========================================
echo Frontend:  http://localhost:5173
echo Backend:   http://localhost:3000
echo Flask:     http://localhost:%FLASK_PORT%
echo ========================================
echo.
echo Check logs if a service does not respond:
echo   - logs\backend.log
echo   - logs\frontend.log
echo   - logs\flask.log
echo.
echo Close the command windows to stop services
echo.
pause
