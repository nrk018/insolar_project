# Insolare Safety System

A workplace safety monitoring platform that combines **employee face recognition**, **PPE (Personal Protective Equipment) detection**, and **live camera monitoring** (RTSP IP cameras or laptop webcam).

---

## System overview

```mermaid
flowchart LR
    subgraph Browser
        UI[React Frontend :5173]
    end
    subgraph Server
        API[Node.js Backend :3000]
        ML[Flask Video Server :5000/5001]
    end
    subgraph Data
        DB[(Supabase PostgreSQL)]
        FS[backend/uploads]
    end
    Cam[RTSP IP Camera / Webcam]

    UI -->|login, records, admin| API
    UI -->|live feed, camera control| ML
    API --> DB
    ML -->|detection events| API
    ML --> Cam
    API --> FS
    ML --> FS
```

| Service | Port | Role |
|---------|------|------|
| **Frontend** | 5173 | Web UI — dashboard, camera monitor, image analysis, admin |
| **Backend** | 3000 | REST API, auth, Supabase, employee uploads, detection storage |
| **Flask** | 5000 or 5001 | Face recognition, PPE detection, MJPEG stream, RTSP/webcam |

The frontend talks to the backend for auth and data. It talks to Flask directly for the live video feed and camera controls.

---

## Prerequisites

- **Node.js** 18+ (20+ recommended)
- **Python** 3.9+
- **Supabase** account (free tier works) — used as the database (**MongoDB is not used**)
- **npm** (comes with Node.js)
- For **RTSP cameras**: camera and server on the same network (or routable IP)

---

## Quick start (recommended)

From the project root:

```bash
chmod +x start.sh   # first time only
./start.sh
```

This script:

1. Creates `logs/` and `backend/uploads/` if missing
2. Installs dependencies on first run (npm + Python venv)
3. Starts backend, frontend, and Flask in parallel
4. Verifies each service is actually listening before reporting success
5. Auto-picks Flask port **5001** if **5000** is taken (common on macOS due to AirPlay Receiver)

Press **Ctrl+C** to stop all services.

Alternative launchers: `start.py` (cross-platform), `start.bat` (Windows).

Logs: `logs/backend.log`, `logs/frontend.log`, `logs/flask.log`

---

## Environment setup

### 1. Backend — `backend/.env`

Copy the template and fill in your values:

```bash
cp backend/.env.example backend/.env
```

| Variable | Required | Where to get it |
|----------|----------|-----------------|
| `SUPABASE_URL` | Yes | Supabase → Project Settings → API → **Project URL** |
| `SUPABASE_ANON_KEY` | Yes | Same page → **anon public** key |
| `SUPABASE_SERVICE_ROLE_KEY` | Yes | Same page → **service_role** key (keep secret) |
| `JWT_SECRET` | Yes | Any long random string: `openssl rand -hex 32` |

Optional (email/SMS alerts):

| Variable | Purpose |
|----------|---------|
| `EMAIL_USER` / `EMAIL_PASS` | SMTP / Gmail for notifications |
| `SMSIDEA_USERNAME` / `SMSIDEA_PASSWORD` / `SMSIDEA_SENDER_ID` | SMS via SMSIdea |
| `FLASK_PORT` | Flask port if not 5001 (used when notifying Flask to reload embeddings) |
| `FLASK_URL` | Full Flask URL, e.g. `http://localhost:5001` |

### 2. Frontend — `frontend/.env`

```bash
cp frontend/.env.example frontend/.env
```

```env
VITE_FLASK_URL=http://localhost:5001
```

Use **5001** on macOS when AirPlay Receiver occupies port 5000. Match this to whatever port Flask actually starts on (shown by `./start.sh`).

### 3. Flask — RTSP URL (optional)

Set before starting Flask if you use an IP camera:

```bash
export RTSP_URL="rtsp://username:password@192.168.1.100:554/stream1?rtsp_transport=tcp"
```

Or add to a `flaskServer/.env` and source it before launch. Default in code if unset:

```
rtsp://admin:InSolare%402025@192.168.1.2:554/stream1?rtsp_transport=tcp
```

---

## Supabase database setup

Run these in **Supabase → SQL Editor** (in order):

1. **`supabase/schema.sql`** — `users`, `attendance`, `detection_events`
2. **`backend/migrations/add_detection_events.sql`** — only if `detection_events` was not created by step 1
3. **`supabase/migration_add_ppe.sql`** — PPE columns on `attendance` (if table already existed)

The backend uses the **service role** key for server-side writes (registration, detections). You do **not** need MongoDB.

---

## RTSP camera configuration

The Flask server connects to IP cameras over **RTSP** using OpenCV + FFmpeg. You can also use a **laptop webcam** from the UI without any RTSP setup.

### How to get your RTSP URL

1. **Find the camera IP**
   - Check your router’s DHCP client list, or
   - Use the manufacturer’s discovery tool (e.g. Hikvision SADP, Dahua Config Tool), or
   - Log into the camera web UI (often `http://192.168.x.x`).

2. **Get credentials**
   - Default is often `admin` + a password set during camera setup.
   - Use the username/password from the camera’s web interface or NVR.

3. **Find the RTSP path** (varies by brand)

   | Brand | Example path |
   |-------|----------------|
   | Hikvision | `/Streaming/Channels/101` or `/101` |
   | Dahua | `/cam/realmonitor?channel=1&subtype=0` |
   | Generic / many NVRs | `/stream1`, `/live`, `/h264` |

4. **Build the URL**

   ```
   rtsp://USERNAME:PASSWORD@CAMERA_IP:554/PATH?rtsp_transport=tcp
   ```

   - Port **554** is the standard RTSP port.
   - **`?rtsp_transport=tcp`** is recommended for stable streaming on most networks.

5. **URL-encode special characters in the password**

   If the password contains `@`, `#`, `:`, etc., encode them:

   | Character | Encoded |
   |-----------|---------|
   | `@` | `%40` |
   | `#` | `%23` |
   | `:` | `%3A` |

   Example: password `Test@1122` → `Test%401122`

   Full example:

   ```
   rtsp://admin:Test%401122@192.168.1.216:554/101?rtsp_transport=tcp
   ```

6. **Test in VLC (important)**

   - Open **VLC → File → Open Network…**
   - Paste the RTSP URL (you can use the raw password here, not URL-encoded, depending on VLC)
   - If video plays in VLC, the URL works; use the encoded form in `RTSP_URL` for Flask

7. **Configure the project**

   ```bash
   export RTSP_URL="rtsp://admin:Your%40Password@192.168.1.216:554/101?rtsp_transport=tcp"
   ./start.sh
   ```

   Or edit the default in `flaskServer/videoServer.py` (look for `RTSP_URL = os.getenv(...)`).

### Using RTSP in the app

1. Open http://localhost:5173 and sign in
2. Go to **Camera Monitor** (or use camera controls on the Dashboard)
3. Click **Start RTSP**
4. The live feed appears at `http://localhost:5001/video_feed` (proxied via `VITE_FLASK_URL`)

Flask does **not** auto-start the camera; you must click **Start RTSP** or **Start Webcam** in the UI.

### RTSP troubleshooting

| Issue | What to try |
|-------|-------------|
| “RTSP camera not found” | Confirm URL in VLC; ping camera IP; same Wi‑Fi/LAN as the machine running Flask |
| Wrong path | Try alternate paths from your camera manual (see table above) |
| Auth failed | Double-check username/password; URL-encode `@` in password |
| Slow / frozen stream | Keep `rtsp_transport=tcp`; reduce camera resolution in camera settings |
| Works in VLC, not in app | Use URL-encoded password in `RTSP_URL`; restart `./start.sh` after changing env |

### Webcam (macOS)

On first use, macOS may prompt for **camera permission** for Terminal or your IDE. Allow it.

If OpenCV fails with authorization errors, grant access in **System Settings → Privacy & Security → Camera**, then restart Flask.

---

## Manual installation (alternative to `start.sh`)

### Backend

```bash
cd backend
npm install
cp .env.example .env   # then edit .env
node app.js            # http://localhost:3000
```

### Frontend

```bash
cd frontend
npm install
cp .env.example .env   # set VITE_FLASK_URL
npm run dev            # http://localhost:5173
```

If you see a `lightningcss` native module error:

```bash
cd frontend
rm -rf node_modules
npm install
```

### Flask

```bash
cd flaskServer
python3 -m venv myenv
source myenv/bin/activate   # Windows: myenv\Scripts\activate
pip install -r requirements.txt flask-cors
export RTSP_URL="rtsp://..."   # optional
export FLASK_PORT=5001         # if 5000 is busy on macOS
python videoServer.py
```

---

## Face recognition and recent detections

### Register employees with photos

1. Sign up at `/signup` or add employees via **Admin**
2. Upload **multiple clear face photos** per person
3. On registration, the backend runs `createEmbeddings.py` and notifies Flask to reload embeddings

Reload embeddings manually:

```bash
curl -X POST http://localhost:5001/api/embeddings/reload
```

Check status:

```bash
curl http://localhost:5001/api/health
```

Look for `"face_recognition": true` and `"embeddings_count" > 0`.

### Recent detections panel

When the camera sees a **recognized** employee (not “Unknown”):

1. Flask saves an annotated snapshot
2. Flask POSTs to `http://localhost:3000/api/detections/event`
3. Backend stores/updates a row in **`detection_events`**
4. Frontend polls `/api/detections/recent` every few seconds

Requirements:

- `detection_events` table exists in Supabase (see [Supabase setup](#supabase-database-setup))
- Face embeddings loaded for at least one employee
- Camera running and person visible in frame

---

## Features

- Employee registration and login (JWT cookies)
- Admin employee management
- Live camera monitor (RTSP + webcam)
- Real-time face recognition and anti-spoofing
- PPE detection (helmet, gloves, boots, jacket)
- Recent detections with snapshots
- Single-image analysis upload
- Attendance and detection history
- Optional email/SMS notifications for PPE violations

---

## Project structure

```
insolar_project/
├── frontend/          React + Vite + Tailwind (port 5173)
├── backend/           Express API + Supabase (port 3000)
├── flaskServer/       videoServer.py — ML + camera (port 5000/5001)
├── supabase/          SQL schema and migrations
├── start.sh           Start all services (macOS/Linux)
├── start.py           Start all services (cross-platform)
├── start.bat          Start all services (Windows)
└── logs/              Runtime logs (created by start scripts)
```

Key files:

| File | Purpose |
|------|---------|
| `flaskServer/videoServer.py` | Main ML server, RTSP/webcam, `/video_feed` |
| `flaskServer/ppeDetection.py` | PPE YOLO model |
| `flaskServer/createEmbeddings.py` | Build face embeddings from upload photos |
| `backend/app.js` | API routes, auth, detection events |
| `frontend/src/Pages/CameraMonitor.jsx` | Live camera + recent detections UI |

---

## Common issues

| Symptom | Fix |
|---------|-----|
| `./start.sh` says success but nothing runs | Check `logs/*.log`; ensure `logs/` exists (fixed in current `start.sh`) |
| Port 5000 in use | Disable macOS **AirPlay Receiver**, or use port 5001 + `VITE_FLASK_URL=http://localhost:5001` |
| Failed to register user | Fill `backend/.env`; run Supabase `schema.sql`; backend uses service role key |
| No recent detections | Create `detection_events` table; reload embeddings; ensure face is recognized by name |
| Face always “Unknown” | Register with photos; `curl -X POST .../api/embeddings/reload`; restart Flask |
| Frontend `lightningcss` error | `cd frontend && rm -rf node_modules && npm install` |
| RTSP fails | Test URL in VLC; set `RTSP_URL` with encoded password; same network as camera |

---

## Development notes

- Frontend proxies `/api`, `/login`, `/register` to the backend via `vite.config.js`
- Flask CORS allows `http://localhost:5173`
- Employee photos and embeddings live in `backend/uploads/<Employee Name>/`
- PPE model: `flaskServer/runs/detect/ppe_detection/weights/best.pt` (falls back to pretrained if missing)
- Detection snapshots: `flaskServer/detection_snapshots/` (cleaned periodically)

---

## License

Internal / project use — see repository owner for licensing terms.
