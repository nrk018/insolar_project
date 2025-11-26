# Insolare Safety System

A safety monitoring system that detects Personal Protective Equipment (PPE) compliance and provides employee face recognition capabilities.

## System Architecture

The application consists of three main components:
- Frontend: React-based web interface
- Backend: Node.js/Express API server
- Flask Server: Python-based machine learning service for face recognition and PPE detection

## Prerequisites

- Node.js (v14 or higher)
- Python 3.8 or higher
- MongoDB database
- Supabase account and credentials

## Installation

### Backend Installation

1. Navigate to the backend directory:
```bash
cd backend
```

2. Install dependencies:
```bash
npm install
```

3. Create a `.env` file in the backend directory with the following variables:
```
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_anon_key
JWT_SECRET=your_jwt_secret
MONGODB_URI=your_mongodb_connection_string
```

4. Start the backend server:
```bash
node app.js
```

The backend server will run on port 3000.

### Frontend Installation

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Create a `.env` file in the frontend directory with the following variables:
```
VITE_FLASK_URL=http://localhost:5000
```

4. Start the development server:
```bash
npm run dev
```

The frontend will run on port 5173.

### Flask Server Installation

1. Navigate to the flaskServer directory:
```bash
cd flaskServer
```

2. Create a virtual environment (recommended):
```bash
python -m venv myenv
```

3. Activate the virtual environment:
   - On macOS/Linux:
   ```bash
   source myenv/bin/activate
   ```
   - On Windows:
   ```bash
   myenv\Scripts\activate
   ```

4. Install Python dependencies:
```bash
pip install -r requirements.txt
```

5. Start the Flask server:
```bash
python videoServer.py
```

The Flask server will run on port 5000.

## Running the Application

1. Start the backend server (port 3000)
2. Start the Flask server (port 5000)
3. Start the frontend development server (port 5173)

Access the application at http://localhost:5173

## Features

- Employee face recognition
- Real-time PPE compliance detection
- Camera monitoring
- Image analysis
- Detection history and records
- Admin dashboard for employee management

## Configuration

Ensure all three servers are running simultaneously for the application to function properly. The frontend communicates with the backend API, and the backend communicates with the Flask server for machine learning operations.

