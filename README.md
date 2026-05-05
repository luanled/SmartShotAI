# SmartShotAI

An AI-powered mobile photography app that scores photos for aesthetic quality in real-time. Point your camera at a scene and get an instant aesthetic score — then capture the shot to see how it compares.

## Repository Structure

```
SmartShotAI/
├── mobile_app/          # React Native (Expo) app
│   ├── screens/         # LiveCameraScreen, SettingsScreen
│   ├── hooks/           # useAestheticScorer — calls backend /score endpoint
│   ├── context/         # ModelContext — tracks selected scoring model
│   ├── styles/          # StyleSheet files per screen
│   └── config.js        # BACKEND_URL configuration
├── backend/             # FastAPI inference server
│   ├── main.py          # /score and /health endpoints
│   └── requirements.txt
└── scorer/              # Aesthetic scoring models (see scorer/readme.md)
```

## Scoring Models

Three models are available, selectable from the app's Settings screen:

| Key | Size | Description |
|-----|------|-------------|
| `optimized` | ~7 MB | EMOv2 finetuned — fastest, default |
| `standard` | ~20 MB | EMOv2 finetuned — full quality |
| `siglip` | ~356 MB | SigLIP finetuned — highest accuracy |

> Models are not tracked in git due to file size. See `tflite/readme.md.txt` for details.

## Mobile App Setup

### Prerequisites

- Node.js 18+ and npm
- Expo Go app on your phone (or a simulator)
- Backend server running (see below)

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Verify: `http://localhost:8000/health`

### 2. Configure the Backend URL

Edit `mobile_app/config.js` to point to your machine:

```js
// iOS Simulator
export const BACKEND_URL = 'http://localhost:8000';

// Android Emulator
export const BACKEND_URL = 'http://10.0.2.2:8000';

// Physical device — use your machine's LAN IP
export const BACKEND_URL = 'http://192.168.x.x:8000';
```

### 3. Run the App

```bash
cd mobile_app
npm install
npx expo start
```

Scan the QR code with Expo Go, or press `i` for iOS simulator / `a` for Android.

## API Reference

### `POST /score`

Score an image for aesthetic quality.

| Field | Type | Description |
|-------|------|-------------|
| `file` | image | JPEG/PNG image |
| `model` | string | `optimized` \| `standard` \| `siglip` (default: `optimized`) |
| `source` | string | `live` \| `capture` (for logging) |

**Response:**
```json
{
  "score": 72,
  "score_linear": 65,
  "raw_score": 6.5432,
  "server_ms": 48.3
}
```

### `GET /health`

```json
{ "status": "ok", "models": ["standard", "optimized", "siglip"] }
```
