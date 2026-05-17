# Anomaly Detection and Behavior Analysis via Deep Learning on Infrared Videos

A desktop application for real-time infrared video anomaly detection using a cascaded deep learning pipeline: **CR-AE** (Convolutional Recurrent Autoencoder) for anomaly scoring followed by **3D-CNN** for behavior classification.

**Stack:** FastAPI (Python) backend · Flutter (Windows) frontend · Google Drive API v3

---

## How It Works

1. **CR-AE** reconstructs 16-frame clips; high reconstruction error (MSE > threshold) signals an anomaly.
2. **3D-CNN** classifies anomalous clips into one of four behaviors: Normal, Loitering, Trespassing, Object Abandonment.
3. Results are streamed to the UI via WebSocket with per-clip heatmaps and latency breakdowns.

---

## Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.11+ |
| Flutter SDK | 3.x |
| Git | any recent |
| CUDA (optional) | 11.8+ for GPU acceleration |

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/DuranOzcelik/-Anomaly-Detection-and-Behavior-Analysis-via-Deep-Learning-on-Infrared-Videos-.git
cd "-Anomaly-Detection-and-Behavior-Analysis-via-Deep-Learning-on-Infrared-Videos-"
```

### 2. Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

Create a `.env` file in the `backend/` directory:

```
GOOGLE_DRIVE_CREDENTIALS_PATH=google_drive_credentials.json
DRIVE_WATCH_FOLDER_PATH=your/drive/folder/path
```

### 3. Google Drive credentials

The app uses a **service account** to read videos and model weights from Google Drive.

1. Go to [Google Cloud Console](https://console.cloud.google.com/) and create a project.
2. Enable the **Google Drive API**.
3. Create a **Service Account** and download the JSON key file.
4. Rename the key file to `google_drive_credentials.json` and place it in `backend/`.
5. Share your Drive folder with the service account email (view-only is sufficient).

> **Security:** `google_drive_credentials.json` is listed in `.gitignore` and must never be committed to version control. Share it with teammates directly (e.g., via a secure channel), not through git.

### 4. Model weights

Model weights are downloaded automatically from Google Drive on first run and cached in `backend/model_cache/`. Ensure your Drive folder structure matches the paths in `main.py`:

```
CRAE_DRIVE_PATH  = "Data_Subset_.../models/crae_winter_finetuned2.pth"
CNN3D_DRIVE_PATH = "Data_Annotated_.../best_model.pth"
```

### 5. Frontend

```bash
cd frontend
flutter pub get
flutter run -d windows
```

---

## Running

**Backend:**
```bash
cd backend
venv\Scripts\activate
uvicorn main:app --host 127.0.0.1 --port 8000
```

**Frontend** (separate terminal):
```bash
cd frontend
flutter run -d windows
```

The backend must be running before the Flutter app connects.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_DRIVE_CREDENTIALS_PATH` | `google_drive_credentials.json` | Path to service account key |
| `DRIVE_WATCH_FOLDER_PATH` | `archive/system_video_final_test` | Drive folder to watch for videos |

---

## Project Structure

```
ThesisApp/
├── backend/
│   ├── main.py                  # FastAPI app, WebSocket endpoints
│   ├── requirements.txt
│   ├── .env                     # Not committed — create manually
│   ├── google_drive_credentials.json  # Not committed — add manually
│   ├── models/
│   │   ├── CR_AE.py             # Convolutional Recurrent Autoencoder
│   │   └── CNN_3D.py            # 3D-CNN classifier (r3d_18 based)
│   └── services/
│       ├── video_processor.py   # Clip extraction, inference, heatmap
│       ├── drive_service.py     # Google Drive API wrapper
│       ├── drive_watcher.py     # Drive folder polling loop
│       └── camera_processor.py # Live camera stream processor
└── frontend/
    └── lib/
        ├── main.dart            # UI, state management
        ├── services/
        │   ├── api_service.dart       # REST client
        │   └── websocket_service.dart # WebSocket client + message models
        └── widgets/
            └── video_heatmap_overlay.dart
```

---

## Security Notes

- **Credentials file:** `google_drive_credentials.json` is excluded from git via `.gitignore`. Never commit it.
- **`.env` file:** Also excluded from git. Contains only the path to the credentials file.
- **CORS:** The backend restricts allowed origins to `http://127.0.0.1:8000` and `http://localhost:8000`. It is a locally-run app and should not be exposed to the internet.
- **Service account scope:** The service account uses `drive.readonly` scope — it cannot modify or delete any files on Drive.
- **Model cache:** Cached weights are stored in `backend/model_cache/` which is excluded from git via `.gitignore`.
- **Temp videos:** Downloaded videos are stored temporarily in `backend/temp_videos/` (also excluded from git) and are not persisted between sessions.

---

## WebSocket API

| Endpoint | Purpose |
|----------|---------|
| `ws://127.0.0.1:8000/ws` | Date-range batch analysis |
| `ws://127.0.0.1:8000/ws/drive` | Drive folder watch / single video |
| `ws://127.0.0.1:8000/ws/camera` | Live camera stream |

---

## Acknowledgements

- [LTD Dataset](https://github.com/LTD-Dataset) — infrared surveillance video dataset used for training
- [torchvision r3d_18](https://pytorch.org/vision/stable/models/video_resnet.html) — 3D ResNet backbone
