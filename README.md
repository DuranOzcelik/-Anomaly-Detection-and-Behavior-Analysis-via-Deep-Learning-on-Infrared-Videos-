# Anomaly Detection and Behavior Analysis via Deep Learning on Infrared Videos

A Windows desktop application for real-time infrared surveillance video analysis. The system detects anomalous events and classifies behavior types using a cascaded deep learning pipeline, visualizing results through heatmap overlays streamed live to the UI.

---

## Overview

Traditional surveillance systems rely on visible-light cameras and rule-based motion detection, which fail in low-light conditions and produce high false-positive rates. Infrared (IR) cameras capture thermal signatures independent of lighting, but analyzing IR footage at scale requires automated tools that go beyond simple motion thresholds.

This project addresses that gap by combining:

- **Unsupervised anomaly detection** — a Convolutional Recurrent Autoencoder (CR-AE) trained only on normal IR footage learns what "normal" looks like. At inference time, clips that reconstruct poorly (high MSE) are flagged as anomalous.
- **Supervised behavior classification** — a 3D-CNN then classifies the detected anomalous clips into specific behavior categories.
- **Explainability** — per-pixel reconstruction error maps are overlaid on the original frame as a heatmap, making the model's decision spatially interpretable.

The result is a fully automated, end-to-end system that processes IR video from Google Drive or a live camera, streams results in real time, and presents them in a professional desktop UI.

---

## Technical Approach

### Dataset

The system was trained and evaluated on the **LTD (Long-Term Dataset)** — a publicly available infrared surveillance video dataset containing labeled clips of four behavior categories:

| Class | Description |
|-------|-------------|
| Normal | Routine pedestrian or vehicle activity |
| Loitering | Individual staying in an area for an unusually long time |
| Trespassing | Entry into a restricted or forbidden zone |
| Object Abandonment | An item left unattended in the scene |

### Pipeline Architecture

```
Video Input
    │
    ▼
Frame Extraction (every 4th frame, ~7 fps effective)
    │
    ▼
16-frame sliding window clips (step = 8 frames)
    │
    ├──► CR-AE Reconstruction
    │         │
    │         ├── MSE ≤ 0.012 ──► Normal (CNN skipped)
    │         │
    │         └── MSE > 0.012 ──► 3D-CNN Classification
    │                                   │
    │                                   └── Normal / Loitering / Trespassing / Obj. Aband.
    │
    └──► Per-pixel MSE map ──► JET colormap heatmap ──► UI overlay
```

### CR-AE (Convolutional Recurrent Autoencoder)

- **Input:** `(B, T, 1, 128, 128)` — batch of grayscale 16-frame clips
- **Encoder:** 4× stride-2 Conv2D layers (32→64→128→256 filters) with BatchNorm + LeakyReLU, compressing each frame to a 256×8×8 spatial embedding
- **Temporal modeling:** Linear projection → 2-layer LSTM (hidden=256) → linear unprojection
- **Decoder:** 4× ConvTranspose2D mirroring the encoder, output via Sigmoid
- **Training:** Reconstruction loss (MSE) on normal-only clips. Anomaly score = normalized MSE relative to a recall-90 threshold derived from validation data.
- **Threshold:** MSE > 0.012 triggers the CNN stage (tuned for 90% recall on the validation set)

### 3D-CNN (Behavior Classifier)

- **Backbone:** `torchvision.models.video.r3d_18` (3D ResNet-18) pretrained weights discarded, trained from scratch on the LTD dataset
- **Input:** `(B, 3, 16, 112, 112)` — RGB 16-frame clips
- **Head:** `Dropout(0.5) → Linear(512, 4)`
- **Output:** 4-class softmax probabilities
- **Training:** Cross-entropy loss, multi-label annotation support

### Cascade Design

Running the 3D-CNN on every clip is computationally expensive. The CR-AE acts as a fast pre-filter: only clips flagged as anomalous reach the CNN stage. This reduces CNN inference calls significantly while maintaining overall recall at 90%+.

### Heatmap Visualization

The per-pixel reconstruction error map `(H, W)` is:
1. Resized to match the original frame resolution
2. Smoothed with Gaussian blur (kernel 21×21)
3. Mapped to `COLORMAP_JET` (blue = low error, green/yellow = mid, red = high)
4. Alpha-blended onto the original frame (low-activation regions suppressed to avoid visual noise)

---

## System Architecture

```
┌─────────────────────────────┐     WebSocket     ┌────────────────────────────┐
│     Flutter Desktop App     │ ◄────────────────► │     FastAPI Backend        │
│                             │                    │                            │
│  • Drive folder browser     │                    │  • /ws          (batch)    │
│  • Live video panel         │                    │  • /ws/drive    (folder)   │
│  • Heatmap overlay          │                    │  • /ws/camera   (live)     │
│  • MSE timeline chart       │                    │  • /health                 │
│  • Class distribution       │                    │  • /video/{name}           │
│  • Per-clip latency stats   │                    │                            │
└─────────────────────────────┘                    │  CR-AE + 3D-CNN models     │
                                                   │  Google Drive API          │
                                                   └────────────────────────────┘
```

**Backend:** FastAPI + uvicorn, async WebSocket streaming, `torch.inference_mode()` for inference, Google Drive API v3 with service account authentication.

**Frontend:** Flutter (Windows), Material 3 light theme, real-time WebSocket message handling, MSE chart painter, heatmap base64 image overlay.

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
4. Rename the key file to `google_drive_credentials.json` and place it inside `backend/`.
5. Share your Drive folder with the service account email (Viewer access is sufficient).

> **Important:** `google_drive_credentials.json` is excluded from git via `.gitignore`. Share it with teammates directly — never commit it to version control.

### 4. Model weights

Weights are downloaded automatically from Google Drive on first startup and cached in `backend/model_cache/`. Make sure the Drive paths in `main.py` match your folder structure:

```python
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

**Backend** (terminal 1):
```bash
cd backend
venv\Scripts\activate
uvicorn main:app --host 127.0.0.1 --port 8000
```

**Frontend** (terminal 2):
```bash
cd frontend
flutter run -d windows
```

The backend must be running before the Flutter app connects.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_DRIVE_CREDENTIALS_PATH` | `google_drive_credentials.json` | Path to the service account JSON key |
| `DRIVE_WATCH_FOLDER_PATH` | `archive/system_video_final_test` | Drive folder to watch for videos |

---

## Project Structure

```
ThesisApp/
├── backend/
│   ├── main.py                  # FastAPI app — all endpoints and WebSocket handlers
│   ├── requirements.txt
│   ├── .env                     # Not committed — create manually
│   ├── google_drive_credentials.json  # Not committed — obtain from project owner
│   ├── models/
│   │   ├── CR_AE.py             # CR-AE architecture (must match training config)
│   │   └── CNN_3D.py            # 3D-CNN classifier (r3d_18 backbone)
│   └── services/
│       ├── video_processor.py   # Frame extraction, inference loop, heatmap generation
│       ├── drive_service.py     # Google Drive API v3 wrapper
│       ├── drive_watcher.py     # Drive folder polling async generator
│       └── camera_processor.py # Live webcam stream with IR effect simulation
└── frontend/
    └── lib/
        ├── main.dart                  # Full UI — panels, charts, dialogs, state
        ├── services/
        │   ├── api_service.dart       # REST HTTP client
        │   └── websocket_service.dart # WebSocket client + ProcessingMessage model
        └── widgets/
            └── video_heatmap_overlay.dart  # Heatmap opacity overlay widget
```

---

## Security

- `google_drive_credentials.json` and `.env` are in `.gitignore` — they are never committed.
- The service account uses `drive.readonly` scope — it cannot write, modify, or delete any Drive files.
- CORS is restricted to `http://127.0.0.1:8000` and `http://localhost:8000` — the backend is not designed to be internet-facing.
- Model weights and temporary video files are excluded from git (`model_cache/`, `temp_videos/`).
- No user credentials, tokens, or API keys of any kind are hardcoded anywhere in the source code.

---

## Key Challenges

**Cascade efficiency:** Running a 3D-CNN on every 16-frame clip at 30 fps is not feasible in real time. The CR-AE pre-filter reduces CNN calls to only those clips that exceed the anomaly threshold, making the pipeline practical on consumer hardware.

**State dict compatibility:** Model weights trained with `torch.compile()` include an `_orig_mod.` prefix in all keys. The loader strips this prefix automatically so checkpoints from compiled and non-compiled training runs are both usable.

**Async streaming:** FastAPI's WebSocket endpoints use `asyncio.to_thread` for all CPU-bound inference calls, keeping the event loop unblocked and allowing the UI to receive frames while inference runs concurrently.

**Drive pagination:** The Google Drive API returns at most 1000 results per page. The `_list_files` wrapper handles pagination transparently and also searches `sharedWithMe` as a fallback, supporting both personal and shared drive configurations.

---

## Acknowledgements

- [LTD Dataset](https://github.com/LTD-Dataset) — infrared surveillance video dataset used for training and evaluation
- [torchvision r3d_18](https://pytorch.org/vision/stable/models/video_resnet.html) — 3D ResNet backbone for the behavior classifier
