from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import asyncio
import uvicorn
import os
from datetime import datetime
import base64
import io
import numpy as np
from PIL import Image
import json
import logging
import torch
from pathlib import Path

try:
    from services.drive_service import get_drive_service
    DRIVE_AVAILABLE = True
except ImportError:
    DRIVE_AVAILABLE = False
    def get_drive_service():
        return None

from services.video_processor import get_video_processor
from models.CR_AE import ConvolutionalRecurrentAutoencoder
from models.CNN_3D import CNN3D

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s'
)
logger = logging.getLogger(__name__)

cr_ae_model = None
cnn_3d_model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(
    title="Infrared Image Anomaly Detection API",
    description="Kızılötesi görüntü anomali tespiti sistemi",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

processing_status = {}
UPLOAD_DIR = "uploads"
TEMP_VIDEO_DIR = "temp_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)
BASE_DIR = Path(__file__).resolve().parent

# Drive'daki model dosyalarının gerçek path'leri (Drive kök dizinine göre)
CRAE_DRIVE_PATH = "Data_Subset_Autoencoders_Anomaly_Detectors-CRAE/models/crae_winter_finetuned2.pth"
CNN3D_DRIVE_PATH = "Data_Annotated_Subset_Object_Detectors-CNN/training_final/cnn_multilabel_final_model/best_model.pth"

# Lokalde cache'leneceği path'ler (BASE_DIR'e göre)
CRAE_LOCAL_PATH = "model_cache/crae_winter_finetuned2.pth"
CNN3D_LOCAL_PATH = "model_cache/best_model.pth"

# Drive'da izlenecek klasör yolu (kök dizinden itibaren '/' ile ayrılmış)
# Örnek: "Kamera Kayitlari/IR_Videos"
# Bu değeri kendi Drive klasör yolunuzla değiştirin.
DRIVE_WATCH_FOLDER_PATH = os.getenv("DRIVE_WATCH_FOLDER_PATH", "archive/system_video_final_test")


def resolve_archive_path(relative_path: str) -> Path:
    return BASE_DIR / relative_path


def _extract_state_dict(checkpoint) -> dict:
    """
    Checkpoint dict'inden model ağırlıklarını çıkarır.
    Desteklenen key'ler: 'model_state', 'model_state_dict', 'state_dict'
    torch.compile prefix'i (_orig_mod.) varsa temizler.
    """
    sd = checkpoint
    if isinstance(sd, dict):
        for key in ('model_state', 'model_state_dict', 'state_dict'):
            if key in sd:
                sd = sd[key]
                break

    # torch.compile() ile eğitilen modellerde '_orig_mod.' prefix'i gelir
    if isinstance(sd, dict) and any(k.startswith('_orig_mod.') for k in sd):
        sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}

    return sd


def load_model_weights(model, local_rel_path: str, drive_rel_path: str, drive_service=None) -> bool:
    """
    Modeli önce local cache'den yüklemeye çalışır.
    Bulunamazsa Drive'dan indirir ve cache'e kaydeder.
    """
    local_path = resolve_archive_path(local_rel_path)

    if local_path.exists():
        try:
            checkpoint = torch.load(local_path, map_location=device, weights_only=False)
            state_dict = _extract_state_dict(checkpoint)
            result = model.load_state_dict(state_dict, strict=False)
            missing = [k for k in result.missing_keys if 'num_batches_tracked' not in k]
            if missing:
                logger.warning(f"⚠ Eksik key'ler ({len(missing)}): {missing[:5]}...")
            logger.info(f"✓ Model local cache'den yüklendi: {local_path}")
            return True
        except Exception as e:
            logger.warning(f"⚠ Local cache bozuk, Drive'dan yeniden indiriliyor: {e}")
            local_path.unlink(missing_ok=True)

    if drive_service is None:
        logger.warning(f"⚠ '{local_rel_path}' bulunamadı ve Drive servisi kullanılamıyor")
        return False

    try:
        logger.info(f"⏳ Drive'dan indiriliyor: {drive_rel_path}")
        weights_buffer = drive_service.download_file_by_path(drive_rel_path)

        if weights_buffer is None or weights_buffer.getbuffer().nbytes == 0:
            logger.error(f"✗ Drive'dan boş dosya geldi: {drive_rel_path}")
            return False

        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(weights_buffer.getvalue())
        logger.info(f"✓ Drive'dan indirildi ve cache'e kaydedildi: {local_path}")

        weights_buffer.seek(0)
        checkpoint = torch.load(weights_buffer, map_location=device, weights_only=False)
        state_dict = _extract_state_dict(checkpoint)
        result = model.load_state_dict(state_dict, strict=False)
        missing = [k for k in result.missing_keys if 'num_batches_tracked' not in k]
        if missing:
            logger.warning(f"⚠ Eksik key'ler ({len(missing)}): {missing[:5]}...")
        logger.info(f"✓ Model ağırlıkları yüklendi: {drive_rel_path}")
        return True

    except FileNotFoundError as e:
        logger.error(f"✗ Drive'da dosya bulunamadı: {drive_rel_path} — {e}")
        return False
    except Exception as err:
        logger.error(f"✗ Drive'dan model indirilemedi: {drive_rel_path} — {err}", exc_info=True)
        return False


@app.on_event("startup")
async def startup_event():
    global cr_ae_model, cnn_3d_model

    try:
        logger.info("=" * 60)
        logger.info("MODEL YÜKLEMESİ BAŞLIYOR")
        logger.info(f"Cihaz: {device}")
        logger.info("=" * 60)

        drive_service = None
        if DRIVE_AVAILABLE:
            try:
                drive_service = get_drive_service()
                logger.info("✓ Google Drive servisi başlatıldı")
            except Exception as drive_err:
                logger.warning(f"⚠ Google Drive servisi başlatılamadı: {drive_err}")
                drive_service = None
        else:
            logger.warning("⚠ Google Drive kütüphanesi kurulu değil")

        # CR-AE Model
        try:
            logger.info("⏳ CR-AE modeli yükleniyor...")
            cr_ae_model = ConvolutionalRecurrentAutoencoder()
            loaded = load_model_weights(
                cr_ae_model,
                local_rel_path=CRAE_LOCAL_PATH,
                drive_rel_path=CRAE_DRIVE_PATH,
                drive_service=drive_service,
            )
            if not loaded:
                logger.warning("⚠ CR-AE ağırlıkları yüklenemedi → eğitimsiz model kullanılacak")
            cr_ae_model.to(device)
            cr_ae_model.eval()
            logger.info(f"✓ CR-AE modeli hazır ({device})")
        except Exception as model_err:
            logger.error(f"✗ CR-AE modeli yüklenirken hata: {model_err}", exc_info=True)
            cr_ae_model = None

        # CNN 3D Model
        try:
            logger.info("⏳ 3D-CNN modeli yükleniyor...")
            cnn_3d_model = CNN3D(num_classes=4, dropout=0.5)
            loaded = load_model_weights(
                cnn_3d_model,
                local_rel_path=CNN3D_LOCAL_PATH,
                drive_rel_path=CNN3D_DRIVE_PATH,
                drive_service=drive_service,
            )
            if not loaded:
                logger.warning("⚠ 3D-CNN ağırlıkları yüklenemedi → eğitimsiz model kullanılacak")
            cnn_3d_model.to(device)
            cnn_3d_model.eval()
            logger.info(f"✓ 3D-CNN modeli hazır ({device})")
        except Exception as model_err:
            logger.error(f"✗ 3D-CNN modeli yüklenirken hata: {model_err}", exc_info=True)
            cnn_3d_model = None

        if cr_ae_model is not None and cnn_3d_model is not None:
            logger.info("=" * 60)
            logger.info("✓ TÜM MODELLER BAŞARIYLA YÜKLENDİ")
            logger.info("=" * 60)
        else:
            logger.warning("=" * 60)
            logger.warning("⚠ UYARI: Bazı modeller yüklenemedi (eğitimsiz model ile devam ediliyor)")
            logger.warning("=" * 60)

    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"✗ BAŞLANGIÇ HATASI: {e}", exc_info=True)
        logger.error("=" * 60)


def generate_heatmap(width=640, height=480):
    heatmap = np.zeros((height, width), dtype=np.float32)
    for _ in range(3):
        cx = np.random.randint(0, width)
        cy = np.random.randint(0, height)
        radius = np.random.randint(20, 80)
        y, x = np.ogrid[:height, :width]
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
        heatmap[mask] = 255
    if heatmap.max() > 0:
        heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
    else:
        heatmap = heatmap.astype(np.uint8)
    return Image.fromarray(heatmap, mode='L')


def image_to_base64(img):
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


@app.get("/")
async def root():
    return {"message": "Kızılötesi Görüntü Anomali Tespiti API"}


@app.get("/video/{filename:path}")
async def serve_video(filename: str):
    """Geçici olarak cache'lenmiş videoyu HTTP üzerinden servis eder (Flutter video_player için)."""
    safe_name = Path(filename).name
    file_path = Path(TEMP_VIDEO_DIR) / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Video bulunamadı: {safe_name}")
    return FileResponse(
        str(file_path),
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes"},
    )


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "cr_ae_loaded": cr_ae_model is not None,
        "cnn_3d_loaded": cnn_3d_model is not None,
        "drive_available": DRIVE_AVAILABLE,
        "device": str(device)
    }


@app.get("/videos")
async def list_videos(start_date: str, end_date: str):
    """
    Belirtilen tarih aralığındaki Drive videolarını listeler.
    Flutter'daki listVideos() çağrısına karşılık gelen endpoint.
    """
    if not DRIVE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Google Drive kütüphanesi kurulu değil")

    try:
        drive_service = get_drive_service()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Drive bağlantısı kurulamadı: {e}")

    try:
        videos = drive_service.list_videos_by_date_range(start_date, end_date)
        logger.info(f"✓ /videos endpoint: {len(videos)} video listelendi ({start_date} → {end_date})")
        return videos
    except Exception as e:
        logger.error(f"✗ Video listeleme hatası: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/drive/videos")
async def list_drive_folder_videos():
    """DRIVE_WATCH_FOLDER_PATH klasöründeki videoları listeler (Flutter Drive seçim dialogu için)."""
    if not DRIVE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Google Drive kütüphanesi kurulu değil")
    if not DRIVE_WATCH_FOLDER_PATH:
        raise HTTPException(status_code=400, detail="DRIVE_WATCH_FOLDER_PATH ayarlanmamış")
    try:
        drive_service = await asyncio.to_thread(get_drive_service)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Drive bağlantısı kurulamadı: {e}")
    try:
        folder_id = await asyncio.to_thread(
            drive_service.find_folder_by_path, DRIVE_WATCH_FOLDER_PATH
        )
        if not folder_id:
            raise HTTPException(
                status_code=404,
                detail=f'Klasör bulunamadı: "{DRIVE_WATCH_FOLDER_PATH}"'
            )

        VIDEO_MIME = (
            "mimeType='video/mp4' or mimeType='video/x-msvideo' or "
            "mimeType='video/quicktime' or mimeType='video/x-matroska'"
        )

        def _build_video_entry(f: dict, subfolder: str = "") -> dict:
            return {
                "file_id":  f["id"],
                "filename": f["name"],
                "size_mb":  round(int(f.get("size", 0)) / 1024 / 1024, 1),
                "subfolder": subfolder,
            }

        all_videos = []

        # Doğrudan video çocukları
        direct_q = f"'{folder_id}' in parents and trashed=false and ({VIDEO_MIME})"
        direct = await asyncio.to_thread(
            drive_service._list_files, direct_q, 'files(id, name, size, mimeType)', 200
        )
        all_videos.extend(_build_video_entry(f) for f in direct)

        # Alt klasörler → her birindeki videolar
        sf_q = (
            f"'{folder_id}' in parents and trashed=false and "
            "mimeType='application/vnd.google-apps.folder'"
        )
        subfolders = await asyncio.to_thread(
            drive_service._list_files, sf_q, 'files(id, name)', 50
        )
        for sf in subfolders:
            sf_video_q = f"'{sf['id']}' in parents and trashed=false and ({VIDEO_MIME})"
            sf_videos = await asyncio.to_thread(
                drive_service._list_files, sf_video_q, 'files(id, name, size, mimeType)', 200
            )
            all_videos.extend(_build_video_entry(f, sf["name"]) for f in sf_videos)

        return {"folder_path": DRIVE_WATCH_FOLDER_PATH, "videos": all_videos}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Drive video listesi hatası: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_id = id(websocket)
    logger.info(f"📡 WebSocket istemcisi bağlandı [ID: {client_id}]")

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get('action') == 'start_analysis':
                start_date = message.get('start_date')
                end_date = message.get('end_date')
                logger.info(f"🔍 Analiz başlatılıyor [ID: {client_id}]: {start_date} → {end_date}")

                try:
                    if not DRIVE_AVAILABLE:
                        logger.error(f"❌ Google Drive kütüphanesi yüklü değil [ID: {client_id}]")
                        await websocket.send_json({
                            "type": "error",
                            "error_message": "Google Drive kütüphanesi yüklü değil."
                        })
                        continue

                    try:
                        drive_service = get_drive_service()
                        logger.info(f"✓ Drive servisi başlatıldı [ID: {client_id}]")
                    except Exception as drive_err:
                        logger.error(f"❌ Drive doğrulaması başarısız [ID: {client_id}]: {drive_err}", exc_info=True)
                        await websocket.send_json({
                            "type": "error",
                            "error_message": f"Google Drive bağlantısı kurulamadı: {drive_err}"
                        })
                        continue

                    if drive_service is None or drive_service.service is None:
                        logger.error(f"❌ Drive servisi None döndü [ID: {client_id}]")
                        await websocket.send_json({
                            "type": "error",
                            "error_message": "Google Drive servisi başlatılamadı."
                        })
                        continue

                    processor = get_video_processor(cr_ae_model, cnn_3d_model)
                    logger.info(f"✓ Video işlemcisi hazırlandı [ID: {client_id}]")

                    videos = drive_service.list_videos_by_date_range(start_date, end_date)
                    logger.info(f"📹 {len(videos)} video bulundu [ID: {client_id}]")

                    if not videos:
                        logger.warning(f"⚠ Tarih aralığında video bulunamadı [ID: {client_id}]: {start_date} - {end_date}")
                        await websocket.send_json({
                            "type": "error",
                            "error_message": f"{start_date} - {end_date} aralığında video bulunamadı."
                        })
                        continue

                    batch_start_time = datetime.now()

                    for idx, video in enumerate(videos, 1):
                        logger.info(f"▶ Video işleniyor [{idx}/{len(videos)}] [ID: {client_id}]: {video['filename']}")
                        await websocket.send_json({
                            "type": "video_start",
                            "filename": video['filename'],
                            "date_folder": video['date_folder']
                        })

                        try:
                            video_buffer = drive_service.download_video(
                                video['file_id'],
                                video['filename']
                            )
                            logger.info(f"✓ Video indirildi [ID: {client_id}]: {video['filename']}")

                            # Videoyu temp'e kaydet ve Flutter'a URL gönder
                            safe_name = Path(video['filename']).name
                            temp_path = Path(TEMP_VIDEO_DIR) / safe_name
                            video_buffer.seek(0)
                            with open(temp_path, 'wb') as vf:
                                vf.write(video_buffer.read())
                            video_buffer.seek(0)
                            await websocket.send_json({
                                "type": "video_ready",
                                "filename": video['filename'],
                                "video_url": f"http://127.0.0.1:8000/video/{safe_name}",
                            })

                            frame_count = 0
                            async for frame_result in processor.process_video(
                                video_buffer,
                                video['filename']
                            ):
                                if frame_result.get('type') == 'video_summary':
                                    logger.info(f"✓ Video summary gönderiliyor [ID: {client_id}]: {frame_result.get('filename')}")
                                    await websocket.send_json(frame_result)
                                else:
                                    frame_count += 1
                                    if 'type' not in frame_result:
                                        await websocket.send_json({
                                            "type": "frame_processed",
                                            **frame_result
                                        })

                            logger.info(f"✓ Video tamamlandı [ID: {client_id}]: {video['filename']} ({frame_count} sample frame)")
                            await websocket.send_json({
                                "type": "video_complete",
                                "filename": video['filename']
                            })

                        except Exception as e:
                            logger.error(f"❌ Video işleme hatası [ID: {client_id}]: {video['filename']} - {e}", exc_info=True)
                            await websocket.send_json({
                                "type": "error",
                                "filename": video['filename'],
                                "error_message": str(e)
                            })

                    processing_time = int((datetime.now() - batch_start_time).total_seconds())
                    logger.info(f"✓ Batch tamamlandı [ID: {client_id}]: {len(videos)} video işlendi ({processing_time}s)")
                    await websocket.send_json({
                        "type": "batch_complete",
                        "total_videos": len(videos),
                        "processing_time_seconds": processing_time
                    })

                except Exception as e:
                    logger.error(f"❌ WebSocket analiz hatası [ID: {client_id}]: {e}", exc_info=True)
                    await websocket.send_json({
                        "type": "error",
                        "error_message": str(e)
                    })

            elif message.get('action') == 'start_video':
                # Tek video işleme (Flutter'daki startVideoProcessing)
                filename = message.get('filename')
                logger.info(f"🔁 Tek video işleme talebi [ID: {client_id}]: {filename}")

                try:
                    if not DRIVE_AVAILABLE:
                        await websocket.send_json({
                            "type": "error",
                            "error_message": "Google Drive kütüphanesi yüklü değil."
                        })
                        continue

                    drive_service = get_drive_service()
                    processor = get_video_processor(cr_ae_model, cnn_3d_model)

                    # Dosyayı Drive'da ara
                    file_id = drive_service.find_file_by_path(filename)
                    if not file_id:
                        await websocket.send_json({
                            "type": "error",
                            "error_message": f"Video bulunamadı: {filename}"
                        })
                        continue

                    await websocket.send_json({
                        "type": "video_start",
                        "filename": filename
                    })

                    video_buffer = drive_service.download_video(file_id, filename)

                    # Videoyu temp'e kaydet ve Flutter'a URL gönder
                    safe_name = Path(filename).name
                    temp_path = Path(TEMP_VIDEO_DIR) / safe_name
                    video_buffer.seek(0)
                    with open(temp_path, 'wb') as vf:
                        vf.write(video_buffer.read())
                    video_buffer.seek(0)
                    await websocket.send_json({
                        "type": "video_ready",
                        "filename": filename,
                        "video_url": f"http://127.0.0.1:8000/video/{safe_name}",
                    })

                    async for frame_result in processor.process_video(video_buffer, filename):
                        if frame_result.get('type') == 'video_summary':
                            await websocket.send_json(frame_result)
                        else:
                            if 'type' not in frame_result:
                                await websocket.send_json({
                                    "type": "frame_processed",
                                    **frame_result
                                })

                    await websocket.send_json({
                        "type": "video_complete",
                        "filename": filename
                    })

                except Exception as e:
                    logger.error(f"❌ Tek video işleme hatası [ID: {client_id}]: {filename} - {e}", exc_info=True)
                    await websocket.send_json({
                        "type": "error",
                        "error_message": str(e)
                    })

            elif message.get('action') == 'stop':
                logger.info(f"⏹ Analiz durduruldu [ID: {client_id}]")
                await websocket.send_json({"type": "stopped"})

    except WebSocketDisconnect:
        logger.info(f"📡 WebSocket istemcisi bağlantısı kesildi [ID: {client_id}]")
    except Exception as e:
        logger.error(f"❌ WebSocket hatası [ID: {client_id}]: {e}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "error_message": f"Server error: {str(e)}"
            })
        except Exception:
            pass


@app.websocket("/ws/drive")
async def drive_websocket_endpoint(websocket: WebSocket):
    """Google Drive klasörünü izleme ve gerçek zamanlı video işleme WebSocket."""
    await websocket.accept()
    client_id  = id(websocket)
    stop_event = asyncio.Event()
    logger.info(f"☁️ Drive WebSocket bağlandı [ID: {client_id}]")

    if not DRIVE_AVAILABLE:
        await websocket.send_json({
            'type': 'error',
            'error_message': 'Google Drive kütüphanesi kurulu değil. pip install google-api-python-client google-auth'
        })
        return

    if not DRIVE_WATCH_FOLDER_PATH:
        await websocket.send_json({
            'type': 'error',
            'error_message': 'DRIVE_WATCH_FOLDER_PATH ayarlanmamış. main.py içindeki sabiti veya DRIVE_WATCH_FOLDER_PATH env değişkenini ayarlayın.'
        })
        return

    # İlk mesajı bekle (start_drive_watch komutu)
    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
        init_msg = json.loads(raw)
        if init_msg.get('action') != 'start_drive_watch':
            await websocket.send_json({
                'type': 'error',
                'error_message': 'start_drive_watch komutu beklendi'
            })
            return
    except asyncio.TimeoutError:
        await websocket.send_json({
            'type': 'error',
            'error_message': 'start_drive_watch komutu beklendi (timeout)'
        })
        return
    except Exception as exc:
        await websocket.send_json({'type': 'error', 'error_message': str(exc)})
        return

    # Tek video modu için opsiyonel filtreler
    file_id_filter  = init_msg.get('file_id')
    filename_filter = init_msg.get('filename')

    try:
        drive_service = get_drive_service()
    except Exception as e:
        await websocket.send_json({
            'type': 'error',
            'error_message': f'Drive bağlantısı kurulamadı: {e}'
        })
        return

    # Klasör yolunu Drive'da çöz → folder_id bul
    logger.info(f"☁️ Drive klasörü çözülüyor [ID: {client_id}]: {DRIVE_WATCH_FOLDER_PATH}")
    try:
        folder_id = await asyncio.to_thread(
            drive_service.find_folder_by_path, DRIVE_WATCH_FOLDER_PATH
        )
    except Exception as e:
        await websocket.send_json({
            'type': 'error',
            'error_message': f'Drive klasörü bulunamadı: {e}'
        })
        return

    if not folder_id:
        await websocket.send_json({
            'type': 'error',
            'error_message': f'Drive klasörü bulunamadı: "{DRIVE_WATCH_FOLDER_PATH}" — yol doğru mu?'
        })
        return

    logger.info(f"☁️ Drive klasörü bulundu [ID: {client_id}]: {folder_id}")

    from services.drive_watcher import DriveWatcher
    processor = get_video_processor(cr_ae_model, cnn_3d_model)
    watcher   = DriveWatcher(processor, drive_service, folder_id)

    async def _listen_stop():
        try:
            while True:
                raw2 = await websocket.receive_text()
                msg2 = json.loads(raw2)
                if msg2.get('action') == 'stop_drive_watch':
                    logger.info(f"⏹ stop_drive_watch alındı [ID: {client_id}]")
                    stop_event.set()
                    break
        except (WebSocketDisconnect, Exception):
            stop_event.set()

    stop_task = asyncio.create_task(_listen_stop())

    try:
        async for result in watcher.watch(stop_event, file_id=file_id_filter, filename=filename_filter):
            try:
                await websocket.send_json(result)
            except Exception:
                break
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.error(f"Drive WebSocket hatası: {exc}", exc_info=True)
        try:
            await websocket.send_json({'type': 'error', 'error_message': str(exc)})
        except Exception:
            pass
    finally:
        stop_event.set()
        stop_task.cancel()
        logger.info(f"☁️ Drive WebSocket kapandı [ID: {client_id}]")


@app.websocket("/ws/camera")
async def camera_websocket_endpoint(websocket: WebSocket):
    """Gerçek zamanlı IR kamera akışı ve anomali analizi WebSocket."""
    await websocket.accept()
    client_id  = id(websocket)
    stop_event = asyncio.Event()
    logger.info(f"📸 Kamera WebSocket bağlandı [ID: {client_id}]")

    from services.camera_processor import CameraProcessor

    processor    = get_video_processor(cr_ae_model, cnn_3d_model)
    camera_proc  = CameraProcessor(processor)

    # İlk mesajı bekle (start_camera komutu)
    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
        init_msg = json.loads(raw)
        if init_msg.get('action') == 'start_camera':
            camera_proc.camera_index = int(init_msg.get('camera_index', 0))
            logger.info(f"📸 Kamera {camera_proc.camera_index} başlatılıyor [ID: {client_id}]")
    except asyncio.TimeoutError:
        await websocket.send_json({'type': 'error', 'error_message': 'start_camera komutu beklendi (timeout)'})
        return
    except Exception as exc:
        await websocket.send_json({'type': 'error', 'error_message': str(exc)})
        return

    # Arka planda stop komutunu dinle
    async def _listen_stop():
        try:
            while True:
                raw2 = await websocket.receive_text()
                msg2 = json.loads(raw2)
                if msg2.get('action') == 'stop_camera':
                    logger.info(f"⏹ stop_camera alındı [ID: {client_id}]")
                    stop_event.set()
                    break
        except (WebSocketDisconnect, Exception):
            stop_event.set()

    stop_task = asyncio.create_task(_listen_stop())

    try:
        async for result in camera_proc.stream(stop_event):
            try:
                await websocket.send_json(result)
            except Exception:
                break
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.error(f"Kamera WebSocket hatası: {exc}", exc_info=True)
        try:
            await websocket.send_json({'type': 'error', 'error_message': str(exc)})
        except Exception:
            pass
    finally:
        stop_event.set()
        stop_task.cancel()
        logger.info(f"📸 Kamera WebSocket kapandı [ID: {client_id}]")


@app.post("/process-video")
async def process_video(file: UploadFile = File(...)):
    logger.info(f"📤 Video yükleniyor: {file.filename}")

    try:
        if not file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            logger.error(f"❌ Desteklenmeyen format: {file.filename}")
            raise HTTPException(status_code=400, detail="Desteklenen format: MP4, AVI, MOV, MKV")

        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            contents = await file.read()
            buffer.write(contents)
        logger.info(f"✓ Video kaydedildi: {file_path}")

        job_id = f"job_{int(datetime.now().timestamp() * 1000)}"
        processing_status[job_id] = {
            "status": "Tamamlandı",
            "filename": file.filename,
            "created_at": datetime.now().isoformat(),
            "progress": 100,
            "classification": "Normal",
            "confidence": 0.87
        }
        logger.info(f"✓ İşlem başlatıldı [Job ID: {job_id}]: {file.filename}")

        heatmap_img = generate_heatmap(640, 480)
        heatmap_base64 = image_to_base64(heatmap_img)
        logger.info(f"✓ Heatmap oluşturuldu [Job ID: {job_id}]")

        return {
            "job_id": job_id,
            "message": "Video işlendi",
            "filename": file.filename,
            "status": "Tamamlandı",
            "classification": "Normal",
            "confidence": 0.87,
            "latency_ms": 1240,
            "heatmap": heatmap_base64
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Video işleme başarısız: {file.filename} - {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/results/{job_id}")
async def get_results(job_id: str):
    if job_id not in processing_status:
        raise HTTPException(status_code=404, detail="İşlem bulunamadı")
    job_info = processing_status[job_id]
    return {
        "job_id": job_id,
        "status": job_info["status"],
        "filename": job_info["filename"],
        "created_at": job_info["created_at"],
        "progress": job_info["progress"],
        "results": None
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)