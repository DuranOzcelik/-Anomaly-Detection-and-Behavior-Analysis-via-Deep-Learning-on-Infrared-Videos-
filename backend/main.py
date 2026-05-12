from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
from datetime import datetime
import base64
import io
import numpy as np
from PIL import Image

app = FastAPI(
    title="Infrared Image Anomaly Detection API",
    description="Kızılötesi görüntü anomali tespiti sistemi",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for processing status
processing_status = {}
UPLOAD_DIR = "uploads"

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

def generate_heatmap(width=640, height=480):
    """Mock heatmap oluştur (model integration placeholder)"""
    # Anomali bölgelerini simüle et
    heatmap = np.zeros((height, width), dtype=np.float32)

    # Rastgele anomali bölgeleri ekle
    for _ in range(3):
        cx = np.random.randint(0, width)
        cy = np.random.randint(0, height)
        radius = np.random.randint(20, 80)

        y, x = np.ogrid[:height, :width]
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
        heatmap[mask] = 255

    # Min-max normalization
    if heatmap.max() > 0:
        heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
    else:
        heatmap = heatmap.astype(np.uint8)

    img = Image.fromarray(heatmap, mode='L')
    return img

def image_to_base64(img):
    """Pillow Image'ı base64'e çevir"""
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

@app.get("/")
async def root():
    return {"message": "Kızılötesi Görüntü Anomali Tespiti API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/process-video")
async def process_video(file: UploadFile = File(...)):
    """Video dosyasını işleme kuyruğuna ekle ve heatmap döndür"""
    try:
        # Dosya adını kontrol et
        if not file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            raise HTTPException(status_code=400, detail="Desteklenen format: MP4, AVI, MOV, MKV")

        # Dosyayı kaydet
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            contents = await file.read()
            buffer.write(contents)

        # İşlem durumunu kaydet
        job_id = f"job_{int(datetime.now().timestamp() * 1000)}"
        processing_status[job_id] = {
            "status": "Tamamlandı",
            "filename": file.filename,
            "created_at": datetime.now().isoformat(),
            "progress": 100,
            "classification": "Normal",
            "confidence": 0.87
        }

        # Mock heatmap oluştur
        heatmap_img = generate_heatmap(640, 480)
        heatmap_base64 = image_to_base64(heatmap_img)

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

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results/{job_id}")
async def get_results(job_id: str):
    """İşlem sonuçlarını getir"""
    if job_id not in processing_status:
        raise HTTPException(status_code=404, detail="İşlem bulunamadı")

    job_info = processing_status[job_id]
    return {
        "job_id": job_id,
        "status": job_info["status"],
        "filename": job_info["filename"],
        "created_at": job_info["created_at"],
        "progress": job_info["progress"],
        "results": None  # Sonuçlar burada eklenecek
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
