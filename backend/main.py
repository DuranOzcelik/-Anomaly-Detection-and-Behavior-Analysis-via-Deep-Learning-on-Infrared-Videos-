from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
from datetime import datetime

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

@app.get("/")
async def root():
    return {"message": "Kızılötesi Görüntü Anomali Tespiti API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/process-video")
async def process_video(file: UploadFile = File(...)):
    """Video dosyasını işleme kuyruğuna ekle"""
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
        job_id = f"job_{datetime.now().timestamp()}"
        processing_status[job_id] = {
            "status": "İşleniyor",
            "filename": file.filename,
            "created_at": datetime.now().isoformat(),
            "progress": 0
        }

        return {
            "job_id": job_id,
            "message": "Video işlenmeye başladı",
            "filename": file.filename,
            "status": "İşleniyor"
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
