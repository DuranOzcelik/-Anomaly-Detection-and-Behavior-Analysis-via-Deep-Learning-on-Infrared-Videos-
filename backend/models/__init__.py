from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class ImageUpload(BaseModel):
    filename: str
    data: str  # base64 encoded image

class AnomalyResult(BaseModel):
    image_id: str
    anomaly_detected: bool
    confidence: float
    timestamp: datetime
    anomaly_regions: Optional[list] = None
