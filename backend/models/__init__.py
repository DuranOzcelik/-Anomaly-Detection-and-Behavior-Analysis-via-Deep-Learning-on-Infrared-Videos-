from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from .CR_AE import CR_AE
from .CNN_3D import CNN_3D

# Pydantic schemas
class ImageUpload(BaseModel):
    filename: str
    data: str  # base64 encoded image

class AnomalyResult(BaseModel):
    image_id: str
    anomaly_detected: bool
    confidence: float
    timestamp: datetime
    anomaly_regions: Optional[list] = None

class VideoProcessingRequest(BaseModel):
    job_id: str
    filename: str

class ClassificationResult(BaseModel):
    job_id: str
    class_name: str
    confidence: float
    timestamp: datetime

class ReconstructionErrorResult(BaseModel):
    job_id: str
    error: float
    is_anomaly: bool
    threshold: float
    timestamp: datetime

__all__ = ['CR_AE', 'CNN_3D', 'ImageUpload', 'AnomalyResult', 'VideoProcessingRequest', 'ClassificationResult', 'ReconstructionErrorResult']
