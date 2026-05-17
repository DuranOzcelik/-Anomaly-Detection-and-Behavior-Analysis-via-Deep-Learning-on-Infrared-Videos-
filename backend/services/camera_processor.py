import cv2
import numpy as np
import asyncio
import base64
import logging
from typing import AsyncGenerator
from datetime import datetime
from collections import Counter

from services.video_processor import CRAE_THRESHOLD_MSE

logger = logging.getLogger(__name__)


class CameraProcessor:
    """Live IR camera stream: CLAHE enhancement + 16-frame sliding window analysis."""

    CLIP_LENGTH   = 16
    CLIP_STEP     = 16   # no overlap in live mode
    DISPLAY_EVERY = 2    # send every 2nd frame (~12 fps effective)

    def __init__(self, video_processor, camera_index: int = 0):
        self.vp           = video_processor
        self.camera_index = camera_index

    def apply_ir_effect(self, frame: np.ndarray) -> np.ndarray:
        """Simulate thermal IR appearance: CLAHE + histogram stretch + COLORMAP_INFERNO."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        p_low, p_high = np.percentile(enhanced, [2, 98])
        if p_high > p_low:
            stretched = np.clip(
                (enhanced.astype(np.float32) - p_low) / (p_high - p_low) * 255,
                0, 255).astype(np.uint8)
        else:
            stretched = enhanced

        blurred = cv2.GaussianBlur(stretched, (5, 5), 1.2)
        thermal = cv2.applyColorMap(blurred, cv2.COLORMAP_INFERNO)

        noise = np.random.normal(0, 3, thermal.shape).astype(np.int16)
        thermal = np.clip(thermal.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return thermal

    def _to_b64(self, frame: np.ndarray, quality: int = 72) -> str:
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

    async def stream(self, stop_event: asyncio.Event) -> AsyncGenerator:
        def _open_camera():
            cap = cv2.VideoCapture(self.camera_index)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 25)
            return cap

        cap = await asyncio.to_thread(_open_camera)

        if not cap.isOpened():
            yield {
                'type': 'error',
                'error_message': (
                    f'Camera {self.camera_index} could not be opened. '
                    'Make sure the camera is connected and not in use by another application.'
                )
            }
            return

        yield {
            'type': 'camera_connected',
            'camera_id': str(self.camera_index),
            'message': f'Camera {self.camera_index} connected',
        }

        frame_buffer: list = []
        frame_num:    int  = 0
        clip_idx:     int  = 0
        session = {
            'anomaly_scores': [],
            'classifications': [],
            'anomaly_clips': 0,
        }

        try:
            while not stop_event.is_set():
                ret, raw_frame = await asyncio.to_thread(cap.read)
                if not ret:
                    logger.warning("Camera frame read failed, retrying...")
                    await asyncio.sleep(0.05)
                    continue

                ir_frame = self.apply_ir_effect(raw_frame)
                rgb_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
                frame_buffer.append(rgb_frame)

                if frame_num % self.DISPLAY_EVERY == 0:
                    yield {
                        'type': 'camera_frame',
                        'frame_base64': self._to_b64(ir_frame),
                        'frame_number': frame_num,
                    }

                if len(frame_buffer) >= self.CLIP_LENGTH:
                    clip = frame_buffer[:self.CLIP_LENGTH]
                    frame_buffer = []

                    try:
                        t0 = datetime.now()

                        crae_t0 = datetime.now()
                        anomaly_score, raw_mse, recon_b64, orig_b64, pixel_mse_map = \
                            await asyncio.to_thread(self.vp._infer_crae, clip)
                        crae_ms = int((datetime.now() - crae_t0).total_seconds() * 1000)

                        threshold_exceeded = raw_mse > CRAE_THRESHOLD_MSE

                        all_class_probs = {}
                        top2_classes    = []
                        cnn_ms          = 0
                        classification  = 'Normal'
                        confidence      = 1.0

                        if threshold_exceeded:
                            cnn_t0 = datetime.now()
                            cnn_r  = await asyncio.to_thread(self.vp._infer_cnn3d, clip)
                            cnn_ms = int((datetime.now() - cnn_t0).total_seconds() * 1000)
                            all_class_probs = cnn_r.get('all_class_probs', {})
                            top2_classes    = cnn_r.get('top2_classes', [])
                            classification  = cnn_r['classification']
                            confidence      = cnn_r['confidence']

                        mid_frame = clip[len(clip) // 2]
                        hm_t0     = datetime.now()
                        heatmap_b64 = self.vp._generate_heatmap(
                            mid_frame, anomaly_score, pixel_mse_map)
                        hm_ms = int((datetime.now() - hm_t0).total_seconds() * 1000)

                        latency_ms = int((datetime.now() - t0).total_seconds() * 1000)

                        session['anomaly_scores'].append(anomaly_score)
                        session['classifications'].append(classification)
                        if threshold_exceeded:
                            session['anomaly_clips'] += 1

                        yield {
                            'type': 'clip_result',
                            'frame_number':        frame_num,
                            'clip_index':          clip_idx,
                            'anomaly_score':       anomaly_score,
                            'raw_mse':             raw_mse,
                            'threshold_exceeded':  threshold_exceeded,
                            'classification':      classification,
                            'confidence':          confidence,
                            'heatmap_base64':      heatmap_b64,
                            'latency_ms':          latency_ms,
                            'crae_time_ms':        crae_ms,
                            'cnn_time_ms':         cnn_ms,
                            'heatmap_time_ms':     hm_ms,
                            'all_class_probs':     all_class_probs,
                            'top2_classes':        top2_classes,
                            'recon_frame_base64':  recon_b64,
                            'orig_frame_base64':   orig_b64,
                        }

                        clip_idx += 1
                    except Exception as exc:
                        logger.error(f"Camera clip analysis error: {exc}", exc_info=True)

                await asyncio.sleep(0.001)

        finally:
            await asyncio.to_thread(cap.release)
            logger.info("Camera released")

        if session['anomaly_scores']:
            scores  = session['anomaly_scores']
            classes = session['classifications']
            dist    = dict(Counter(classes))
            dominant = Counter(classes).most_common(1)[0][0]

            yield {
                'type':              'camera_summary',
                'total_clips':        clip_idx,
                'anomaly_clips':      session['anomaly_clips'],
                'avg_anomaly_score':  float(np.mean(scores)),
                'dominant_class':     dominant,
                'class_distribution': dist,
            }

        yield {'type': 'camera_stopped'}
