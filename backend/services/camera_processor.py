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
    """Gerçek zamanlı IR kamera akışı: CLAHE + COLORMAP_INFERNO + 16-frame kayan pencere analizi."""

    CLIP_LENGTH    = 16
    CLIP_STEP      = 16   # Canlı modda örtüşme yok (her klip bağımsız)
    DISPLAY_EVERY  = 2    # Her 2 frame'de bir gönder (~12 fps efektif)

    def __init__(self, video_processor, camera_index: int = 0):
        self.vp            = video_processor
        self.camera_index  = camera_index

    # ── IR görsel dönüşümü ────────────────────────────────────────────────────

    def apply_ir_effect(self, frame: np.ndarray) -> np.ndarray:
        """
        Normal webcam frame'ini termal IR kamera görüntüsüne dönüştürür.

        Gerçek FLIR / termal kameralar:
          - Sıcak nesneler  → beyaz / sarı / turuncu
          - Soğuk nesneler  → siyah / mor / koyu kırmızı
          - Hafif lens bulanıklığı ve sensör gürültüsü
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # CLAHE: gerçek IR sensörlerinde sıcaklık gradyanlarını öne çıkarır
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Histogram germe: düşük-yüksek aralığını tam skalaya yay
        p_low, p_high = np.percentile(enhanced, [2, 98])
        if p_high > p_low:
            stretched = np.clip(
                (enhanced.astype(np.float32) - p_low) / (p_high - p_low) * 255,
                0, 255).astype(np.uint8)
        else:
            stretched = enhanced

        # IR lens hafif blur (IR lensler optik olarak biraz yumuşaktır)
        blurred = cv2.GaussianBlur(stretched, (5, 5), 1.2)

        # COLORMAP_INFERNO: en gerçekçi termal IR paleti
        # Siyah → Mor → Kırmızı → Turuncu → Sarı → Beyaz
        thermal = cv2.applyColorMap(blurred, cv2.COLORMAP_INFERNO)

        # Hafif sensör gürültüsü (gerçek IR sensörlerde doğal)
        noise = np.random.normal(0, 3, thermal.shape).astype(np.int16)
        thermal = np.clip(thermal.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return thermal

    def _to_b64(self, frame: np.ndarray, quality: int = 72) -> str:
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

    # ── Ana akış ─────────────────────────────────────────────────────────────

    async def stream(self, stop_event: asyncio.Event) -> AsyncGenerator:
        """
        Kameradan canlı IR görüntü akışı.
        Her CLIP_LENGTH frame'de CR-AE + CNN analizi yapar ve sonuçları yield eder.
        """
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
                    f'Kamera açılamadı (index={self.camera_index}). '
                    'USB/dahili kameranızın bağlı ve başka program tarafından '
                    'kullanılmıyor olduğundan emin olun.'
                )
            }
            return

        yield {
            'type': 'camera_connected',
            'camera_id': str(self.camera_index),
            'message': f'Kamera {self.camera_index} bağlantısı kuruldu',
        }

        frame_buffer: list  = []   # Analiz için RGB frame'ler
        frame_num:    int   = 0
        clip_idx:     int   = 0
        session = {
            'anomaly_scores': [],
            'classifications': [],
            'anomaly_clips': 0,
        }

        try:
            while not stop_event.is_set():
                ret, raw_frame = await asyncio.to_thread(cap.read)
                if not ret:
                    logger.warning("Kamera frame okunamadı, bekleniyor...")
                    await asyncio.sleep(0.05)
                    continue

                # IR dönüşümü (görsel akış için)
                ir_frame = self.apply_ir_effect(raw_frame)

                # Analiz buffer'ı için RGB sakla
                rgb_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
                frame_buffer.append(rgb_frame)

                # Frame'i belirli aralıklarla gönder (bant genişliği tasarrufu)
                if frame_num % self.DISPLAY_EVERY == 0:
                    yield {
                        'type': 'camera_frame',
                        'frame_base64': self._to_b64(ir_frame),
                        'frame_number': frame_num,
                    }

                # Her CLIP_LENGTH frame'de analiz
                if len(frame_buffer) >= self.CLIP_LENGTH:
                    clip = frame_buffer[:self.CLIP_LENGTH]
                    frame_buffer = []   # buffer'ı sıfırla

                    try:
                        t0 = datetime.now()

                        # CR-AE aşaması
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

                        # 3D-CNN yalnızca eşik aşıldığında çalışır
                        if threshold_exceeded:
                            cnn_t0 = datetime.now()
                            cnn_r  = await asyncio.to_thread(self.vp._infer_cnn3d, clip)
                            cnn_ms = int((datetime.now() - cnn_t0).total_seconds() * 1000)
                            all_class_probs = cnn_r.get('all_class_probs', {})
                            top2_classes    = cnn_r.get('top2_classes', [])
                            classification  = cnn_r['classification']
                            confidence      = cnn_r['confidence']

                        # GradCAM heatmap
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
                        logger.error(f"Kamera klip analizi hatası: {exc}", exc_info=True)

                await asyncio.sleep(0.001)   # event loop'a nefes aldır

        finally:
            await asyncio.to_thread(cap.release)
            logger.info("Kamera serbest bırakıldı")

        # Oturum özeti (sadece en az 1 klip işlendiyse)
        if session['anomaly_scores']:
            scores  = session['anomaly_scores']
            classes = session['classifications']
            dist    = dict(Counter(classes))
            dominant = Counter(classes).most_common(1)[0][0]

            yield {
                'type':             'camera_summary',
                'total_clips':       clip_idx,
                'anomaly_clips':     session['anomaly_clips'],
                'avg_anomaly_score': float(np.mean(scores)),
                'dominant_class':    dominant,
                'class_distribution': dist,
            }

        yield {'type': 'camera_stopped'}
