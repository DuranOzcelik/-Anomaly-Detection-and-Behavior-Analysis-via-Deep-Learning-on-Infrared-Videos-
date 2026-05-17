import cv2
import numpy as np
import torch
import asyncio
from typing import AsyncGenerator, List, Dict, Optional
from io import BytesIO
import logging
from datetime import datetime
import base64
from collections import Counter

logger = logging.getLogger(__name__)

CLIP_LENGTH = 16    # Eğitimde kullanılan klip uzunluğu (frame)
CLIP_STEP   = 8     # Sliding window adımı (frame) — pipeline diyagramındaki değer
CRAE_SIZE   = 128   # CR-AE giriş boyutu (eğitimde kullanılan)
CNN3D_SIZE  = 112   # 3D-CNN giriş boyutu (eğitimde kullanılan)
FPS_SAMPLE  = 4     # Her 4 frame'den birini al (~7-8fps efektif)

# CR-AE cascade eşiği (raw MSE).
# crae_results_v3.json'daki recall-90 threshold değeriyle güncelle.
# Örnek: normal_mse_mean=0.002 ise threshold ~0.004-0.006 civarı olur.
CRAE_THRESHOLD_MSE = 0.012


class VideoProcessor:
    def __init__(self, cr_ae_model=None, cnn_3d_model=None, gradcam_fn=None):
        self.cr_ae_model = cr_ae_model
        self.cnn_3d_model = cnn_3d_model
        self.gradcam_fn = gradcam_fn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def extract_frames(self, video_buffer: BytesIO, fps_downsample: int = FPS_SAMPLE) -> List[np.ndarray]:
        """Her fps_downsample'ıncı frame'i alır. 2dk video @30fps → ~450 frame."""
        import tempfile, os
        video_buffer.seek(0)
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp.write(video_buffer.read())
            tmp_path = tmp.name

        frames = []
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            logger.error(f"❌ Video açılamadı: {tmp_path}")
            os.remove(tmp_path)
            return []

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % fps_downsample == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            frame_count += 1

        cap.release()
        os.remove(tmp_path)
        logger.info(f"✓ {len(frames)} frame çıkarıldı (toplam={frame_count}, downsample={fps_downsample}x)")
        return frames

    def _build_clips(self, frames: List[np.ndarray]) -> List[List[np.ndarray]]:
        """
        16-frame sliding window clips oluşturur (step=8).
        En az 16 frame yoksa eldekilerle tek klip döner.
        """
        if len(frames) < CLIP_LENGTH:
            return [frames]  # çok kısa video, hepsini ver

        clips = []
        for start in range(0, len(frames) - CLIP_LENGTH + 1, CLIP_STEP):
            clips.append(frames[start:start + CLIP_LENGTH])
        return clips

    def _frames_to_crae_tensor(self, clip: List[np.ndarray]) -> torch.Tensor:
        """
        CR-AE giriş tensörü: (1, seq_len, 1, H, W) — grayscale, float32, [0,1]
        Eğitimde grayscale kullanıldı (unsqueeze(1) ile tek kanal).
        """
        tensors = []
        for f in clip:
            resized = cv2.resize(f, (CRAE_SIZE, CRAE_SIZE))
            gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)   # (H, W)
            t = torch.from_numpy(gray.astype(np.float32) / 255.0).unsqueeze(0)  # (1, H, W)
            tensors.append(t)

        # Pad: eğer klip CLIP_LENGTH'ten kısaysa başa tekrar et
        while len(tensors) < CLIP_LENGTH:
            tensors.insert(0, tensors[0].clone())
        tensors = tensors[:CLIP_LENGTH]

        seq = torch.stack(tensors, dim=0).unsqueeze(0).to(self.device)  # (1, seq, C, H, W)
        return seq

    def _frames_to_cnn3d_tensor(self, clip: List[np.ndarray]) -> torch.Tensor:
        """
        3D-CNN giriş tensörü: (1, C, depth, H, W) — float32, [0,1]
        """
        tensors = []
        for f in clip:
            resized = cv2.resize(f, (CNN3D_SIZE, CNN3D_SIZE))
            t = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1)
            tensors.append(t)

        while len(tensors) < CLIP_LENGTH:
            tensors.insert(0, tensors[0].clone())
        tensors = tensors[:CLIP_LENGTH]

        stacked = torch.stack(tensors, dim=1)   # (C, depth, H, W)
        return stacked.unsqueeze(0).to(self.device)  # (1, C, depth, H, W)

    def _infer_crae(self, clip: List[np.ndarray]) -> tuple:
        """CR-AE reconstruction error → (anomaly_score [0,1], raw_mse, recon_b64, orig_b64, pixel_mse_map)."""
        if not self.cr_ae_model:
            return 0.0, 0.0, None, None, None
        try:
            seq = self._frames_to_crae_tensor(clip)
            if hasattr(self.cr_ae_model, 'sequence_length'):
                self.cr_ae_model.sequence_length = seq.shape[1]

            with torch.inference_mode():
                reconstructed = self.cr_ae_model.forward(seq)

            raw_mse = torch.mean((seq - reconstructed) ** 2).item()
            # Eşikte (0.012) → %50, 2× eşikte → %100 olacak şekilde ölçekle
            anomaly_score = min(raw_mse / (CRAE_THRESHOLD_MSE * 2.0), 1.0)

            # Per-pixel MSE map averaged over batch and time → (H, W)
            pixel_mse_map = torch.mean((seq - reconstructed) ** 2, dim=[0, 1, 2]).cpu().numpy()

            orig_b64 = None
            recon_b64 = None
            try:
                mid_idx = seq.shape[1] // 2
                orig_arr = seq[0, mid_idx, 0].cpu().numpy()
                recon_arr = reconstructed[0, mid_idx, 0].cpu().numpy()
                orig_img = (orig_arr * 255).clip(0, 255).astype(np.uint8)
                recon_img = (recon_arr * 255).clip(0, 255).astype(np.uint8)
                _, ob = cv2.imencode('.jpg', orig_img, [cv2.IMWRITE_JPEG_QUALITY, 60])
                _, rb = cv2.imencode('.jpg', recon_img, [cv2.IMWRITE_JPEG_QUALITY, 60])
                orig_b64 = "data:image/jpeg;base64," + base64.b64encode(ob.tobytes()).decode()
                recon_b64 = "data:image/jpeg;base64," + base64.b64encode(rb.tobytes()).decode()
            except Exception:
                pass

            return anomaly_score, raw_mse, recon_b64, orig_b64, pixel_mse_map
        except Exception as e:
            logger.error(f"❌ CR-AE inference hatası: {e}", exc_info=True)
            return 0.0, 0.0, None, None, None

    def _infer_cnn3d(self, clip: List[np.ndarray]) -> Dict:
        """3D-CNN sınıflandırma → {classification, confidence, all_class_probs, top2_classes}."""
        class_names = ['Normal', 'Loitering', 'Trespass', 'Obj. Aband.']
        if not self.cnn_3d_model:
            default_probs = {'Normal': 0.7, 'Loitering': 0.1, 'Trespass': 0.1, 'Obj. Aband.': 0.1}
            return {
                'classification': 'Normal',
                'confidence': 0.7,
                'all_class_probs': default_probs,
                'top2_classes': [['Normal', 0.7], ['Loitering', 0.1]],
            }
        try:
            x = self._frames_to_cnn3d_tensor(clip)

            with torch.inference_mode():
                logits = self.cnn_3d_model(x)

            if hasattr(self.cnn_3d_model, 'class_names'):
                class_names = self.cnn_3d_model.class_names

            probs = torch.softmax(logits, dim=1)
            probs_list = probs[0].tolist()
            all_class_probs = {class_names[i]: float(probs_list[i]) for i in range(len(class_names))}
            top2 = sorted(all_class_probs.items(), key=lambda x: x[1], reverse=True)[:2]

            confidence, pred_idx = torch.max(probs, dim=1)
            classification = class_names[pred_idx.item()]
            logger.info(f"✓ 3D-CNN: {classification} ({confidence.item():.2%})")
            return {
                'classification': classification,
                'confidence': confidence.item(),
                'all_class_probs': all_class_probs,
                'top2_classes': [[k, v] for k, v in top2],
            }
        except Exception as e:
            logger.error(f"❌ 3D-CNN inference hatası: {e}", exc_info=True)
            return {'classification': 'Error', 'confidence': 0.0, 'all_class_probs': {}, 'top2_classes': []}

    def _generate_heatmap(self, frame: np.ndarray, anomaly_score: float,
                          pixel_mse_map: Optional[np.ndarray] = None) -> str:
        """CR-AE per-pixel MSE haritasını sarı/kırmızı tonlarla frame üzerine bindirer."""
        h, w = frame.shape[:2]

        if pixel_mse_map is not None and pixel_mse_map.max() > 1e-9:
            # Gerçek uzamsal rekonstrüksiyon hatası
            mse_resized = cv2.resize(pixel_mse_map, (w, h))
            mse_norm = (mse_resized / mse_resized.max() * 255).clip(0, 255).astype(np.uint8)
            heatmap = cv2.GaussianBlur(mse_norm, (21, 21), 0)
        elif anomaly_score > 0.1:
            # Yedek: Gauss dağılımlı blob
            heatmap_f = np.zeros((h, w), dtype=np.float32)
            cy, cx = h // 2, w // 2
            sigma = max(min(h, w) * anomaly_score * 0.25, 1.0)
            yy, xx = np.ogrid[:h, :w]
            gaussian = np.exp(-0.5 * ((xx - cx) ** 2 + (yy - cy) ** 2) / sigma ** 2)
            heatmap = (gaussian * 255).clip(0, 255).astype(np.uint8)
        else:
            heatmap = np.zeros((h, w), dtype=np.uint8)

        # COLORMAP_JET: mavi → cyan → yeşil → sarı → kırmızı (standart GradCAM paleti)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Alfa maskeleme: düşük aktivasyon bölgelerinde orijinal frame, yüksek bölgelerde JET
        max_val = float(heatmap.max()) if heatmap.max() > 0 else 1.0
        norm = heatmap.astype(np.float32) / max_val
        # Eşik altını bastır (düşük aktivasyonlar mavi bulanıklık yaratmasın)
        alpha = np.clip((norm - 0.2) / 0.8, 0.0, 1.0)[:, :, np.newaxis] * 0.65
        blended = (frame_bgr * (1.0 - alpha) + heatmap_color * alpha).clip(0, 255).astype(np.uint8)

        _, buf = cv2.imencode('.jpg', blended, [cv2.IMWRITE_JPEG_QUALITY, 82])
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

    async def process_video(self, video_buffer: BytesIO, video_filename: str) -> AsyncGenerator:
        """
        Videoyu 16-frame sliding window kliplere böler ve her klip için
        CR-AE + 3D-CNN inference yaparak sonuçları yield eder.
        """
        logger.info(f"▶ Video işleniyor: {video_filename}")

        frames = self.extract_frames(video_buffer, fps_downsample=FPS_SAMPLE)
        if not frames:
            yield {
                'type': 'error',
                'filename': video_filename,
                'error_message': 'Video frame çıkarılamadı'
            }
            return

        clips = self._build_clips(frames)
        logger.info(f"✓ {len(clips)} klip oluşturuldu (frame={len(frames)}, klip={CLIP_LENGTH}, step={CLIP_STEP})")

        video_metrics = {
            'anomaly_scores': [],
            'classifications': [],
            'confidences': [],
        }

        for clip_idx, clip in enumerate(clips):
            start_time = datetime.now()
            try:
                # CR-AE adımı
                crae_start = datetime.now()
                anomaly_score, raw_mse, recon_frame_b64, orig_frame_b64, pixel_mse_map = await asyncio.to_thread(self._infer_crae, clip)
                crae_time_ms = int((datetime.now() - crae_start).total_seconds() * 1000)
                threshold_exceeded = raw_mse > CRAE_THRESHOLD_MSE

                all_class_probs: Dict = {}
                top2_classes: list = []
                cnn_time_ms = 0

                if threshold_exceeded:
                    cnn_start = datetime.now()
                    cnn_result = await asyncio.to_thread(self._infer_cnn3d, clip)
                    cnn_time_ms = int((datetime.now() - cnn_start).total_seconds() * 1000)
                    all_class_probs = cnn_result.get('all_class_probs', {})
                    top2_classes = cnn_result.get('top2_classes', [])
                    logger.info(
                        f"✓ Klip {clip_idx}: MSE={raw_mse:.5f} > eşik={CRAE_THRESHOLD_MSE} "
                        f"→ CNN: {cnn_result['classification']} ({cnn_result['confidence']:.2%})"
                    )
                else:
                    cnn_result = {'classification': 'Normal', 'confidence': 1.0}
                    logger.info(
                        f"✓ Klip {clip_idx}: MSE={raw_mse:.5f} <= eşik={CRAE_THRESHOLD_MSE} "
                        f"→ CNN atlandı, Normal"
                    )

                classification = cnn_result['classification']
                confidence = cnn_result['confidence']

                mid_frame = clip[len(clip) // 2]
                heatmap_start = datetime.now()
                heatmap_b64 = self._generate_heatmap(mid_frame, anomaly_score, pixel_mse_map)
                heatmap_time_ms = int((datetime.now() - heatmap_start).total_seconds() * 1000)

                latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)

                video_metrics['anomaly_scores'].append(anomaly_score)
                video_metrics['classifications'].append(classification)
                video_metrics['confidences'].append(confidence)

                frame_number = clip_idx * CLIP_STEP

                yield {
                    'frame_number': frame_number,
                    'clip_index': clip_idx,
                    'total_clips': len(clips),
                    'anomaly_score': anomaly_score,
                    'raw_mse': raw_mse,
                    'threshold_exceeded': threshold_exceeded,
                    'classification': classification,
                    'confidence': confidence,
                    'heatmap_base64': heatmap_b64,
                    'latency_ms': latency_ms,
                    'crae_time_ms': crae_time_ms,
                    'cnn_time_ms': cnn_time_ms,
                    'heatmap_time_ms': heatmap_time_ms,
                    'all_class_probs': all_class_probs,
                    'top2_classes': top2_classes,
                    'recon_frame_base64': recon_frame_b64,
                    'orig_frame_base64': orig_frame_b64,
                    'total_frames_processed': len(frames) * FPS_SAMPLE,
                    'frames_sampled': len(frames),
                    'avg_anomaly_score': float(np.mean(video_metrics['anomaly_scores'])),
                    'class_distribution': dict(Counter(video_metrics['classifications'])),
                }

            except Exception as e:
                logger.error(f"❌ Klip {clip_idx} işleme hatası: {e}")
                yield {'frame_number': clip_idx * CLIP_STEP, 'error': str(e)}

        # Video özet
        if video_metrics['anomaly_scores']:
            summary = self._calculate_video_summary(
                video_filename,
                len(frames) * FPS_SAMPLE,
                video_metrics
            )
            yield {'type': 'video_summary', **summary}

    def _calculate_video_summary(self, filename: str, total_frames: int, metrics: Dict) -> Dict:
        anomaly_scores = metrics['anomaly_scores']
        classifications = metrics['classifications']
        confidences = metrics['confidences']

        class_dist = dict(Counter(classifications))
        dominant_class = Counter(classifications).most_common(1)[0][0] if classifications else 'Unknown'

        anomaly_dist = {
            '0-25':   sum(1 for s in anomaly_scores if s < 0.25),
            '25-50':  sum(1 for s in anomaly_scores if 0.25 <= s < 0.5),
            '50-75':  sum(1 for s in anomaly_scores if 0.5 <= s < 0.75),
            '75-100': sum(1 for s in anomaly_scores if s >= 0.75)
        }

        return {
            'filename': filename,
            'total_frames_processed': total_frames,
            'frames_sampled': len(anomaly_scores),
            'avg_anomaly_score': float(np.mean(anomaly_scores)) if anomaly_scores else 0.0,
            'max_anomaly_score': float(np.max(anomaly_scores)) if anomaly_scores else 0.0,
            'anomaly_scores': anomaly_scores,
            'anomaly_distribution': anomaly_dist,
            'avg_confidence': float(np.mean(confidences)) if confidences else 0.0,
            'dominant_class': dominant_class,
            'class_distribution': class_dist
        }

    async def process_video_batch(
        self,
        videos: List[Dict],
        drive_service=None,
        progress_callback=None
    ) -> AsyncGenerator:
        total_videos = len(videos)
        batch_metrics = {
            'all_anomaly_scores': [],
            'all_classifications': [],
            'video_summaries': []
        }
        batch_start = datetime.now()

        for idx, video in enumerate(videos):
            filename = video['filename']
            file_id = video.get('file_id')

            yield {
                'type': 'video_start',
                'filename': filename,
                'video_index': idx,
                'total_videos': total_videos
            }

            try:
                if drive_service is None or not file_id:
                    raise ValueError(f"Drive servisi veya file_id eksik: {filename}")

                video_buffer = drive_service.download_video(file_id, filename)

                async for frame_result in self.process_video(video_buffer, filename):
                    if frame_result.get('type') == 'video_summary':
                        batch_metrics['video_summaries'].append(frame_result)
                        batch_metrics['all_anomaly_scores'].extend(
                            frame_result.get('anomaly_scores', [])
                        )
                        batch_metrics['all_classifications'].append(
                            frame_result.get('dominant_class', 'Unknown')
                        )
                        yield frame_result
                    else:
                        yield {'type': 'frame_processed', **frame_result}

                yield {'type': 'video_complete', 'filename': filename, 'video_index': idx}

            except Exception as e:
                logger.error(f"❌ Video işleme hatası: {filename} - {e}", exc_info=True)
                yield {'type': 'error', 'filename': filename, 'error_message': str(e)}

            if progress_callback:
                progress_callback(idx + 1, total_videos)

        processing_time = int((datetime.now() - batch_start).total_seconds())
        batch_summary = self._calculate_batch_summary(total_videos, batch_metrics, processing_time)
        yield {'type': 'batch_summary', **batch_summary}
        yield {'type': 'batch_complete', 'total_videos': total_videos, 'processing_time_seconds': processing_time}

    def _calculate_batch_summary(self, total_videos: int, metrics: Dict, processing_time: int = 0) -> Dict:
        summaries = metrics['video_summaries']
        classes = metrics['all_classifications']
        all_anomalies = metrics.get('all_anomaly_scores', [])

        return {
            'total_videos': total_videos,
            'total_videos_with_anomalies': sum(
                1 for s in summaries if s.get('avg_anomaly_score', 0) > 0.3
            ),
            'avg_anomaly_across_all': float(np.mean(all_anomalies)) if all_anomalies else 0.0,
            'total_frames_processed': sum(s.get('total_frames_processed', 0) for s in summaries),
            'class_distribution_overall': dict(Counter(classes)),
            'processing_time_seconds': processing_time
        }


def get_video_processor(cr_ae_model=None, cnn_3d_model=None, gradcam_fn=None) -> VideoProcessor:
    return VideoProcessor(cr_ae_model, cnn_3d_model, gradcam_fn)
