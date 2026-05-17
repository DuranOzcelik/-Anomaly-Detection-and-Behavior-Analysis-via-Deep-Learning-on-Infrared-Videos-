import asyncio
import logging
from pathlib import Path
from typing import AsyncGenerator

logger = logging.getLogger(__name__)

POLL_INTERVAL = 8  # seconds between Drive polls


class DriveWatcher:
    """Watch a Google Drive folder and run new videos through the CR-AE + CNN pipeline."""

    def __init__(self, video_processor, drive_service, folder_id: str):
        self.vp        = video_processor
        self.ds        = drive_service
        self.folder_id = folder_id
        self._seen_ids: set = set()

    _VIDEO_MIME = (
        "mimeType='video/mp4' or mimeType='video/x-msvideo' or "
        "mimeType='video/quicktime' or mimeType='video/x-matroska'"
    )

    def _list_videos(self) -> list:
        """List videos in the folder and one level of subfolders."""
        all_videos = []

        direct_q = (
            f"'{self.folder_id}' in parents and trashed=false and "
            f"({self._VIDEO_MIME})"
        )
        for f in self.ds._list_files(direct_q, fields='files(id, name, size, mimeType)', page_size=200):
            f['_subfolder'] = ''
            all_videos.append(f)

        sf_q = (
            f"'{self.folder_id}' in parents and trashed=false and "
            "mimeType='application/vnd.google-apps.folder'"
        )
        for sf in self.ds._list_files(sf_q, fields='files(id, name)', page_size=50):
            sf_video_q = (
                f"'{sf['id']}' in parents and trashed=false and "
                f"({self._VIDEO_MIME})"
            )
            for f in self.ds._list_files(sf_video_q, fields='files(id, name, size, mimeType)', page_size=200):
                f['_subfolder'] = sf['name']
                all_videos.append(f)

        return all_videos

    async def _process_one(self, file_id: str, filename: str) -> AsyncGenerator:
        """Download one video from Drive and run it through the pipeline."""
        video_buffer = await asyncio.to_thread(
            self.ds.download_video, file_id, filename
        )
        safe_name = Path(filename).name
        temp_path = Path('temp_videos') / safe_name
        video_buffer.seek(0)
        with open(temp_path, 'wb') as vf:
            vf.write(video_buffer.read())
        video_buffer.seek(0)

        yield {
            'type': 'video_ready',
            'filename': filename,
            'video_url': f'http://127.0.0.1:8000/video/{safe_name}',
        }

        async for result in self.vp.process_video(video_buffer, filename):
            if result.get('type') == 'video_summary':
                yield result
            elif 'type' not in result:
                yield {'type': 'frame_processed', **result}
            else:
                yield result

    async def watch(
        self,
        stop_event: asyncio.Event,
        file_id:  str = None,
        filename: str = None,
    ) -> AsyncGenerator:
        """
        Single-video mode (file_id provided): process that one file then exit.
        Folder mode: process existing videos, then poll for new ones.
        """
        yield {'type': 'drive_connected', 'folder_id': self.folder_id}

        if file_id:
            target_name = filename or file_id
            yield {'type': 'drive_status', 'message': f'Analyzing: {target_name}'}
            yield {'type': 'video_start', 'filename': target_name, 'date_folder': 'drive'}
            try:
                async for result in self._process_one(file_id, target_name):
                    if stop_event.is_set():
                        break
                    yield result
                yield {'type': 'video_complete', 'filename': target_name}
            except Exception as e:
                logger.error(f"Video processing error ({target_name}): {e}", exc_info=True)
                yield {'type': 'error', 'filename': target_name, 'error_message': str(e)}
            yield {'type': 'drive_stopped'}
            return

        try:
            all_files = await asyncio.to_thread(self._list_videos)
        except Exception as e:
            logger.error(f"Drive folder initial scan error: {e}", exc_info=True)
            yield {'type': 'error', 'error_message': f'Could not read folder: {e}'}
            return

        yield {
            'type': 'drive_status',
            'message': (
                f'Processing {len(all_files)} existing video(s)...'
                if all_files else
                'Folder is empty — waiting for new videos...'
            ),
        }

        for f in all_files:
            if stop_event.is_set():
                break
            self._seen_ids.add(f['id'])
            yield {'type': 'video_start', 'filename': f['name'], 'date_folder': f.get('_subfolder', '') or 'drive'}
            try:
                async for result in self._process_one(f['id'], f['name']):
                    if stop_event.is_set():
                        break
                    yield result
                yield {'type': 'video_complete', 'filename': f['name']}
            except Exception as e:
                logger.error(f"Video processing error ({f['name']}): {e}", exc_info=True)
                yield {'type': 'error', 'filename': f['name'], 'error_message': str(e)}

        if stop_event.is_set():
            yield {'type': 'drive_stopped'}
            return

        yield {'type': 'drive_status', 'message': 'Watching for new videos...'}

        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=POLL_INTERVAL)
            except asyncio.TimeoutError:
                pass

            if stop_event.is_set():
                break

            try:
                all_files = await asyncio.to_thread(self._list_videos)
                new_files = [f for f in all_files if f['id'] not in self._seen_ids]
            except Exception as e:
                logger.error(f"Drive poll error: {e}")
                yield {'type': 'drive_poll_error', 'error_message': str(e)}
                continue

            for f in new_files:
                if stop_event.is_set():
                    break
                self._seen_ids.add(f['id'])
                logger.info(f"New video detected: {f['name']}")
                yield {'type': 'video_start', 'filename': f['name'], 'date_folder': f.get('_subfolder', '') or 'drive'}
                try:
                    async for result in self._process_one(f['id'], f['name']):
                        if stop_event.is_set():
                            break
                        yield result
                    yield {'type': 'video_complete', 'filename': f['name']}
                except Exception as e:
                    logger.error(f"Video processing error ({f['name']}): {e}", exc_info=True)
                    yield {'type': 'error', 'filename': f['name'], 'error_message': str(e)}

        yield {'type': 'drive_stopped'}
