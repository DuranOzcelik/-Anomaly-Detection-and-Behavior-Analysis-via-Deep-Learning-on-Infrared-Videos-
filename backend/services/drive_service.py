import os
from typing import List, Optional, Tuple
from io import BytesIO
import logging
from dotenv import load_dotenv

load_dotenv()

try:
    from google.oauth2.service_account import Credentials
    from googleapiclient import discovery
    from googleapiclient.http import MediaIoBaseDownload
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    Credentials = None
    discovery = None

logger = logging.getLogger(__name__)

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

FOLDER_BASE_PATH = "archive/LTD Dataset/LTD Dataset/Video Clips"
MODEL_ROOT_FOLDER = "archive"


class DriveService:
    def __init__(self, credentials_path: Optional[str] = None):
        self.credentials_path = credentials_path or os.getenv('GOOGLE_DRIVE_CREDENTIALS_PATH')
        self.service = None
        self.folder_cache = {}

    def _check_service(self):
        if self.service is None:
            raise RuntimeError("Drive service not initialized. Call authenticate() first.")

    def authenticate(self):
        if not GOOGLE_AVAILABLE:
            raise RuntimeError(
                "google-api-python-client not installed. "
                "Run: pip install google-api-python-client google-auth"
            )
        if not self.credentials_path:
            raise ValueError(
                "GOOGLE_DRIVE_CREDENTIALS_PATH environment variable not set."
            )
        if not os.path.exists(self.credentials_path):
            raise FileNotFoundError(
                f"Service account credentials file not found: {self.credentials_path}"
            )

        credentials = Credentials.from_service_account_file(
            self.credentials_path, scopes=SCOPES
        )
        self.service = discovery.build('drive', 'v3', credentials=credentials)
        logger.info("Google Drive authentication successful")
        return self.service

    def _list_files(self, query: str, fields: str, page_size: int = 100) -> list:
        """List files with pagination support; works for both My Drive and Shared Drives."""
        self._check_service()
        all_files = []
        page_token = None

        while True:
            params = dict(
                q=query,
                spaces='drive',
                fields=f"nextPageToken, {fields}",
                pageSize=min(page_size, 1000),
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
            )
            if page_token:
                params['pageToken'] = page_token

            results = self.service.files().list(**params).execute()
            all_files.extend(results.get('files', []))

            page_token = results.get('nextPageToken')
            if not page_token or len(all_files) >= page_size:
                break

        return all_files

    def find_folder_by_path(self, folder_path: str, parent_id: str = "root") -> Optional[str]:
        """Resolve a slash-separated folder path to a Drive folder ID."""
        self._check_service()
        path_parts = [p for p in folder_path.split('/') if p]
        current_parent = parent_id

        for part in path_parts:
            cache_key = f"{current_parent}/{part}"
            if cache_key in self.folder_cache:
                current_parent = self.folder_cache[cache_key]
                continue

            safe_part = part.replace("'", "\\'")

            query = (
                f"name='{safe_part}' and "
                f"mimeType='application/vnd.google-apps.folder' and "
                f"'{current_parent}' in parents and trashed=false"
            )
            files = self._list_files(query, fields='files(id, name)', page_size=10)

            if not files and current_parent == "root":
                query_shared = (
                    f"name='{safe_part}' and "
                    f"mimeType='application/vnd.google-apps.folder' and "
                    f"sharedWithMe=true and trashed=false"
                )
                files = self._list_files(query_shared, fields='files(id, name)', page_size=10)

            if not files:
                logger.warning(f"Folder not found: '{part}' (parent: {current_parent})")
                return None

            current_parent = files[0]['id']
            self.folder_cache[cache_key] = current_parent
            logger.info(f"Folder resolved: '{part}' -> {current_parent}")

        return current_parent

    def list_date_folders(self, start_date: str, end_date: str) -> List[Tuple[str, str]]:
        """List YYYYMMDD-named subfolders within the date range."""
        self._check_service()
        base_folder_id = self.find_folder_by_path(FOLDER_BASE_PATH)
        if not base_folder_id:
            logger.error(f"Base video folder not found: {FOLDER_BASE_PATH}")
            return []

        start_int, end_int = int(start_date), int(end_date)
        query = (
            f"'{base_folder_id}' in parents and "
            f"mimeType='application/vnd.google-apps.folder' and trashed=false"
        )
        files = self._list_files(query, fields='files(id, name)', page_size=500)

        date_folders = []
        for file in files:
            name = file['name']
            if len(name) == 8 and name.isdigit():
                if start_int <= int(name) <= end_int:
                    date_folders.append((name, file['id']))

        date_folders.sort(key=lambda x: x[0])
        logger.info(f"{len(date_folders)} date folders found ({start_date} -> {end_date})")
        return date_folders

    def list_videos_by_date_range(self, start_date: str, end_date: str) -> List[dict]:
        """List all video files within the given date range."""
        self._check_service()
        videos = []
        for date_folder, folder_id in self.list_date_folders(start_date, end_date):
            query = (
                f"'{folder_id}' in parents and trashed=false and "
                f"(mimeType='video/mp4' or mimeType='video/x-msvideo' or "
                f"mimeType='video/quicktime' or mimeType='video/x-matroska')"
            )
            files = self._list_files(
                query,
                fields='files(id, name, size, mimeType)',
                page_size=500
            )
            for file in files:
                videos.append({
                    'filename': file['name'],
                    'file_id': file['id'],
                    'date_folder': date_folder,
                    'size': int(file.get('size', 0)),
                    'mime_type': file['mimeType']
                })

        logger.info(f"Total {len(videos)} videos found")
        return videos

    def download_video(self, file_id: str, filename: str) -> BytesIO:
        """Download a video file from Drive using chunked MediaIoBaseDownload."""
        self._check_service()
        try:
            fh = BytesIO()
            request = self.service.files().get_media(
                fileId=file_id,
                supportsAllDrives=True
            )
            downloader = MediaIoBaseDownload(fh, request, chunksize=10 * 1024 * 1024)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    logger.debug(f"  Download: {int(status.progress() * 100)}% — {filename}")

            fh.seek(0)
            logger.info(f"Downloaded: {filename} ({fh.getbuffer().nbytes / 1024 / 1024:.1f} MB)")
            return fh
        except Exception as e:
            logger.error(f"Video download error ({filename}): {e}", exc_info=True)
            raise

    def download_file(self, file_id: str) -> BytesIO:
        """Download any file from Drive (used for model weights)."""
        self._check_service()
        try:
            fh = BytesIO()
            request = self.service.files().get_media(
                fileId=file_id,
                supportsAllDrives=True
            )
            downloader = MediaIoBaseDownload(fh, request, chunksize=10 * 1024 * 1024)
            done = False
            while not done:
                _, done = downloader.next_chunk()

            fh.seek(0)
            logger.info(f"File downloaded: {file_id} ({fh.getbuffer().nbytes} bytes)")
            return fh
        except Exception as e:
            logger.error(f"File download error ({file_id}): {e}", exc_info=True)
            raise

    def find_file_by_path(self, file_path: str, parent_id: str = "root") -> Optional[str]:
        """Resolve a Drive file path to a file ID."""
        self._check_service()
        normalized = file_path.strip().strip('/')
        if not normalized:
            return None

        parts = normalized.split('/')
        file_name = parts[-1]
        dir_parts = parts[:-1]

        current_parent = parent_id

        if dir_parts:
            dir_path = '/'.join(dir_parts)
            current_parent = self.find_folder_by_path(dir_path, parent_id=parent_id)
            if not current_parent:
                logger.warning(f"Folder not found, searching sharedWithMe for: {file_name}")
                safe_name = file_name.replace("'", "\\'")
                query_shared = (
                    f"name='{safe_name}' and "
                    f"sharedWithMe=true and trashed=false"
                )
                files = self._list_files(query_shared, fields='files(id, name)', page_size=5)
                if files:
                    logger.info(f"File found in sharedWithMe: {file_name} -> {files[0]['id']}")
                    return files[0]['id']
                logger.warning(f"File not found: {file_path}")
                return None

        safe_name = file_name.replace("'", "\\'")
        query = (
            f"name='{safe_name}' and "
            f"'{current_parent}' in parents and trashed=false"
        )
        files = self._list_files(query, fields='files(id, name)', page_size=5)

        if not files and current_parent == 'root':
            query_shared = (
                f"name='{safe_name}' and "
                f"sharedWithMe=true and trashed=false"
            )
            files = self._list_files(query_shared, fields='files(id, name)', page_size=5)

        if not files:
            logger.warning(f"Drive file not found: {file_path}")
            return None

        logger.info(f"File found: {file_name} -> {files[0]['id']}")
        return files[0]['id']

    def download_file_by_path(self, file_path: str) -> BytesIO:
        """Find and download a file by its Drive path. Prepends MODEL_ROOT_FOLDER automatically."""
        full_path = f"{MODEL_ROOT_FOLDER}/{file_path}" if MODEL_ROOT_FOLDER else file_path
        file_id = self.find_file_by_path(full_path)
        if not file_id:
            raise FileNotFoundError(
                f"File not found on Drive: '{full_path}'\n"
                f"  - Does the service account have access to this file?\n"
                f"  - Is MODEL_ROOT_FOLDER='{MODEL_ROOT_FOLDER}' correct?\n"
                f"  - Exact filename on Drive: '{file_path.split('/')[-1]}'"
            )
        return self.download_file(file_id)


def get_drive_service() -> DriveService:
    service = DriveService()
    service.authenticate()
    return service
