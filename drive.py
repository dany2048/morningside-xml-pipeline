"""Google Drive download and upload via service account or OAuth."""
from __future__ import annotations

import io
import os
import re

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload


SCOPES = ["https://www.googleapis.com/auth/drive"]
_TOKEN_PATH = os.path.join(os.path.dirname(__file__), "token.json")
_CLIENT_SECRET_GLOB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "reference",
)


def _get_service():
    """Build Drive API service using OAuth2 credentials."""
    creds = None

    if os.path.exists(_TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(_TOKEN_PATH, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Find client secret file
            client_secret = os.getenv("GOOGLE_CLIENT_SECRET_PATH")
            if not client_secret:
                # Look in reference/ for client_secret*.json
                for f in os.listdir(_CLIENT_SECRET_GLOB):
                    if f.startswith("client_secret") and f.endswith(".json"):
                        client_secret = os.path.join(_CLIENT_SECRET_GLOB, f)
                        break
            if not client_secret:
                raise RuntimeError(
                    "No Google OAuth client secret found. Set GOOGLE_CLIENT_SECRET_PATH "
                    "or place client_secret*.json in reference/"
                )
            flow = InstalledAppFlow.from_client_secrets_file(client_secret, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(_TOKEN_PATH, "w") as f:
            f.write(creds.to_json())

    return build("drive", "v3", credentials=creds)


def parse_drive_file_id(url: str) -> str:
    """Extract file ID from various Google Drive URL formats."""
    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"id=([a-zA-Z0-9_-]+)",
        r"/d/([a-zA-Z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract Drive file ID from URL: {url}")


def download_file(file_id: str, dest_path: str) -> str:
    """Download a file from Drive with progress logging."""
    service = _get_service()

    # Get file metadata
    meta = service.files().get(fileId=file_id, fields="name,size,mimeType").execute()
    name = meta.get("name", "unknown")
    size = int(meta.get("size", 0))
    size_gb = size / (1024**3)
    print(f"  Downloading: {name} ({size_gb:.1f} GB)")

    request = service.files().get_media(fileId=file_id)
    with open(dest_path, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        last_pct = 0
        while not done:
            status, done = downloader.next_chunk()
            if status:
                pct = int(status.progress() * 100)
                if pct >= last_pct + 10:
                    print(f"  Download: {pct}%")
                    last_pct = pct

    print(f"  Download complete: {dest_path}")
    return dest_path


def upload_file(local_path: str, folder_id: str, filename: str) -> str:
    """Upload a file to Drive and return the shareable URL."""
    service = _get_service()

    file_metadata = {
        "name": filename,
        "parents": [folder_id],
    }
    media = MediaFileUpload(local_path, mimetype="application/xml")
    uploaded = service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id,webViewLink",
    ).execute()

    link = uploaded.get("webViewLink", f"https://drive.google.com/file/d/{uploaded['id']}/view")
    print(f"  Uploaded to Drive: {link}")
    return link


def get_file_name(file_id: str) -> str:
    """Get just the filename for a Drive file."""
    service = _get_service()
    meta = service.files().get(fileId=file_id, fields="name").execute()
    return meta.get("name", "unknown")
