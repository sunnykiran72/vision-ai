from __future__ import annotations

import atexit
import mimetypes
import shutil
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx

DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 30
DEFAULT_IMAGE_EXTENSION = ".jpg"

_DEFAULT_DOWNLOAD_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Process-wide pooled HTTP client: reused across downloads so repeat fetches to the same host (Azure
# blob) skip the ~0.58s TCP+TLS handshake via keep-alive (measured: 1.77s cold -> ~0.36s reused).
# httpx.Client is thread-safe for concurrent .get() (tryon downloads user+garment in a ThreadPool).
_DOWNLOAD_CLIENT: httpx.Client | None = None
_DOWNLOAD_CLIENT_LOCK = threading.Lock()


def _get_download_client() -> httpx.Client:
    global _DOWNLOAD_CLIENT
    client = _DOWNLOAD_CLIENT
    if client is None:
        with _DOWNLOAD_CLIENT_LOCK:
            client = _DOWNLOAD_CLIENT
            if client is None:
                client = httpx.Client(
                    follow_redirects=True,
                    timeout=httpx.Timeout(float(DEFAULT_DOWNLOAD_TIMEOUT_SECONDS)),
                    headers=_DEFAULT_DOWNLOAD_HEADERS,
                    limits=httpx.Limits(
                        max_keepalive_connections=20,
                        max_connections=40,
                        keepalive_expiry=60.0,
                    ),
                )
                _DOWNLOAD_CLIENT = client
                atexit.register(client.close)
    return client


@dataclass(frozen=True)
class DownloadedMedia:
    content: bytes
    content_type: str | None
    source_url: str
    filename: str


@dataclass(frozen=True)
class JobMediaPaths:
    job_id: str
    job_dir: Path
    input_path: Path
    output_path: Path


@dataclass(frozen=True)
class TryonJobMediaPaths:
    job_id: str
    job_dir: Path
    person_path: Path
    garment_reference_path: Path
    output_path: Path


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_filename(filename: str, default_name: str = "output") -> str:
    candidate = Path((filename or "").strip()).name
    if not candidate:
        candidate = default_name
    return candidate.replace("/", "_").replace("\\", "_")


def ensure_filename_extension(filename: str, extension: str = DEFAULT_IMAGE_EXTENSION) -> str:
    safe_name = sanitize_filename(filename)
    suffix = extension if extension.startswith(".") else f".{extension}"
    if Path(safe_name).suffix:
        return safe_name
    return f"{safe_name}{suffix}"


def build_job_media_paths(
    root_dir: Path,
    input_extension: str = DEFAULT_IMAGE_EXTENSION,
    output_extension: str = DEFAULT_IMAGE_EXTENSION,
) -> JobMediaPaths:
    job_id = uuid.uuid4().hex
    job_dir = ensure_directory(root_dir / job_id)
    input_name = ensure_filename_extension("input", input_extension)
    output_name = ensure_filename_extension("output", output_extension)
    return JobMediaPaths(
        job_id=job_id,
        job_dir=job_dir,
        input_path=job_dir / input_name,
        output_path=job_dir / output_name,
    )


def build_tryon_job_media_paths(root_dir: Path) -> TryonJobMediaPaths:
    job_id = uuid.uuid4().hex
    job_dir = ensure_directory(root_dir / job_id)
    return TryonJobMediaPaths(
        job_id=job_id,
        job_dir=job_dir,
        person_path=job_dir / "person.jpg",
        garment_reference_path=job_dir / "garment_reference.jpg",
        output_path=job_dir / "output.jpg",
    )


def build_storage_object_name(
    output_filename: str | None,
    *,
    prefix: str,
    default_name: str = "output",
    extension: str = DEFAULT_IMAGE_EXTENSION,
) -> str:
    safe_name = ensure_filename_extension(output_filename or default_name, extension)
    safe_prefix = str(prefix).strip().strip("/")
    return f"{safe_prefix}/{safe_name}" if safe_prefix else safe_name


def infer_extension_from_url_or_content_type(
    source_url: str,
    content_type: str | None,
    default_extension: str = DEFAULT_IMAGE_EXTENSION,
) -> str:
    parsed = urlparse(source_url)
    suffix = Path(parsed.path).suffix
    if suffix:
        return suffix.lower()
    guessed = mimetypes.guess_extension((content_type or "").split(";")[0].strip())
    if guessed:
        return guessed.lower()
    return default_extension


def download_media_from_url(
    source_url: str,
    timeout_seconds: int = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
) -> DownloadedMedia:
    client = _get_download_client()
    response = client.get(source_url, timeout=max(10, int(timeout_seconds)))
    response.raise_for_status()
    content_type = response.headers.get("content-type")
    extension = infer_extension_from_url_or_content_type(source_url, content_type)
    filename = ensure_filename_extension(
        Path(urlparse(source_url).path).name or "input",
        extension,
    )
    return DownloadedMedia(
        content=response.content,
        content_type=content_type,
        source_url=source_url,
        filename=filename,
    )


def write_bytes_to_file(path: Path, content: bytes) -> Path:
    ensure_directory(path.parent)
    path.write_bytes(content)
    return path


def cleanup_directory(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
