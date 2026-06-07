from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import httpx

from app.clients.storage import AzureStorageClient
from app.config import Settings
from app.constants import wardrobe as wardrobe_constants

logger = logging.getLogger("glamify-ai")

_UPLOAD_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="wardrobe-upload")
_SYNC_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="wardrobe-sync")


@dataclass(frozen=True)
class TimedUploadResult:
    url: str
    wall_seconds: float
    container: str
    object_name: str
    bytes: int


class GlamifyProgressClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def is_configured(self) -> bool:
        return bool(str(self._settings.glamify_api_base_url or "").strip())

    def upload_background(
        self,
        *,
        content: bytes,
        object_name: str,
        container: str,
        content_type: str,
    ) -> Future[str]:
        """Upload bytes to a specific Azure container off the request path; returns a Future URL."""
        return _UPLOAD_EXECUTOR.submit(
            self._upload_bytes,
            content=content,
            object_name=object_name,
            container=container,
            content_type=content_type,
        )

    def upload_background_timed(
        self,
        *,
        content: bytes,
        object_name: str,
        container: str,
        content_type: str,
    ) -> Future[TimedUploadResult]:
        """Upload bytes off the request path and retain upload duration for debug metadata."""
        return _UPLOAD_EXECUTOR.submit(
            self._upload_bytes_timed,
            content=content,
            object_name=object_name,
            container=container,
            content_type=content_type,
        )

    def submit_progress_background(
        self,
        *,
        access_token: str,
        progress_id: str,
        input_url_future: Future[str | TimedUploadResult],
        output_url: str,
        prompt_description: str,
        classification: dict[str, Any],
        marqo: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        if not self.is_configured:
            logger.info("Skipping wardrobe progress sync: GLAMIFY_API_BASE_URL is empty.")
            return

        def _job() -> None:
            try:
                input_url = input_url_future.result(
                    timeout=wardrobe_constants.AZURE_UPLOAD_TIMEOUT_SECONDS,
                )
                if isinstance(input_url, TimedUploadResult):
                    input_url = input_url.url
                self.create_or_update_progress(
                    access_token=access_token,
                    progress_id=progress_id,
                    input_url=input_url,
                    output_url=output_url,
                    prompt_description=prompt_description,
                    classification=classification,
                    marqo=marqo,
                    metadata=metadata,
                )
            except Exception as exc:
                logger.error("Background wardrobe progress sync failed: %s", exc, exc_info=True)

        _SYNC_EXECUTOR.submit(_job)

    def create_or_update_progress(
        self,
        *,
        access_token: str,
        progress_id: str,
        input_url: str,
        output_url: str,
        prompt_description: str,
        classification: dict[str, Any],
        marqo: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        endpoint_url = _wardrobe_progress_endpoint_url(self._settings.glamify_api_base_url)
        payload = {
            "id": progress_id,
            "inputImage": input_url,
            "outputImage": output_url,
            "promptDescription": prompt_description,
            "metadata": {
                "classification": classification,
                "marqo": marqo,
                **metadata,
            },
        }
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        timeout = max(5, int(wardrobe_constants.GLAMIFY_API_TIMEOUT_SECONDS))
        with httpx.Client(timeout=timeout) as client:
            response = client.post(endpoint_url, headers=headers, json=payload)
            response.raise_for_status()

    def _upload_bytes(
        self,
        *,
        content: bytes,
        object_name: str,
        container: str,
        content_type: str,
    ) -> str:
        storage_client = AzureStorageClient(self._settings)
        return storage_client.upload_bytes(
            content,
            object_name=object_name,
            content_type=content_type,
            container=container,
        )

    def _upload_bytes_timed(
        self,
        *,
        content: bytes,
        object_name: str,
        container: str,
        content_type: str,
    ) -> TimedUploadResult:
        started = perf_counter()
        url = self._upload_bytes(
            content=content,
            object_name=object_name,
            container=container,
            content_type=content_type,
        )
        return TimedUploadResult(
            url=url,
            wall_seconds=float(round(perf_counter() - started, 3)),
            container=container,
            object_name=object_name,
            bytes=len(content),
        )


def _wardrobe_progress_endpoint_url(raw_base_url: str) -> str:
    base_url = str(raw_base_url or "").rstrip("/")
    suffix = "/wardrobe/progress"
    if base_url.endswith(suffix):
        return base_url
    return f"{base_url}{suffix}"
