from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import httpx

from app.clients.storage import AzureStorageClient
from app.config import Settings
from app.constants import wardrobe as wardrobe_constants

logger = logging.getLogger("glamify-ai")

_UPLOAD_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="wardrobe-upload")
_SYNC_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="wardrobe-sync")


class GlamifyProgressClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def is_configured(self) -> bool:
        return bool(str(self._settings.glamify_api_base_url or "").strip())

    def upload_input_background(
        self,
        *,
        content: bytes,
        object_name: str,
        content_type: str,
    ) -> Future[str]:
        return _UPLOAD_EXECUTOR.submit(
            self._upload_bytes,
            content=content,
            object_name=object_name,
            content_type=content_type,
        )

    def submit_output_and_progress_background(
        self,
        *,
        bearer_token: str,
        progress_id: str,
        input_url_future: Future[str],
        output_content: bytes,
        output_object_name: str,
        output_content_type: str,
        classification: dict[str, Any],
        marqo: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        if not self.is_configured:
            logger.info("Skipping wardrobe progress sync: GLAMIFY_API_BASE_URL is empty.")
            return

        def _job() -> None:
            try:
                input_url = input_url_future.result()
                output_url = self._upload_bytes(
                    content=output_content,
                    object_name=output_object_name,
                    content_type=output_content_type,
                )
                self.create_or_update_progress(
                    bearer_token=bearer_token,
                    progress_id=progress_id,
                    input_url=input_url,
                    output_url=output_url,
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
        bearer_token: str,
        progress_id: str,
        input_url: str,
        output_url: str,
        classification: dict[str, Any],
        marqo: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        base_url = str(self._settings.glamify_api_base_url).rstrip("/")
        payload = {
            "id": progress_id,
            "inputImage": input_url,
            "outputImage": output_url,
            "metadata": {
                "classification": classification,
                "marqo": marqo,
                **metadata,
            },
        }
        headers = {
            "Authorization": bearer_token,
            "Content-Type": "application/json",
        }
        timeout = max(5, int(wardrobe_constants.GLAMIFY_API_TIMEOUT_SECONDS))
        with httpx.Client(timeout=timeout) as client:
            response = client.post(f"{base_url}/wardrobe/progress", headers=headers, json=payload)
            response.raise_for_status()

    def _upload_bytes(
        self,
        *,
        content: bytes,
        object_name: str,
        content_type: str,
    ) -> str:
        storage_client = AzureStorageClient(self._settings)
        return storage_client.upload_bytes(
            content,
            object_name=object_name,
            content_type=content_type,
        )
