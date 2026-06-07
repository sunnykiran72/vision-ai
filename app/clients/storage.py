from __future__ import annotations

import mimetypes
from functools import lru_cache
from pathlib import Path

from azure.storage.blob import BlobServiceClient, ContentSettings

from app.config import Settings


class AzureStorageClient:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client: BlobServiceClient | None = None
        self._account_name = self._extract_account_name(settings.azure_storage_connection_string)

    @property
    def is_configured(self) -> bool:
        return bool(
            self._settings.azure_storage_connection_string
            and self._settings.azure_storage_container
        )

    def upload_bytes(
        self,
        content: bytes,
        *,
        object_name: str,
        content_type: str | None = None,
        container: str | None = None,
    ) -> str:
        if not self.is_configured:
            raise RuntimeError("Azure storage is not configured.")

        resolved_container = container or self._settings.azure_storage_container
        blob_client = self._get_client().get_blob_client(
            container=resolved_container,
            blob=object_name,
        )
        resolved_content_type = content_type or self._infer_content_type(object_name)
        content_settings = None
        if resolved_content_type:
            content_settings = ContentSettings(
                content_type=resolved_content_type,
            )
        blob_client.upload_blob(
            content,
            overwrite=True,
            content_settings=content_settings,
        )
        return self.build_blob_url(object_name, container=resolved_container)

    def upload_file(
        self,
        file_path: Path,
        *,
        object_name: str,
        content_type: str | None = None,
        container: str | None = None,
    ) -> str:
        return self.upload_bytes(
            file_path.read_bytes(),
            object_name=object_name,
            content_type=content_type,
            container=container,
        )

    def build_blob_url(self, object_name: str, *, container: str | None = None) -> str:
        if not self._account_name:
            raise RuntimeError("Unable to derive Azure storage account name.")
        resolved_container = container or self._settings.azure_storage_container
        return (
            f"https://{self._account_name}.blob.core.windows.net/"
            f"{resolved_container}/{object_name}"
        )

    def _get_client(self) -> BlobServiceClient:
        if self._client is None:
            self._client = _get_blob_service_client(
                self._settings.azure_storage_connection_string,
            )
        return self._client

    @staticmethod
    def _extract_account_name(connection_string: str) -> str | None:
        marker = "AccountName="
        if marker not in connection_string:
            return None
        return connection_string.split(marker, 1)[1].split(";", 1)[0].strip() or None

    @staticmethod
    def _infer_content_type(object_name: str) -> str:
        guessed, _ = mimetypes.guess_type(Path(object_name).name)
        return guessed or "application/octet-stream"


@lru_cache(maxsize=8)
def _get_blob_service_client(connection_string: str) -> BlobServiceClient:
    return BlobServiceClient.from_connection_string(connection_string)
