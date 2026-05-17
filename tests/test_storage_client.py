from pathlib import Path

from app.clients.storage import AzureStorageClient
from app.config import Settings


def test_build_blob_url_uses_account_and_container() -> None:
    settings = Settings(
        AZURE_STORAGE_CONNECTION_STRING=(
            "DefaultEndpointsProtocol=https;"
            "AccountName=glamifydevstorage;"
            "AccountKey=test;"
            "EndpointSuffix=core.windows.net"
        ),
        AZURE_STORAGE_CONTAINER="wardrobe-outputs",
    )
    client = AzureStorageClient(settings)

    url = client.build_blob_url("seedvr2/output.jpg")

    assert (
        url
        == "https://glamifydevstorage.blob.core.windows.net/wardrobe-outputs/seedvr2/output.jpg"
    )


def test_storage_client_infers_jpg_content_type() -> None:
    settings = Settings(
        AZURE_STORAGE_CONNECTION_STRING=(
            "DefaultEndpointsProtocol=https;"
            "AccountName=glamifydevstorage;"
            "AccountKey=test;"
            "EndpointSuffix=core.windows.net"
        ),
        AZURE_STORAGE_CONTAINER="wardrobe-outputs",
    )
    client = AzureStorageClient(settings)

    assert client._infer_content_type("result.jpg") == "image/jpeg"


def test_upload_file_raises_when_storage_not_configured(tmp_path: Path) -> None:
    settings = Settings()
    client = AzureStorageClient(settings)
    file_path = tmp_path / "result.jpg"
    file_path.write_bytes(b"test")

    try:
        client.upload_file(file_path, object_name="seedvr2/result.jpg")
    except RuntimeError as exc:
        assert str(exc) == "Azure storage is not configured."
    else:
        raise AssertionError("Expected RuntimeError when Azure storage is not configured")
