from __future__ import annotations

import base64
from io import BytesIO

from PIL import Image

from app.clients.minicpm_v46 import MiniCPMDescription
from app.constants import http_status
from app.models.minicpm import MiniCPMGarmentRequest
from app.services import minicpm as minicpm_service


class FakeMiniCPMClient:
    def __init__(self) -> None:
        self.prompt = ""

    def describe_garment(
        self,
        *,
        image: Image.Image,
        prompt: str,
    ) -> MiniCPMDescription:
        assert image.size == (512, 512)
        self.prompt = prompt
        return MiniCPMDescription(
            text="cropped top with a square neckline and fitted torso construction.",
            latency_ms=123,
            model_id="openbmb/MiniCPM-V-4.6",
            device="cuda:0",
            dtype="torch.bfloat16",
            downsample_mode="16x",
            max_new_tokens=160,
            max_slice_nums=36,
        )


def test_minicpm_garment_request_uses_default_prompt(monkeypatch) -> None:
    fake_client = FakeMiniCPMClient()
    monkeypatch.setattr(minicpm_service, "get_minicpm_v46_client", lambda: fake_client)

    response = minicpm_service.run_minicpm_garment_request(
        MiniCPMGarmentRequest(
            image=_sample_base64_image(),
            type="top",
        ),
    )

    assert response.status == http_status.OK
    assert response.data is not None
    assert response.data.type == "top"
    assert response.data.model == "openbmb/MiniCPM-V-4.6"
    assert response.data.description.startswith("cropped top")
    assert "Describe only the TOP garment" in response.data.prompt
    assert fake_client.prompt == response.data.prompt
    assert response.data.metadata["prompt_source"] == "default"


def test_minicpm_garment_request_accepts_prompt_override(monkeypatch) -> None:
    fake_client = FakeMiniCPMClient()
    monkeypatch.setattr(minicpm_service, "get_minicpm_v46_client", lambda: fake_client)

    response = minicpm_service.run_minicpm_garment_request(
        MiniCPMGarmentRequest(
            image=_sample_base64_image(),
            type="bottom",
            prompt="Describe only visible trouser construction.",
        ),
    )

    assert response.status == http_status.OK
    assert response.data is not None
    assert response.data.prompt == "Describe only visible trouser construction."
    assert response.data.metadata["prompt_source"] == "override"


def test_minicpm_garment_request_rejects_invalid_image() -> None:
    response = minicpm_service.run_minicpm_garment_request(
        MiniCPMGarmentRequest(
            image="not-base64",
            type="dress",
        ),
    )

    assert response.status == http_status.UNPROCESSABLE_CONTENT
    assert response.data is None


def _sample_base64_image() -> str:
    image = Image.new("RGB", (512, 512), "white")
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
