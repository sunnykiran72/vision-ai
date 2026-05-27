from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = Field(default="local", alias="APP_ENV")
    startup_warmup_enabled: bool = Field(default=True, alias="STARTUP_WARMUP_ENABLED")

    jwt_access_secret: str = Field(default="", alias="JWT_ACCESS_SECRET")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    azure_storage_connection_string: str = Field(
        default="",
        alias="AZURE_STORAGE_CONNECTION_STRING",
    )
    azure_storage_container: str = Field(default="", alias="AZURE_STORAGE_CONTAINER")

    qwen_image_edit_model_path: str = Field(
        default="/workspace/models/qwen-image-edit-2511",
        alias="QWEN_IMAGE_EDIT_MODEL_PATH",
    )
    ai_toolkit_root: str = Field(
        default="/workspace/ai-toolkit",
        alias="AI_TOOLKIT_ROOT",
    )

    wardrobe_lora_path: str = Field(default="", alias="WARDROBE_LORA_PATH")
    wardrobe_lora_weight_name: str = Field(default="", alias="WARDROBE_LORA_WEIGHT_NAME")

    tryon_lora_path: str = Field(default="", alias="TRYON_LORA_PATH")
    tryon_lora_weight_name: str = Field(default="", alias="TRYON_LORA_WEIGHT_NAME")
    tryon_lora_rank: int = Field(default=64, alias="TRYON_LORA_RANK")
    tryon_lora_alpha: int = Field(default=64, alias="TRYON_LORA_ALPHA")
    tryon_lora_scale: float = Field(default=1.0, alias="TRYON_LORA_SCALE")
    tryon_default_seed: int = Field(default=43, alias="TRYON_DEFAULT_SEED")
    tryon_default_steps: int = Field(default=25, alias="TRYON_DEFAULT_STEPS")
    tryon_default_guidance_scale: float = Field(
        default=1.0,
        alias="TRYON_DEFAULT_GUIDANCE_SCALE",
    )
    tryon_guidance_rescale: float = Field(default=0.0, alias="TRYON_GUIDANCE_RESCALE")
    tryon_do_cfg_norm: bool = Field(default=False, alias="TRYON_DO_CFG_NORM")
    tryon_sampler: str = Field(default="flowmatch", alias="TRYON_SAMPLER")
    tryon_output_width: int = Field(default=1024, alias="TRYON_OUTPUT_WIDTH")
    tryon_output_height: int = Field(default=1536, alias="TRYON_OUTPUT_HEIGHT")
    tryon_preview_width: int = Field(default=896, alias="TRYON_PREVIEW_WIDTH")
    tryon_preview_height: int = Field(default=1344, alias="TRYON_PREVIEW_HEIGHT")
    tryon_preview_steps: int = Field(default=20, alias="TRYON_PREVIEW_STEPS")
    tryon_queue_max_size: int = Field(default=8, alias="TRYON_QUEUE_MAX_SIZE")
    tryon_queue_wait_timeout_seconds: int = Field(
        default=30,
        alias="TRYON_QUEUE_WAIT_TIMEOUT_SECONDS",
    )
    tryon_work_root: str = Field(default="/tmp/glamify/tryon", alias="TRYON_WORK_ROOT")
    tryon_storage_prefix: str = Field(
        default="wardrobe_output/tryon",
        alias="TRYON_STORAGE_PREFIX",
    )

    upscale_model_path: str = Field(
        default="/workspace/models/upscale/seedvr2",
        alias="UPSCALE_MODEL_PATH",
    )
    upscale_model_variant: str = Field(
        default="seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors",
        alias="UPSCALE_MODEL_VARIANT",
    )
    upscale_cli_path: str = Field(
        default="/workspace/seedvr2_eval/ComfyUI-SeedVR2_VideoUpscaler/inference_cli.py",
        alias="UPSCALE_CLI_PATH",
    )
    upscale_queue_max_size: int = Field(default=8, alias="UPSCALE_QUEUE_MAX_SIZE")
    upscale_queue_wait_timeout_seconds: int = Field(
        default=30,
        alias="UPSCALE_QUEUE_WAIT_TIMEOUT_SECONDS",
    )
    upscale_work_root: str = Field(default="/tmp/glamify/upscale", alias="UPSCALE_WORK_ROOT")
    upscale_storage_prefix: str = Field(
        default="wardrobe_output/upscale",
        alias="UPSCALE_STORAGE_PREFIX",
    )


def validate_startup_settings(settings: Settings) -> None:
    missing_fields: list[str] = []

    required_values = {
        "JWT_ACCESS_SECRET": settings.jwt_access_secret,
        "AZURE_STORAGE_CONNECTION_STRING": settings.azure_storage_connection_string,
        "AZURE_STORAGE_CONTAINER": settings.azure_storage_container,
        "AI_TOOLKIT_ROOT": settings.ai_toolkit_root,
        "QWEN_IMAGE_EDIT_MODEL_PATH": settings.qwen_image_edit_model_path,
        "WARDROBE_LORA_PATH": settings.wardrobe_lora_path,
        "WARDROBE_LORA_WEIGHT_NAME": settings.wardrobe_lora_weight_name,
        "TRYON_LORA_PATH": settings.tryon_lora_path,
        "UPSCALE_MODEL_PATH": settings.upscale_model_path,
        "UPSCALE_MODEL_VARIANT": settings.upscale_model_variant,
        "UPSCALE_CLI_PATH": settings.upscale_cli_path,
    }

    for field_name, value in required_values.items():
        if not str(value).strip():
            missing_fields.append(field_name)

    if missing_fields:
        raise RuntimeError(
            "Missing required startup configuration: " + ", ".join(missing_fields),
        )

    invalid_paths: list[str] = []
    for field_name, raw_path in {
        "AI_TOOLKIT_ROOT": settings.ai_toolkit_root,
        "QWEN_IMAGE_EDIT_MODEL_PATH": settings.qwen_image_edit_model_path,
        "TRYON_LORA_PATH": settings.tryon_lora_path,
    }.items():
        if raw_path and not Path(str(raw_path)).expanduser().exists():
            invalid_paths.append(field_name)

    if invalid_paths:
        raise RuntimeError(
            "Startup path configuration does not exist: " + ", ".join(invalid_paths),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
