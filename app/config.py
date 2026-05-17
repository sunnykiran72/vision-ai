from functools import lru_cache

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

    tryon_model_path: str = Field(default="/workspace/models/tryon/flux", alias="TRYON_MODEL_PATH")
    tryon_lora_weight_name: str = Field(default="", alias="TRYON_LORA_WEIGHT_NAME")
    tryon_lora_path: str = Field(default="", alias="TRYON_LORA_PATH")

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
    upscale_storage_prefix: str = Field(default="seedvr2", alias="UPSCALE_STORAGE_PREFIX")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
