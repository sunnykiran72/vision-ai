from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.constants import tryon as tryon_constants

TRYON_SPECIALIST_KEYS = ("top", "bottom", "dress", "multi")
RESIDENT_RUNTIME_KEYS = ("wardrobe", "tryon", "upscale")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = Field(default="local", alias="APP_ENV")
    resident_runtimes: str = Field(
        default="wardrobe,tryon,upscale",
        alias="RESIDENT_RUNTIMES",
    )

    # Single process-wide GPU execution queue shared by every GPU-backed feature.
    system_queue_max_size: int = Field(default=8, alias="SYSTEM_QUEUE_MAX_SIZE")
    system_queue_wait_timeout_seconds: int = Field(
        default=30,
        alias="SYSTEM_QUEUE_WAIT_TIMEOUT_SECONDS",
    )

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
    qwen_image_edit_dtype: str = Field(default="bfloat16", alias="QWEN_IMAGE_EDIT_DTYPE")
    qwen_compile: bool = Field(default=False, alias="QWEN_COMPILE")

    wardrobe_lora_top_path: str = Field(default="", alias="WARDROBE_LORA_TOP_PATH")
    wardrobe_lora_bottom_path: str = Field(default="", alias="WARDROBE_LORA_BOTTOM_PATH")
    wardrobe_lora_dress_path: str = Field(default="", alias="WARDROBE_LORA_DRESS_PATH")
    # MiniCPM-V garment captioner, loaded in-process via vLLM inside this service.
    # Model, dtype, memory, and request-shape knobs are environment-specific.
    minicpm_model_path: str = Field(
        default="openbmb/MiniCPM-V-4_5",
        alias="MINICPM_MODEL_PATH",
    )
    minicpm_dtype: str = Field(default="bfloat16", alias="MINICPM_DTYPE")
    minicpm_gpu_memory_utilization: float = Field(
        default=0.27,
        alias="MINICPM_GPU_MEMORY_UTILIZATION",
    )
    minicpm_kv_cache_dtype: str = Field(default="fp8", alias="MINICPM_KV_CACHE_DTYPE")
    minicpm_calculate_kv_scales: bool = Field(
        default=True,
        alias="MINICPM_CALCULATE_KV_SCALES",
    )
    minicpm_attention_backend: str = Field(
        default="",
        alias="MINICPM_ATTENTION_BACKEND",
    )
    minicpm_max_tokens: int = Field(default=100, alias="MINICPM_MAX_TOKENS")
    minicpm_max_model_len: int = Field(default=2048, alias="MINICPM_MAX_MODEL_LEN")
    minicpm_max_slice_nums: int = Field(default=6, alias="MINICPM_MAX_SLICE_NUMS")
    minicpm_resize_long_px: int = Field(default=1024, alias="MINICPM_RESIZE_LONG_PX")
    minicpm_enforce_eager: bool = Field(default=True, alias="MINICPM_ENFORCE_EAGER")
    # Separate private Azure containers for wardrobe input and output images.
    azure_wardrobe_input_container: str = Field(
        default="wardrobe-inputs",
        alias="AZURE_WARDROBE_INPUT_CONTAINER",
    )
    azure_wardrobe_output_container: str = Field(
        default="wardrobe-outputs",
        alias="AZURE_WARDROBE_OUTPUT_CONTAINER",
    )
    glamify_api_base_url: str = Field(default="", alias="GLAMIFY_API_BASE_URL")

    tryon_lora_rank: int = Field(
        default=tryon_constants.DEFAULT_LORA_RANK,
        alias="TRYON_LORA_RANK",
    )
    tryon_lora_alpha: int = Field(
        default=tryon_constants.DEFAULT_LORA_ALPHA,
        alias="TRYON_LORA_ALPHA",
    )
    tryon_lora_scale: float = Field(
        default=tryon_constants.DEFAULT_LORA_SCALE,
        alias="TRYON_LORA_SCALE",
    )

    tryon_enabled_specialists: str = Field(
        default="top,bottom,dress,multi",
        alias="TRYON_ENABLED_SPECIALISTS",
    )
    tryon_lora_top_path: str = Field(default="", alias="TRYON_LORA_TOP_PATH")
    tryon_lora_bottom_path: str = Field(default="", alias="TRYON_LORA_BOTTOM_PATH")
    tryon_lora_dress_path: str = Field(default="", alias="TRYON_LORA_DRESS_PATH")
    tryon_lora_multi_path: str = Field(default="", alias="TRYON_LORA_MULTI_PATH")
    tryon_prompt_trigger_top: str = Field(
        default=tryon_constants.PROMPT_TRIGGER_TOP,
        alias="TRYON_PROMPT_TRIGGER_TOP",
    )
    tryon_prompt_trigger_bottom: str = Field(
        default=tryon_constants.PROMPT_TRIGGER_BOTTOM,
        alias="TRYON_PROMPT_TRIGGER_BOTTOM",
    )
    tryon_prompt_trigger_dress: str = Field(
        default=tryon_constants.PROMPT_TRIGGER_DRESS,
        alias="TRYON_PROMPT_TRIGGER_DRESS",
    )
    tryon_prompt_trigger_multi: str = Field(
        default=tryon_constants.PROMPT_TRIGGER_MULTI,
        alias="TRYON_PROMPT_TRIGGER_MULTI",
    )
    tryon_prompt_identity_clause: str = Field(
        default=tryon_constants.IDENTITY_CLAUSE,
        alias="TRYON_PROMPT_IDENTITY_CLAUSE",
    )

    tryon_default_seed: int = Field(
        default=tryon_constants.DEFAULT_SEED,
        alias="TRYON_DEFAULT_SEED",
    )
    tryon_default_steps: int = Field(
        default=tryon_constants.DEFAULT_STEPS,
        alias="TRYON_DEFAULT_STEPS",
    )
    tryon_default_guidance_scale: float = Field(
        default=tryon_constants.DEFAULT_GUIDANCE_SCALE,
        alias="TRYON_DEFAULT_GUIDANCE_SCALE",
    )
    tryon_storage_prefix: str = Field(
        default=tryon_constants.STORAGE_PREFIX,
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
    enabled_runtimes = get_enabled_resident_runtimes(settings)

    required_values = {
        "JWT_ACCESS_SECRET": settings.jwt_access_secret,
        "AZURE_STORAGE_CONNECTION_STRING": settings.azure_storage_connection_string,
        "AZURE_STORAGE_CONTAINER": settings.azure_storage_container,
    }
    path_fields: dict[str, str] = {}

    if "wardrobe" in enabled_runtimes:
        # Wardrobe extraction runs on the diffusers QwenImageEditPlus backend, with MiniCPM
        # captioning delegated to the vLLM webapp and input/output stored in Azure.
        required_values.update(
            {
                "QWEN_IMAGE_EDIT_MODEL_PATH": settings.qwen_image_edit_model_path,
                "WARDROBE_LORA_TOP_PATH": settings.wardrobe_lora_top_path,
                "WARDROBE_LORA_BOTTOM_PATH": settings.wardrobe_lora_bottom_path,
                "WARDROBE_LORA_DRESS_PATH": settings.wardrobe_lora_dress_path,
                "MINICPM_MODEL_PATH": settings.minicpm_model_path,
                "AZURE_WARDROBE_INPUT_CONTAINER": settings.azure_wardrobe_input_container,
                "AZURE_WARDROBE_OUTPUT_CONTAINER": settings.azure_wardrobe_output_container,
                "GLAMIFY_API_BASE_URL": settings.glamify_api_base_url,
            },
        )
        path_fields.update(
            {
                "QWEN_IMAGE_EDIT_MODEL_PATH": settings.qwen_image_edit_model_path,
                "WARDROBE_LORA_TOP_PATH": settings.wardrobe_lora_top_path,
                "WARDROBE_LORA_BOTTOM_PATH": settings.wardrobe_lora_bottom_path,
                "WARDROBE_LORA_DRESS_PATH": settings.wardrobe_lora_dress_path,
            },
        )

    if "tryon" in enabled_runtimes:
        # Try-on shares the resident diffusers QwenImageEditPlus backend with wardrobe.
        required_values["QWEN_IMAGE_EDIT_MODEL_PATH"] = settings.qwen_image_edit_model_path
        path_fields["QWEN_IMAGE_EDIT_MODEL_PATH"] = settings.qwen_image_edit_model_path
        enabled_specialists = get_enabled_tryon_specialists(settings)
        specialist_paths = {
            f"TRYON_LORA_{specialist.upper()}_PATH": _tryon_specialist_path(
                settings,
                specialist,
            )
            for specialist in enabled_specialists
        }
        required_values.update(specialist_paths)
        path_fields.update(specialist_paths)

    if "upscale" in enabled_runtimes:
        required_values.update(
            {
                "UPSCALE_MODEL_PATH": settings.upscale_model_path,
                "UPSCALE_MODEL_VARIANT": settings.upscale_model_variant,
                "UPSCALE_CLI_PATH": settings.upscale_cli_path,
            },
        )

    for field_name, value in required_values.items():
        if not str(value).strip():
            missing_fields.append(field_name)

    if missing_fields:
        raise RuntimeError(
            "Missing required startup configuration: " + ", ".join(sorted(set(missing_fields))),
        )

    invalid_paths: list[str] = []
    for field_name, raw_path in path_fields.items():
        if raw_path and not Path(str(raw_path)).expanduser().exists():
            invalid_paths.append(field_name)

    if invalid_paths:
        raise RuntimeError(
            "Startup path configuration does not exist: " + ", ".join(sorted(set(invalid_paths))),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def get_enabled_tryon_specialists(settings: Settings) -> tuple[str, ...]:
    raw_values = [
        value.strip().lower()
        for value in str(settings.tryon_enabled_specialists or "").split(",")
        if value.strip()
    ]
    if not raw_values:
        raise RuntimeError("TRYON_ENABLED_SPECIALISTS must include at least one specialist.")
    invalid = sorted(set(raw_values) - set(TRYON_SPECIALIST_KEYS))
    if invalid:
        raise RuntimeError(
            "Invalid TRYON_ENABLED_SPECIALISTS values: "
            + ", ".join(invalid)
            + ". Expected any of: "
            + ", ".join(TRYON_SPECIALIST_KEYS),
        )
    return tuple(dict.fromkeys(raw_values))


def get_enabled_resident_runtimes(settings: Settings) -> tuple[str, ...]:
    raw_values = [
        value.strip().lower()
        for value in str(settings.resident_runtimes or "").split(",")
        if value.strip()
    ]
    if not raw_values:
        raise RuntimeError("RESIDENT_RUNTIMES must include at least one runtime.")
    invalid = sorted(set(raw_values) - set(RESIDENT_RUNTIME_KEYS))
    if invalid:
        raise RuntimeError(
            "Invalid RESIDENT_RUNTIMES values: "
            + ", ".join(invalid)
            + ". Expected any of: "
            + ", ".join(RESIDENT_RUNTIME_KEYS),
        )
    return tuple(dict.fromkeys(raw_values))


def _tryon_specialist_path(settings: Settings, specialist: str) -> str:
    if specialist == "top":
        return settings.tryon_lora_top_path
    if specialist == "bottom":
        return settings.tryon_lora_bottom_path
    if specialist == "dress":
        return settings.tryon_lora_dress_path
    if specialist == "multi":
        return settings.tryon_lora_multi_path
    raise RuntimeError(f"Unknown try-on specialist: {specialist}")
