from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = Field(default="local", alias="APP_ENV")

    jwt_access_secret: str = Field(default="", alias="JWT_ACCESS_SECRET")
    azure_storage_connection_string: str = Field( default="", alias="AZURE_STORAGE_CONNECTION_STRING")
    azure_storage_container: str = Field(default="", alias="AZURE_STORAGE_CONTAINER")
    
    tryon_model_path: str = Field(default="/workspace/models/tryon/flux", alias="TRYON_MODEL_PATH")
    tryon_lora_weight_name: str = Field(default="", alias="TRYON_LORA_WEIGHT_NAME")
    tryon_lora_path: str = Field(default="", alias="TRYON_LORA_PATH")

    upscale_model_path: str = Field(default="/workspace/models/upscale/seedvr2", alias="UPSCALE_MODEL_PATH")
    upscale_model_variant: str = Field(default="", alias="UPSCALE_MODEL_VARIANT")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
