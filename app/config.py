from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    app_name: str = Field(default="Glamify AI", alias="APP_NAME")
    app_env: str = Field(default="local", alias="APP_ENV")
    debug: bool = Field(default=True, alias="APP_DEBUG")
    host: str = Field(default="0.0.0.0", alias="APP_HOST")
    port: int = Field(default=8011, alias="APP_PORT")
    log_level: str = Field(default="INFO", alias="APP_LOG_LEVEL")
    app_version: str = Field(default="0.1.0", alias="APP_VERSION")
    minicpm_service_url: str = Field( default="http://127.0.0.1:8010", alias="MINICPM_SERVICE_URL" )
    azure_storage_connection_string: str = Field( default="", alias="AZURE_STORAGE_CONNECTION_STRING" )
    azure_storage_container: str = Field(default="", alias="AZURE_STORAGE_CONTAINER")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
