from pydantic import BaseModel


class HealthMetadata(BaseModel):
    environment: str
    version: str
    python_version: str
    host: str
    port: int
    configured_services: list[str]
    available_api_groups: list[str]


class HealthResponse(BaseModel):
    status: str
    service: str
    metadata: HealthMetadata

