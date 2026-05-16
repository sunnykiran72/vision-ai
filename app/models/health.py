from pydantic import BaseModel


class HealthMetadata(BaseModel):
    environment: str
    version: str
    python_version: str
    available_api_groups: list[str]
    configured_domains: list[str]


class HealthResponse(BaseModel):
    status: str
    service: str
    metadata: HealthMetadata
