from pydantic import BaseModel


class RuntimeQueueMetadata(BaseModel):
    active_jobs: int
    waiting_jobs: int
    max_queue_size: int


class RuntimeRunnerMetadata(BaseModel):
    loaded: bool
    backend: str | None


class RuntimeMetadata(BaseModel):
    runner: RuntimeRunnerMetadata
    queue: RuntimeQueueMetadata


class HealthMetadata(BaseModel):
    environment: str
    version: str
    python_version: str
    available_api_groups: list[str]
    configured_domains: list[str]
    runtimes: dict[str, RuntimeMetadata]


class HealthResponse(BaseModel):
    status: str
    service: str
    metadata: HealthMetadata
