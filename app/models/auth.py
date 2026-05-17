from __future__ import annotations

from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class AuthType(StrEnum):
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    GOOGLE = "GOOGLE"
    FACEBOOK = "FACEBOOK"
    TIKTOK = "TIKTOK"
    APPLE = "APPLE"


class AuthPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    user_id: UUID = Field(alias="userId")
    auth_type: AuthType = Field(alias="authType")
    token_id: UUID = Field(alias="token_id")
    exp: int
