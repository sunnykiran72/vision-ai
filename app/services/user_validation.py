from app.models.user_validation import UserValidationResponse


def build_user_validation_placeholder() -> UserValidationResponse:
    return UserValidationResponse(
        status="not_implemented",
        message="User validation flow scaffold is ready. Implementation will be added next.",
        feature="user_validation",
    )
