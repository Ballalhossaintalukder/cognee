"""
Custom exceptions for the Cognee API.

This module defines a set of exceptions for handling various application errors,
such as service failures, resource conflicts, and invalid operations.
"""

from .exceptions import (
    CogneeApiError,
    ServiceError,
    EntityNotFoundError,
    GroupNotFoundError,
    UserNotFoundError,
    EntityAlreadyExistsError,
    InvalidOperationError,
    PermissionDeniedError,
    IngestionError,
    InvalidValueError,
    InvalidAttributeError,
)