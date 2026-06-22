from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse


class DocumentNotFoundError(HTTPException):
    def __init__(self, document_id: str):
        super().__init__(
            status_code=404,
            detail={"code": "DOCUMENT_NOT_FOUND", "document_id": document_id},
        )


class DocumentProcessingError(HTTPException):
    def __init__(self, message: str):
        super().__init__(
            status_code=422,
            detail={"code": "PROCESSING_ERROR", "message": message},
        )


class InsufficientContextError(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=200,
            detail={
                "code": "INSUFFICIENT_CONTEXT",
                "message": "Not enough relevant content found in the document.",
            },
        )


class StorageError(HTTPException):
    def __init__(self, message: str):
        super().__init__(
            status_code=500,
            detail={"code": "STORAGE_ERROR", "message": message},
        )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )
