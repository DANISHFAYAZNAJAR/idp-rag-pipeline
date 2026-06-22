from uuid import UUID

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security import decode_token
from app.db.base import get_db
from app.db.models import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


async def get_current_user(
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    token: str | None = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Auth is intentionally deferred.

    Current behavior:
    - If AUTH_ENABLED=true, validate JWT and load user.
    - Else, use X-User-Id if provided; otherwise create/get a single dev user.
    """
    if settings.auth_enabled:
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        try:
            user_id = decode_token(token)
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        result = await db.execute(select(User).where(User.id == UUID(user_id)))
        user = result.scalar_one_or_none()
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user

    if x_user_id:
        result = await db.execute(select(User).where(User.id == UUID(x_user_id)))
        user = result.scalar_one_or_none()
        if user:
            return user

    dev_email = "dev@local"
    existing = await db.execute(select(User).where(User.email == dev_email))
    user = existing.scalar_one_or_none()
    if user:
        return user

    user = User(email=dev_email, hashed_password="disabled")
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user

