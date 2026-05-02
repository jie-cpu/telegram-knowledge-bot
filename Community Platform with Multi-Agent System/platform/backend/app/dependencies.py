"""Application dependencies for authentication and database."""

from typing import AsyncGenerator

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.models.base import Base
from app.models.user import User  # Import to ensure table is created

# Database URL - should be configured via environment variable in production
DATABASE_URL = "sqlite+aiosqlite:///./test.db"

engine = create_async_engine(DATABASE_URL, echo=True)

async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Provide an async database session.

    Yields:
        AsyncSession: The database session.
    """
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Initialize the database by creating all tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> int:
    """Extract and validate the current user ID from the token.

    This is a simplified implementation. In production, decode and verify
    a JWT token to extract the user ID.

    Args:
        credentials: The HTTP Bearer token credentials.

    Returns:
        int: The authenticated user's ID.

    Raises:
        HTTPException 401: If the token is invalid or missing.
    """
    token = credentials.credentials
    # Placeholder: In production, decode JWT and extract user_id
    # For now, assume token is the user_id as string for testing
    try:
        user_id = int(token)
        return user_id
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
