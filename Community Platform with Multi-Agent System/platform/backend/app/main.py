"""Main application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.profiles import router as profile_router
from app.dependencies import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events.

    Args:
        app: The FastAPI application instance.
    """
    await init_db()
    yield


app = FastAPI(
    title="User Profile API",
    description="API for managing user profiles",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(profile_router)


@app.get("/health", tags=["health"])
async def health_check() -> dict:
    """Health check endpoint.

    Returns:
        dict: A simple status message.
    """
    return {"status": "healthy"}
