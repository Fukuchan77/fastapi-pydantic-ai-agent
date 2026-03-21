"""V1 API router - aggregates agent and RAG sub-routers."""

from fastapi import APIRouter

from app.api.v1 import agent
from app.api.v1 import rag


# Create main v1 router
router = APIRouter()

# Include sub-routers
router.include_router(agent.router)
router.include_router(rag.router)
