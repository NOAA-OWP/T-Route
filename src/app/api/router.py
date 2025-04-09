"""Author: Tadd Bindas"""

from fastapi import APIRouter

from app.api.routes import troute_v4

api_router = APIRouter()

# The following "router" is an entrypoint used to build/call T-route to do hydrologic routing
api_router.include_router(
    troute_v4.router, prefix="/flow_routing/v4", tags=["Troute v4 Flow Routing"]
)
