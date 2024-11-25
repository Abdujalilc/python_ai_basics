from fastapi import APIRouter
from fastapi.responses import RedirectResponse

router = APIRouter()

@router.get("/{path:path}")
async def redirect_to_docs():
    return RedirectResponse(url="/docs")