from fastapi import APIRouter
from fastapi.responses import FileResponse
import os

router = APIRouter()

frontend_path = os.path.abspath("../frontend")

@router.get("/")
async def redirect_to_index():
    return FileResponse(os.path.join(frontend_path, "index.html"))