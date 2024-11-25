from fastapi import FastAPI
from routers.chat import router as chat_router
from routers.knowledge import router as knowledge_router
from routers.redirect import router as redirect_router
from custom_faiss.faiss_manager_singleton import faiss_manager

faiss_manager.refresh_index()

app = FastAPI()

# Include routers
app.include_router(chat_router)
app.include_router(knowledge_router)
app.include_router(redirect_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8092, reload=True)
