from fastapi import FastAPI
from routers.chat import router as chat_router
from routers.knowledge import router as knowledge_router
from routers.redirect import router as redirect_router
from custom_faiss.faiss_manager_singleton import faiss_manager
from fastapi.middleware.cors import CORSMiddleware

faiss_manager.refresh_index()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your frontend's URL for better security
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Include routers
app.include_router(chat_router)
app.include_router(knowledge_router)
app.include_router(redirect_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8092, reload=True)
