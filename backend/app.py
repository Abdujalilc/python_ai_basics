from fastapi import FastAPI
from controllers.chat import router as chat_router
from controllers.knowledge import router as knowledge_router
from controllers.redirect import router as redirect_router

app = FastAPI()

# Include routers
app.include_router(chat_router)
app.include_router(knowledge_router)
app.include_router(redirect_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8092, reload=True)
