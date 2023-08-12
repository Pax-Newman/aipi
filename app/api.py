from fastapi import FastAPI

from .utils import load_config

from .chat import router as chat_router
from .text_embeddings import router as text_embeddings_router

app = FastAPI()

app.include_router(chat_router)
app.include_router(text_embeddings_router)

@app.get('/')
async def root():
    return {'message' : 'hello world'}

# Keep most recently used model in memory to avoid loading it at each request
app.state.model = None
app.state.model_name = ""

app.state.config = load_config('config.yaml')

