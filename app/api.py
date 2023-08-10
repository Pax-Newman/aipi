from typing import Literal
from fastapi import FastAPI

from pydantic import BaseModel
from enum import Enum

from .chat import router as chat_router
from .utils import load_config

from sentence_transformers import SentenceTransformer

from skeletonkey import unlock, instantiate

app = FastAPI()

app.include_router(chat_router)

@app.get('/')
async def root():
    return {'message' : 'hello world'}

# Keep most recently used model in memory to avoid loading it at each request
app.state.model = None
app.state.model_name = ""

app.state.config = load_config('config.yaml')

