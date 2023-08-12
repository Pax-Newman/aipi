from fastapi import FastAPI, APIRouter, Depends

from pydantic import BaseModel
from enum import Enum
from typing import Literal

from .utils import get_app_instance, set_model, inject_models_to_enum, load_config

from models.interfaces import TextEmbeddingModelInterface

router = APIRouter()

class TextEmbeddingModels(str, Enum):
    inject_models_to_enum(locals(),
        load_config('config.yaml').models.text_embedding
        )

# --- Models --- #

class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int

class TextEmbeddingRequest(BaseModel):
    model: TextEmbeddingModels
    input: str | list[str]
    user: str | None = None

class Embedding(BaseModel):
    index: int
    object: str = 'embedding'
    data: list[float]

class TextEmbeddingResponse(BaseModel):
    model: TextEmbeddingModels
    object: str = 'list'
    data: list[Embedding]
    usage: Usage

# --- Routes --- #

@router.get('/embedding/models')
async def text_embedding_models() -> list[TextEmbeddingModels]:
    return list(TextEmbeddingModels)

@router.post('/embedding')
async def embed_text(
        req: TextEmbeddingRequest,
        app: FastAPI = Depends(get_app_instance)
        ) -> TextEmbeddingResponse:
    
    model: TextEmbeddingModelInterface = set_model(app, req.model, 'text_embedding')

    resp = TextEmbeddingResponse(
            model=req.model,
            data = [],
            usage = Usage(
                prompt_tokens = 0,
                total_tokens = 0
                ),
            )

    if isinstance(req.input, str):
        resp.data.append(
                Embedding(
                    index = 0,
                    data = model(req.input),
                )
            )
    elif isinstance(req.input, list):
        embeddings = [
                Embedding(
                    index = i,
                    data = embedding,
                    )
                for i, embedding in enumerate(model(req.input))
            ]
        resp.data.extend(embeddings)
    return resp

