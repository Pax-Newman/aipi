from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum

from sentence_transformers import SentenceTransformer


app = FastAPI()

@app.get('/')
async def root():
    return {'message' : 'hello world'}

# Keep most recently used model in memory to avoid loading it at each request
app.state.model = None
app.state.model_name = ""

def set_model(model_name):
    if app.state.model_name != model_name:
        if model_name in TextEmbeddingModels:
            app.state.model = SentenceTransformer(model_name)

        app.state.model_name = model_name

# --- Text Embeddings --- #

class TextEmbeddingModels(str, Enum):
    all_minilm_l6_v2 = 'all-MiniLM-L6-v2'

class TextEmbeddingRequest(BaseModel):
    model: TextEmbeddingModels
    input: str | list[str]

@app.get('/embeddings/text/models')
async def text_embedding_models():
    return list(TextEmbeddingModels)

@app.post('/embeddings/text')
async def text_embeddings(req: TextEmbeddingRequest):
    # Create Model
    set_model(req.model)

    # Embed text
    features = app.state.model.encode(req.input)

    # Convert embeddings to a list of floats
    if isinstance(features, list):
        data = [feature.tolist() for feature in features]
    else:
        data = features.tolist()

    return data


