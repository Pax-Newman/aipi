from typing import Literal
from fastapi import FastAPI

from pydantic import BaseModel
from enum import Enum

from .chat import router as chat_router

from sentence_transformers import SentenceTransformer

from skeletonkey import unlock, instantiate


# def get_models(model_type: str, args: Namespace) -> type:
#     path = args.model_paths.model_type
#
#     new_type = type(model_type, (str, Enum), {})
#
#     for model in os.listdir(path):
#         setattr(new_type, model, model)
#
#     return new_type
#
app = FastAPI()


app.include_router(chat_router)

@app.get('/')
async def root():
    return {'message' : 'hello world'}

# Keep most recently used model in memory to avoid loading it at each request
app.state.model = None
app.state.model_name = ""

@unlock('config.yaml')
def load_config(config):
    return config

app.state.config = load_config()

# Create a better method for loading models!
# Let's make a function first then try using 
# a config to assist loading

# --- Text Embeddings --- #

# Move these to a separate file
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
    # TODO count tokens and return error if too many
    
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


