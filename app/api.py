from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum
from typing import Literal

from sentence_transformers import SentenceTransformer

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

# TextEmbeddingModels = get_models('text_embedding', my_args)

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

# --- Chat Completion --- #

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionModels(str, Enum):
    examplenet = 'examplenet'

class ChatMessage(BaseModel):
    role: Literal['user', 'system', 'assistant', 'function']
    content: str
    name: str | None = None

class ChatMessageChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatCompletionRequest(BaseModel):
    model: ChatCompletionModels
    messages: list[ChatMessage]
    # function_call: str | None = None
    temperature: float = 1.0
    top_p: float = 0.9
    n: int = 1
    stream: bool = False
    stop: str | list[str] | None = None
    max_tokens: int | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logit_bias: dict[str, float] | None = None
    user: str | None = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = 'chat.completion'
    created: int
    choices: list[ChatMessageChoice]
    usage: Usage

@app.get('/chat/completions/models')
async def chat_completion_models() -> list[str]:
    return list(ChatCompletionModels)

@app.post('/chat/completions')
async def chat_completions(req: ChatCompletionRequest) -> ChatCompletionResponse:

    resp = ChatCompletionResponse(
            id = 'test',
            created = 0,
            choices = [],
            usage = Usage(
                prompt_tokens = 0,
                completion_tokens = 0,
                total_tokens = 0
                )
            )

    for i in range(req.n):
        resp.choices.append(ChatMessageChoice(
            index = i,
            message = ChatMessage(
                role = 'assistant',
                content = 'hello world',
                name = 'test'
                ),
            finish_reason = 'stop'
            ))

    return resp


