from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse
from fastapi.encoders import jsonable_encoder
import json

from pydantic import BaseModel
from enum import Enum
from typing import Literal

from sentence_transformers import SentenceTransformer

import ctransformers

import re

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

def set_model(model_name, **kwargs):
    if app.state.model_name != model_name:
        if model_name in TextEmbeddingModels:
            app.state.model = SentenceTransformer(model_name)
        # elif model_name in ChatCompletionModels:
        #     app.state.model = Llama(model_name, **kwargs)

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
    othernet = 'othernet'

class ChatMessageRole(str, Enum):
    user = 'user'
    system = 'system'
    assistant = 'assistant'

class ChatMessage(BaseModel):
    role: ChatMessageRole
    content: str
    name: str | None = None

class ChatDelta(BaseModel):
    role: ChatMessageRole | None = None
    content: str | None = None

class ChatMessageChoice(BaseModel):
    index: int
    message: ChatMessage | ChatDelta
    finish_reason: str | None = None

class ChatCompletionRequest(BaseModel):
    model: ChatCompletionModels
    messages: list[ChatMessage]
    # function_call: str | None = None
    temperature: float = 0.8
    top_p: float = 0.9
    n: int = 1
    stream: bool = False
    stop: str | list[str] | None = None
    max_tokens: int | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 1.1
    logit_bias: dict[str, float] | None = None
    user: str | None = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal['chat.completion', 'chat.completion.chunk']
    created: int
    choices: list[ChatMessageChoice]
    usage: Usage

@app.get('/chat/completions/models')
async def chat_completion_models() -> list[str]:
    return list(ChatCompletionModels)

def utf8_is_continuation_byte(byte: int) -> bool:
    """Checks if a byte is a UTF-8 continuation byte (most significant bit is 1)."""
    return (byte & 0b10000000) != 0

def utf8_split_incomplete(seq: bytes) -> tuple[bytes, bytes]:
    """Splits a sequence of UTF-8 encoded bytes into complete and incomplete bytes."""
    i = len(seq)
    while i > 0 and utf8_is_continuation_byte(seq[i - 1]):
        i -= 1
    return seq[:i], seq[i:]

@app.post('/chat/completions')
async def chat_completions(req: ChatCompletionRequest) -> ChatCompletionResponse:
    app.state.model = ctransformers.LLM(
            model_path='/Users/paxnewman/.models/llm/7B/stablebeluga-7b.ggmlv3.q4_K_M.bin',
            model_type='llama',
            config=ctransformers.Config(
                context_length=2048,
                gpu_layers=1,
                )
            )
    model = app.state.model

    app.state.model_name = 'stablebeluga'

    cap_first = lambda x : x[0].upper() + x[1:]
    
    history = ''.join(f'### {cap_first(message.role)}: {message.content}' for message in req.messages)
    prompt = history + '\n\n### Assistant: '

    if req.stop:
        user_stops = req.stop if isinstance(req.stop, list) else [req.stop]
    else:
        user_stops = []
    stops = [f'### {cap_first(role)}:' for role in ChatMessageRole]
    stops = stops + user_stops if user_stops else stops

    resp = ChatCompletionResponse(
            id = 'test',
            created = 0,
            choices = [],
            usage = Usage(
                prompt_tokens = 0,
                completion_tokens = 0,
                total_tokens = 0
                ),
            object = 'chat.completion.chunk' if req.stream else 'chat.completion',
            )

    # Pre-tokenize prompt to get token usage & save some time
    prompt_tokens = model.tokenize(prompt)
    stop_regex = re.compile("|".join(map(re.escape, stops)))

    print(prompt)


    def stream():
        encode_stream_chunk = lambda c : json.dumps(jsonable_encoder(c))

        for i in range(req.n):
            print(f'Completion {i+1}/{req.n}')
            model.eval(prompt_tokens)

            resp.usage.prompt_tokens += len(prompt_tokens)

            resp.usage.total_tokens = resp.usage.prompt_tokens + resp.usage.completion_tokens

            resp.choices = [ChatMessageChoice(
                index = i,
                message = ChatDelta(
                    role=ChatMessageRole.assistant,
                    ),
                )]
            yield encode_stream_chunk(resp)

            token_count = 0
            text = ''
            incomplete = b""

            while True:
                token = model.sample(
                        top_p=req.top_p,
                        temperature=req.temperature,
                        repetition_penalty=req.frequency_penalty,
                        )
                
                if model.is_eos_token(token):
                    break
                model.eval([token])
                resp.usage.completion_tokens += 1
                resp.usage.total_tokens += 1

                incomplete += model.detokenize([token], decode=False)
                complete, incomplete = utf8_split_incomplete(incomplete)
                
                text += complete.decode('utf-8', errors='ignore')

                # TODO set the finish reason for the response

                if stops:
                    match = stop_regex.search(text)
                    if match:
                        text = text[: match.start()]
                        break

                longest = 0
                for s in stops:
                    for j in range(len(s), 0, -1):
                        if text.endswith(s[:j]):
                            longest = max(i, longest)
                            break
    
                end = len(text) - longest
                if end > 0:
                    resp.choices = [ChatMessageChoice(
                        index = i,
                        message = ChatDelta(
                            content = text[:end],
                            ),
                        )]
                    yield encode_stream_chunk(resp)
                    text = text[end:]
    
                token_count += 1
                if req.max_tokens and token_count >= req.max_tokens:
                    break

            model.reset()

    if req.stream:
        return EventSourceResponse(stream(), media_type='json/event-stream')

    # TODO set finish reason and token usage
    # It'll have to be done similar to streaming, so prolly include
    # streaming/non-streaming in the same function
    
    for i in range(req.n):
        resp.choices.append(ChatMessageChoice(
            index = i,
            message = ChatMessage(
                role = ChatMessageRole.assistant,
                content = model(
                    prompt,
                    max_new_tokens = req.max_tokens if req.max_tokens >= 1 else None,
                    top_p = req.top_p,
                    temperature = req.temperature,
                    repetition_penalty = req.frequency_penalty,
                    stop = stops,
                    stream = False,
                    ),
                ),
                finish_reason='stop',
            ))

    return resp


