from fastapi import FastAPI, Request, APIRouter, Depends
from sse_starlette.sse import EventSourceResponse
from fastapi.encoders import jsonable_encoder

from pydantic import BaseModel
from enum import Enum
from typing import Literal

import json

from .models import LlamaCPPModel, LlamaCPPModelConfig, ModelConfig, ModelInterface
from .utils import set_model, get_app_instance

router = APIRouter()

# It might be good to put other text completion routes in here

# --- Utils --- #

# Should probably move most of these into a general utils file



# --- Models --- #

class ChatCompletionModels(str, Enum):
    examplenet = 'examplenet'
    othernet = 'othernet'

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatMessageRole(str, Enum):
    user = 'user'
    system = 'system'
    assistant = 'assistant'

class ChatMessage(BaseModel):
    role: ChatMessageRole
    content: str
    name: str | None = None

class ChatMessageChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str | None = None

class ChatDelta(BaseModel):
    role: ChatMessageRole | None = None
    content: str | None = None

class ChatChunk(BaseModel):
    index: int
    delta: ChatDelta
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
    choices: list[ChatMessageChoice | ChatChunk]
    usage: Usage

# --- Routes --- #

@router.get('/chat/completions/models')
async def chat_completion_models() -> list[ChatCompletionModels]:
    return list(ChatCompletionModels)

@router.post('/v1/chat/completions')
@router.post('/chat/completions')
async def chat_completions(req: ChatCompletionRequest, app: FastAPI = Depends(get_app_instance)) -> ChatCompletionResponse:

    model = set_model(app, 'stablebeluga', 'completion')

    # Create chat history prompt
    cap_first = lambda x : x[0].upper() + x[1:]
    
    history = ''.join(f'### {cap_first(message.role)}: {message.content}' for message in req.messages)
    prompt = history + '\n\n### Assistant: '

    # Create chat stops
    if req.stop:
        user_stops = req.stop if isinstance(req.stop, list) else [req.stop]
    else:
        user_stops = []
    stops = [f'### {cap_first(role)}:' for role in ChatMessageRole]
    stops = stops + user_stops if user_stops else stops

    # Define response outline
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
    prompt_tokens = model.model.tokenize(prompt)

    resp.usage.prompt_tokens = len(prompt_tokens)
    resp.usage.total_tokens += len(prompt_tokens)
    
    # Stream completions
    def stream():
        encode_stream_chunk = lambda c : json.dumps(jsonable_encoder(c))
        for choice in range(req.n):
            # Send initial message to indicate role
            resp.choices = [ChatChunk(
                index=choice,
                delta=ChatDelta(role=ChatMessageRole.assistant),
                )]
            yield resp

            for token, finish_reason in model(
                    prompt,
                    stops=stops,
                    ):
                resp.choices = [ChatChunk(
                    index=choice,
                    delta=ChatDelta(content=token, role=ChatMessageRole.assistant),
                    finish_reason=finish_reason,
                    )]
                resp.usage.completion_tokens += 1
                resp.usage.total_tokens += 1
                yield encode_stream_chunk(resp)

    if req.stream:
        return EventSourceResponse(stream(), media_type='json/event-stream')

    # Create non-streamed completions
    for choice in range(req.n):
        completion = ''
        finish_reason = None
        tokens = 0
        
        for token, reason in model(
                prompt,
                stops=stops,
                ):
            completion += token
            finish_reason = reason
            tokens += 1

        resp.choices.append(ChatMessageChoice(
            index = choice,
                message = ChatMessage(
                role = ChatMessageRole.assistant,
                content = completion,
                ),
            finish_reason = finish_reason,
            ))
        resp.usage.completion_tokens += tokens

    resp.usage.total_tokens += resp.usage.completion_tokens

    return resp


