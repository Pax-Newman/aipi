from fastapi import FastAPI, Request, APIRouter, Depends
from sse_starlette.sse import EventSourceResponse
from fastapi.encoders import jsonable_encoder

from pydantic import BaseModel, Field
from enum import Enum
from typing import Literal

import re
import json

import ctransformers

router = APIRouter()

# --- Utils --- #

def utf8_is_continuation_byte(byte: int) -> bool:
    """Checks if a byte is a UTF-8 continuation byte (most significant bit is 1)."""
    return (byte & 0b10000000) != 0

def utf8_split_incomplete(seq: bytes) -> tuple[bytes, bytes]:
    """Splits a sequence of UTF-8 encoded bytes into complete and incomplete bytes."""
    i = len(seq)
    while i > 0 and utf8_is_continuation_byte(seq[i - 1]):
        i -= 1
    return seq[:i], seq[i:]

def get_app_instance(request: Request) -> FastAPI:
    return request.app

def generate_text(
        model: ctransformers.LLM,
        tokens: list[int],
        stops: list[str],
        top_p: float = 0.9,
        temperature: float = 0.8,
        max_tokens: int | None = None,
        repetition_penalty: float = 1.1,
        ):

    # Ingest tokens
    model.eval(tokens)

    stop_regex = re.compile("|".join(map(re.escape, stops)))

    count = 0
    finish_reason = None
    text = ''
    incomplete = b''

    while True:
        token = model.sample(
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                )

        # finish reason eos
        if model.is_eos_token(token):
            finish_reason = 'stop'
            break

        # handle incomplete utf-8 multi-byte characters
        incomplete += model.detokenize([token], decode=False)
        complete, incomplete = utf8_split_incomplete(incomplete)

        text += complete.decode('utf-8', errors='ignore')

        if stops:
            match = stop_regex.search(text)
            if match:
                text = text[:match.start()]
                finish_reason = 'stop'
                break

        # get the length of the longest stop prefix that is at the end of the text
        longest = 0
        for stop in stops:
            for i in range(len(stop), 0, -1):
                if text.endswith(stop[:i]):
                    longest = max(longest, i)
                    break

        # text[:end] is the text without the stop
        end = len(text) - longest
        if end > 0:
            yield text[:end], finish_reason
            # save the rest of the text incase the stop prefix doesn't generate a full stop
            text = text[end:]

        count += 1
        if max_tokens and count >= max_tokens:
            finish_reason = 'length'
            break

        model.eval([token])

    yield text, finish_reason


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
    stream_completions: bool = Field(False, alias='stream')
    stop: str | list[str] | None = None
    max_tokens: int | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 1.1
    logit_bias: dict[str, float] | None = None
    chat_user: str | None = Field(None, alias='user')

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
    app.state.model = ctransformers.LLM(
            model_path='/Users/paxnewman/.models/llm/7B/stablebeluga-7b.ggmlv3.q4_K_M.bin',
            model_type='llama',
            config=ctransformers.Config(
                context_length=2048,
                gpu_layers=1,
                )
            )
    app.state.model_name = 'stablebeluga'
    model = app.state.model

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
            object = 'chat.completion.chunk' if req.stream_completions else 'chat.completion',
            )

    # Pre-tokenize prompt to get token usage & save some time
    prompt_tokens = model.tokenize(prompt)

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

            for token, finish_reason in generate_text(
                    model,
                    prompt_tokens,
                    stops
                    ):
                resp.choices = [ChatChunk(
                    index=choice,
                    delta=ChatDelta(content=token, role=ChatMessageRole.assistant),
                    finish_reason=finish_reason,
                    )]
                resp.usage.completion_tokens += 1
                resp.usage.total_tokens += 1
                yield encode_stream_chunk(resp)
            model.reset()

    if req.stream_completions:
        return EventSourceResponse(stream(), media_type='json/event-stream')

    # Create non-streamed completions
    for choice in range(req.n):
        completion = ''
        finish_reason = None
        tokens = 0
        for token, reason in generate_text(
                model,
                prompt_tokens,
                stops
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


