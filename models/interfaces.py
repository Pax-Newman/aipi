from typing import Any, Generator
from pydantic import BaseModel

class ModelConfig(BaseModel):
    name: str
    path: str

class ModelInterface():
    def __init__(self, **kwargs):
        self.config = kwargs
        self.model = None
    def __call__(self, input: Any, **kwargs) -> Any:
        ...

class TextCompletionModelInterface(ModelInterface):
    def __call__(self, input: str, **kwargs) -> tuple[str, str] | Generator[tuple[str, str | None], Any, Any]:
        ...

class TextEmbeddingModelInterface(ModelInterface):
    def __call__(self, input: str | list[str], **kwargs) -> list[float] | list[list[float]]:
        ...

