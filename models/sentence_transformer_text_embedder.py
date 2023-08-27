
import sentence_transformers

from .interfaces import ModelConfig, TextEmbeddingModelInterface

class SentenceTransformerModelConfig(ModelConfig):
    context_length: int

class SentenceTransformerEmbedder(TextEmbeddingModelInterface):
    def __init__(self, **kwargs):
        self.config = SentenceTransformerModelConfig.model_validate(kwargs)
        self.model = sentence_transformers.SentenceTransformer(
                self.config.path,
                )

    def __call__(self, input: str | list[str], **kwargs) -> list[float] | list[list[float]]:
        embeddings = self.model.encode(input, **kwargs)
        embeddings = [embedding.tolist() for embedding in embeddings]

        return embeddings

