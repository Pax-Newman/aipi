
import ctransformers

from .interfaces import ModelConfig, TextEmbeddingModelInterface

class LlamaCPPEmbedderModelConfig(ModelConfig):
    type: str
    context_length: int
    gpu_layers: int

class LlamaCPPEmbedder(TextEmbeddingModelInterface):
    def __init__(self, **kwargs):
        self.config = LlamaCPPEmbedderModelConfig.model_validate(kwargs)
        self.model = ctransformers.LLM(
                self.config.path,
                self.config.type,
                config=ctransformers.Config(
                    context_length=self.config.context_length,
                    gpu_layers=self.config.gpu_layers,
                    )
                )

    def __call__(self, input: str | list[str], **kwargs) -> list[float] | list[list[float]]:
        
        # TODO find a way to report token usage

        if isinstance(input, str):
            tokens = self.model.tokenize(input)

            return self.model.embed(tokens)

        else:
            token_sets = [self.model.tokenize(text) for text in input]
            embeddings = []
            for token_set in token_sets:
                self.model.eval(token_set)
                embeddings.append(self.model.embed(token_set))

            return embeddings


