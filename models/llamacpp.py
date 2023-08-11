from typing import Any, Generator
import re

import ctransformers

from .interfaces import ModelConfig, TextCompletionModelInterface


class LlamaCPPModelConfig(ModelConfig):
    type: str
    context_length: int
    gpu_layers: int

class LlamaCPPModel(TextCompletionModelInterface):
    def __init__(self, **kwargs):
        self.config = LlamaCPPModelConfig.model_validate(kwargs)
        self.model = ctransformers.LLM(
                self.config.path,
                self.config.type,
                config=ctransformers.Config(
                    context_length=self.config.context_length,
                    gpu_layers=self.config.gpu_layers,
                    )
                )
        print(self.model.config)

    def _utf8_is_continuation_byte(self, byte: int) -> bool:
        """Checks if a byte is a UTF-8 continuation byte (most significant bit is 1)."""
        return (byte & 0b10000000) != 0
    
    def _utf8_split_incomplete(self, seq: bytes) -> tuple[bytes, bytes]:
        """Splits a sequence of UTF-8 encoded bytes into complete and incomplete bytes."""
        i = len(seq)
        while i > 0 and self._utf8_is_continuation_byte(seq[i - 1]):
            i -= 1
        return seq[:i], seq[i:]

    def _generate_text(
            self,
            tokens: list[int],
            stops: list[str],
            top_p: float = 0.9,
            temperature: float = 0.8,
            max_tokens: int | None = None,
            repetition_penalty: float = 1.1,
            ) -> tuple[str, str] | Generator[tuple[str, str | None], Any, Any]:

        # Ingest tokens
        self.model.eval(tokens)
    
        stop_regex = re.compile("|".join(map(re.escape, stops)))
    
        count = 0
        finish_reason = None
        text = ''
        incomplete = b''
    
        while True:
            token = self.model.sample(
                    top_p=top_p,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    )
    
            # finish reason eos
            if self.model.is_eos_token(token):
                finish_reason = 'stop'
                break
    
            # handle incomplete utf-8 multi-byte characters
            incomplete += self.model.detokenize([token], decode=False)
            complete, incomplete = self._utf8_split_incomplete(incomplete)
    
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
    
            self.model.eval([token])
        yield text, finish_reason
        self.model.reset()

    def __call__(self, input: str, **kwargs) -> tuple[str, str] | Generator[tuple[str, str | None], Any, Any]:
        return self._generate_text(self.model.tokenize(input), **kwargs)


