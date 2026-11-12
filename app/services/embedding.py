from __future__ import annotations
import math
import os
from app.config import settings

class EmbeddingProviderError(RuntimeError):
    pass

class OpenAIEmbeddingProvider:
    def __init__(self):
        if not settings.openai_api_key:
            raise EmbeddingProviderError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise EmbeddingProviderError("Install the optional dependency: pip install -e '.[openai]'") from exc
        self.client=OpenAI(api_key=settings.openai_api_key)
        self.model=settings.embedding_model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response=self.client.embeddings.create(model=self.model,input=texts)
        return [item.embedding for item in sorted(response.data,key=lambda x:x.index)]


def get_embedding_provider():
    if settings.embedding_provider=="openai":
        return OpenAIEmbeddingProvider()
    return None


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left)!=len(right) or not left:
        return 0.0
    dot=sum(a*b for a,b in zip(left,right))
    nl=math.sqrt(sum(a*a for a in left)); nr=math.sqrt(sum(b*b for b in right))
    if not nl or not nr:
        return 0.0
    return dot/(nl*nr)
