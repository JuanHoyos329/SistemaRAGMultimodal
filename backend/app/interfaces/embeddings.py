from abc import ABC, abstractmethod
from typing import List

class EmbeddingsService(ABC):
    
    @abstractmethod
    def get_text_embedding(self, text: str) -> List[float]:
        """Genera el vector embedding para un texto."""
        pass

    def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Implementations may override this to batch model inference."""
        return [self.get_text_embedding(text) for text in texts]

    def get_image_embedding(self, image_path: str) -> List[float]:
        """Optional capability for providers with a shared text/image embedding space."""
        raise NotImplementedError("This embedding provider does not support image vectors")
