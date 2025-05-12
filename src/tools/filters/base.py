from abc import ABC, abstractmethod

class Filter(ABC):
    @abstractmethod
    def apply(self, images: list) -> list[dict]:
        """Apply the filter to the given images."""
        pass
