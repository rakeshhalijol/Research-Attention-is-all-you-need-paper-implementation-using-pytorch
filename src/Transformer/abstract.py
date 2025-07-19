from abc import abstractmethod, ABC


class Runnable(ABC):
    @abstractmethod
    def forward(self, *args, **kwargs): pass
