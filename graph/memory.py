import typing as tp
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from type import Strategy, UniqueID

@dataclass
class FiFMemoryUnit:
    def __init__(self) -> None:
        self.subquery_list: tp.List[str]
        self.subquery_embeddings: NDArray[np.float32]
        self.subquery_index: int
        self.strategy: Strategy
        self.component_id_list: tp.List[UniqueID]

class FiFMemory:
    def __init__(self) -> None:
        self.query: str = ""
        self.history: tp.List[FiFMemoryUnit] = []

    def get_last_unit(self) -> FiFMemoryUnit:
        assert self.history
        return self.history[-1]
    
    def reset_memory(self, query: str) -> None:
        self.query = query
        self.history = []

    def add_new_history(self, memory_unit: FiFMemoryUnit):
        self.history.append(memory_unit)
