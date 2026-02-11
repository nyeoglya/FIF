import typing as tp

from ._component import Component

class ComponentFactory:
    _registry: tp.Dict[str, tp.Type[Component]] = {}

    def __init__(self) -> None:
        self._instance_name: str = "ComponentFactory"

    @classmethod
    def register(cls, modality_value: str, implementation: tp.Type[Component]) -> None:
        cls._registry[modality_value] = implementation

    def create(self, modality: str, **kwargs: tp.Any) -> Component:
        if modality not in self._registry:
            raise ValueError(
                f"Unknown modality: {modality}. "
                f"Available modalities: {list(self._registry.keys())}"
            )

        implementation_class = self._registry[modality]
        return implementation_class(**kwargs)
