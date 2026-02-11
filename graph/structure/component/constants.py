import typing as tp
from enum import Enum, unique
from type import T

def add_enum_utilities(cls: T) -> T:
    @classmethod
    def from_value(cls_: T, value: tp.Optional[str]) -> tp.Optional[tp.Any]:
        if value is None: return None
        try: return cls_(value)
        except ValueError: raise ValueError(f"No {cls_.__name__} with value: {value}") from None
    
    def __str__(self: tp.Any) -> str:
        return str(self.value)

    cls.from_value = from_value # type: ignore
    cls.__str__ = __str__ # type: ignore
    
    return cls

@unique
@add_enum_utilities
class ParsedWebKeywords(Enum):
    TITLE = "title"
    HIERARCHY = "hierarchy"
    ID_SEQUENCE = "id_sequence"
    ID_TO_COMPONENT = "id_to_component"
    ID_TO_HTML = "id_to_html"

@unique
@add_enum_utilities
class Modality(Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    SENTENCE = "sentence"
    TABLE_SEGMENT = "table_segment"
    SUBIMAGE = "subimage"

    @classmethod
    def get_all_modalities(cls):
        return [
            Modality.TEXT,
            Modality.TABLE,
            Modality.IMAGE,
            Modality.SENTENCE,
            Modality.TABLE_SEGMENT,
            Modality.SUBIMAGE
        ]

@unique
@add_enum_utilities
class EmbeddingMode(Enum):
    IMAGE = "image"
    SUMMARY = "summary"
    OCR = "ocr"

@unique
@add_enum_utilities
class LCGLayer(Enum):
    DOC = 0
    COMP = 1
    SUBCOMP = 2
