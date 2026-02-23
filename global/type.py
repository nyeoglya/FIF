import typing as tp

JSONDict = tp.Dict[str, tp.Any]
SerializedData = tp.Tuple[str, tp.List[str]]
SerializedDataForEmbeding = tp.Tuple[str, str, tp.Optional[tp.Tuple[int, int, int, int]]]
UniqueID = tp.Tuple[str, tp.Optional[str], tp.Optional[str]]
Strategy = tp.Tuple[str, str, str] # document_search_mode, component_search_mode, vector_granularity

T = tp.TypeVar("T", bound=tp.Type[tp.Any])

class UserInfo(tp.TypedDict, total=False):
    name: str
    age: int
