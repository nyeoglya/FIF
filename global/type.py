import typing as tp

JSONDict = tp.Dict[str, tp.Any]
SerializedDataForEmbeding = tp.Tuple[str, str, tp.Optional[tp.Tuple[int, int, int, int]]]
UniqueID = tp.Tuple[str, tp.Optional[str], tp.Optional[str]]
Strategy = tp.Tuple[tp.Any, tp.Any, tp.Any]

T = tp.TypeVar("T", bound=tp.Type[tp.Any])

class UserInfo(tp.TypedDict, total=False):
    name: str
    age: int
