import typing as tp

JSONDict = tp.Dict[str, tp.Any]
Type_DocName = str
Type_NID = tp.Tuple[Type_DocName, str]  # (filename, component_id)
T = tp.TypeVar("T", bound=tp.Type[tp.Any])

class UserInfo(tp.TypedDict, total=False):
    name: str
    age: int
