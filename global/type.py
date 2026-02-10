import typing as tp

JSONDict = tp.Dict[str, tp.Any]

class UserInfo(tp.TypedDict, total=False):
    name: str
    age: int
