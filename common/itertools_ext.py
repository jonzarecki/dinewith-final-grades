import itertools
from typing import List, TypeVar, Union

_S = TypeVar("_S")


def flatten(lst: List[Union[List[_S], _S]]) -> List[_S]:
    return list(itertools.chain.from_iterable([(i if isinstance(i, list) else [i]) for i in lst]))
