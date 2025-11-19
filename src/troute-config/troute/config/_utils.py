from contextlib import contextmanager
from pydantic import BaseModel
import inspect

from typing import Type


def recursively_update_forward_refs(base_model: Type[BaseModel]):
    horizon: list[Type[BaseModel]] = [base_model]
    while horizon:
        model = horizon.pop(0)
        model.model_rebuild()

        for attr_name in dir(model):
            attr = getattr(model, attr_name)
            if inspect.isclass(attr) and issubclass(attr, BaseModel):
                horizon.append(attr)


_strict: bool = False


@contextmanager
def use_strict():
    """Provide a context where strict mode is enabled"""
    global _strict
    _strict = True
    try:
        yield
    finally:
        _strict = False


def strict_set() -> bool:
    """Return if strict mode is enabled"""
    return _strict
