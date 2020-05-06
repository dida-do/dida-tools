"""Utilities for converting functions to callable classes"""

def snake2camal(string: str) -> str:
    """Change string from snake_case to CamalCase"""
    return ''.join(word.title() for word in string.split('_'))

def objectise(fn: Callable) -> type:
    """Take a function and create a class with kwargs as attributes and a call method that evaluates the function."""
    field_list = []
    for k, v in inspect.signature(fn).parameters.items():
        if v.default is not inspect.Parameter.empty:
            field_list.append((k, v.annotation, field(default=v.default)))
    
    def wrap_call(self, *args):
        return fn(*args, **{k: self.__dict__[k] for k, _, _ in field_list})
    
    return make_dataclass(snake2camal(fn.__name__), field_list, namespace={"__call__": wrap_call})