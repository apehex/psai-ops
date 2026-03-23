import functools
import inspect
import typing

# TYPE CHECKING ################################################################

def typecheck(
    func: object=None,
    inputs: bool=True,
    outputs: bool=False,
    ignores: set=None,
    returns: object=None,
) -> object:
    __ignores = set(ignores) or {'self'}
    # adapts the decorator according to the arguments
    def _decorator(_f: object) -> object:
        # parse the metadata
        __sign = inspect.signature(_f)
        __hints = typing.get_type_hints(_f)
        # actual function that will be called
        @functools.wraps(_f)
        def _wrapper(*args, **kwargs):
            # match the values with the argument names
            __bound = __sign.bind_partial(*args, **kwargs)
            __bound.apply_defaults()
            # check the input types
            if inputs:
                # iterate
                for __name, __value in __bound.arguments.items():
                    # by default, ignore the `self` pointer
                    if __name in __ignores:
                        continue
                    # get the target type, with `object` as default
                    __type = __hints.get(__name, object)
                    # everything in python is an `object`, if the type is not specified the check succeeds
                    if not isinstance(__value, __type):
                        return returns
            # actually call the function
            __outputs = _f(*args, **kwargs)
            # check the return value
            if outputs:
                # again unspecified output types => `object`
                __type = __hints.get('return', object)
                # succeeds when the type was not specified
                if not isinstance(__outputs, __type):
                    return returns
            # the wrapper returns the output of its inner function
            return __outputs
        # the factory returns the custom wrapper
        return _wrapper
    # can be called on its own
    if func is None:
        return _decorator
    # finally, wrap the input function
    return _decorator(func)
