# Copyright (C) 2020  Igor Kilbas, Danil Gribanov, Artem Mukhin
#
# This file is part of MakiFlow.
#
# MakiFlow is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MakiFlow is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <https://www.gnu.org/licenses/>.


class ExceptionScope(object):
    __SCOPE_COUNTER = 1
    DEFAULT_NAME = 'ExceptionScope'

    def __init__(self, scope_name=None):
        """
        Context manager used to mark the context in which an exception occurred.

        Parameters
        ----------
        scope_name : str
            A prefix that will be prepended to the message of the raised exception.
        """
        if scope_name is None:
            scope_name = ExceptionScope.DEFAULT_NAME + f'/{ExceptionScope.__SCOPE_COUNTER}'
        self._scope = scope_name

        ExceptionScope.__SCOPE_COUNTER += 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            space_bar = 30 * '-'
            modified_exc_val = f'ExcScope: {self._scope} \n\n{space_bar}\n\nExcMsg: {exc_val}'
            exc = exc_type(modified_exc_val)
            exc.with_traceback(exc_tb)
            raise exc

        # An exception is not raised if True is returned
        return True


def fn_exception_scope(scope_name=None):
    """
    Creates an exception scope for a particular function.

    Parameters
    ----------
    scope_name : str, optional
        Name of the scope. Can be None, in this case name of the method is considered to be the scope name.
    """

    def get_context_name(scope, method):
        if scope is None:
            return method.__name__
        else:
            return scope + ' / ' + method.__name__

    def decorator(method):
        def wrapper(*args, **kwargs):
            with ExceptionScope(get_context_name(scope_name, method)):
                return method(*args, **kwargs)

        return wrapper
    return decorator


def method_exception_scope(scope_name=None):
    """
    Creates an exception scope for a particular method of some object (this method must contain `self` argument).
    It will append information about the owner-object of the wrapped method.

    Parameters
    ----------
    scope_name : str, optional
        Name of the scope. Can be None, in this case name of the method is considered to be the scope name.
    """

    def get_context_name(obj, method):
        obj_signature = f'\nObj={str(obj)}\nObjClsName={obj.__class__.__name__}\nmethod={method.__name__}'
        if scope_name is None:
            return obj_signature
        else:
            return scope_name + obj_signature

    def decorator(method):
        def wrapper(self, *args, **kwargs):
            arg_signature = f'args={args}\nkwargs={kwargs}'
            with ExceptionScope(get_context_name(self, method) + '\n' + arg_signature):
                return method(self, *args, **kwargs)

        return wrapper
    return decorator
