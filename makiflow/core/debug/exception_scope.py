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


class ExceptionScope:
    def __init__(self, scope_name):
        """
        Context manager used to mark the context in which an exception occurred.

        Parameters
        ----------
        scope_name : str
            A prefix that will be prepended to the message of the raised exception.
        """
        self._scope = scope_name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            modified_exc_val = f'{self._scope} / {exc_val}'
            exc = exc_type(modified_exc_val)
            exc.with_traceback(exc_tb)
            raise exc

        # An exception is not raised if True is returned
        return True
