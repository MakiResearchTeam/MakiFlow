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


class DebugContext:
    def __init__(self, msg=None):
        """
        Useful context manager for debugging.
        Used to catch exceptions and print their messages during debug.

        Parameters
        ----------
        msg : str
            A message to be printed when entering the manager's context.
        """
        self._msg = msg

    def __enter__(self):
        if self._msg is not None:
            print('/------------------------------------------START-------------------------------------------/')
            print(self._msg)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            print('|                                           ERROR                                          |')
            print(f'Exception type: {exc_type}')
            print(f'Exception values: {exc_val}')
        else:
            print('|                                           OKAY                                           |')

        # add some space
        print('/-------------------------------------------END--------------------------------------------/')
        print()
        # An exception is not raised if True is returned
        return True


def d_msg(context, msg_content):
    """
    A simple utility that adds the context to the message string.

    Parameters
    ----------
    context : str
        A string to be appended.
    msg_content: str
        The content of the message.
    Returns
    -------
        Modified message.
    """
    return f'{context} / Message = {msg_content}'

