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
            print(self._msg)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f'Exception type: {exc_type}')
        print(f'Exception values: {exc_val}')
        # An exception is not raised if True is returned
        return True
