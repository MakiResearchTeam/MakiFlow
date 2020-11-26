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
