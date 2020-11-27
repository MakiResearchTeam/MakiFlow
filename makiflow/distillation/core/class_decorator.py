DECORATOR_DEBUG = False


def overloaded(method):
    """
    It is a simple dummy-marker indicating that the method
    of the wrapped class is being overloaded.
    Marking the methods explicitly as being overloaded might be
    very useful for the developer.
    """
    global DECORATOR_DEBUG

    def wrapper(*args, **kwargs):
        if DECORATOR_DEBUG:
            print(f'Calling method={method.__name__} overloaded by {args[0].__class__}.')

        return method(*args, **kwargs)

    return wrapper


class ClassDecorator:
    """
    This instance allows for 'inheritance' of all the API the wrapped object has.
    """
    def __init__(self):
        self._obj = None

    def __getattr__(self, item):
        # If the method was not found in the wrapper class, __getattr__ is being called.
        # If the wrapped object has this method, return it.
        # Otherwise an exception is thrown.
        if hasattr(self._obj, item):
            return self._obj.__getattribute__(item)

        return self.__getattribute__(item)

    def __call__(self, obj):
        self._obj = obj
        self._call_init(obj)
        return self

    def get_obj(self):
        return self._obj

    def _call_init(self, obj):
        """
        Called right after the __call__ method has been called.
        """
        pass


if __name__ == '__main__':
    class A:
        def a(self):
            print('A')

    class ADecorator(ClassDecorator):
        def b(self):
            print('B')

        def print_a(self):
            obj = super().get_obj()
            obj.a()

    a = A()
    b = ADecorator()

    a.a()
    b.b()
    print()

    a_wrapped = b(a)
    a_wrapped.a()
    a_wrapped.b()
    a_wrapped.print_a()

    class ADecorator(ClassDecorator):
        def b(self):
            print('B')

        def print_a(self):
            obj = super().get_obj()
            obj.a()

        def a(self):
            print('Overload a')

    print('Overloaded')
    b = ADecorator()
    a_wrapped = b(a)
    a_wrapped.a()
    a_wrapped.b()
    a_wrapped.print_a()
    print(a_wrapped.has_method('b'))
