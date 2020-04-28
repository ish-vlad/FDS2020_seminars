from __future__ import print_function

import time

from functools import wraps


class Logger:
    def __init__(self, verbose=True):
        self.verbose = verbose

    def print_(self, print_string, flush=True, end='\n', level=0):
        print_string = '\t' * level + print_string + end

        if self.verbose:
            print(print_string, flush=flush, end='')

        return print_string


def timeit(dictionary_field):
    def _inner_decorator(method):
        @wraps(method)
        def _impl(self, *method_args, **method_kwargs):
            t1 = time.time()
            method_output = method(self, *method_args, **method_kwargs)
            t2 = time.time()

            d = getattr(self, dictionary_field)
            key = type(self).__name__ + '.' + method.__name__
            d[key] = t2 - t1
            return method_output
        return _impl
    return _inner_decorator
