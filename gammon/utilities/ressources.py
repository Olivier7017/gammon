import time
import sys


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        m = f"Function {func.__name__} took {end_time - start_time:.4f} "
        print(m + "seconds", file=sys.stderr)
        return result
    return wrapper
