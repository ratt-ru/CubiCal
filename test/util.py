
from sys import stderr


def logprint(*args, **kwargs):
    print(*args, file=stderr, **kwargs)
