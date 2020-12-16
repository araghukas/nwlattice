import random


def get_next(N):
    for i in range(N):
        yield random.randint(0, 30)
