"""
TODO:
    - does the last plane fit the first plane?
"""


def is_periodic(wire) -> bool:
    return wire.planes[0].fits(wire.planes[-1])
