from enum import Enum


class State(Enum):
    # closed state
    CLOSED = 1
    # enabled state
    ENABLED = 2
    # open state
    OPEN = 3
    # future enabled state
    FUTURE_ENABLED = 4
