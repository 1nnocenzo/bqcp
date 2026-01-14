"""
This class is used to represent the state of classical bits in the propagation of the classical registers
"""

from enum import Enum, auto

class BitState(Enum):
    ZERO = auto()
    ONE = auto()
    NOT_KNOWN = auto()

    def __repr__(self) -> str:
        return self.name