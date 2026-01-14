
from enum import Enum, auto

class ActivationState(Enum):
    """
    
    """
    ALWAYS = auto()
    NEVER = auto()
    SOMETIMES = auto()
    UNKNOWN = auto()

    def __str__(self) -> str:
        return self.name