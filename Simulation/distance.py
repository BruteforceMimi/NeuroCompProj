from enum import Enum

class Distance(Enum):
    """
        Represent the three kinds of distances each distance sensor can have
    """
    WEAK = 0
    EQUAL = 1 
    STRONG = 2