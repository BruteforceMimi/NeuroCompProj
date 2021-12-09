from enum import Enum

class Terrain(Enum):
    """
    Represent the two types of terrain that the agent can walk on. 
    """
    ROUGH = 80 
    PLAIN = 255