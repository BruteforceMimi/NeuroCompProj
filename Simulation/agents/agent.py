from abc import ABC, abstractmethod #needed to create abstract class 
from terrain import Terrain 
from distance import Distance
import numpy as np

class Agent(ABC):
    """
    Abstract class for the agent.
    
    Attributes
    ----------
    pos : array
        x and y coordinate of the agent 
    radius : int
        agent is a circle which is represented with a given radius 
    left_terrain : Terrain 
        left terrain sensor 
    right_terrain : Terrain 
        right terrain sensor 
    left_target : Distance
        left distance sensor
    right_target : Distance
        right distance sensor
    """
    def __init__(self, pos, radius):
        self.pos = pos 
        self.radius = radius 
        self.left_terrain = Terrain.PLAIN
        self.right_terrain = Terrain.PLAIN
        self.left_target = Distance.EQUAL
        self.right_target = Distance.EQUAL 
        self.left_target_pos = np.array([np.sqrt(2) / 2, np.sqrt(2) / 2]) + self.pos
        self.right_target_pos = np.array([np.sqrt(2) / 2, -np.sqrt(2) / 2]) + self.pos 

        self.left_terrain_pos = np.array([0,1]) + self.pos 
        self.right_terrain_pos = np.array([0, -1]) + self.pos 

    @abstractmethod
    def step(self, action):
        """
            Action that the agent takes [1,1]
        """
        pass