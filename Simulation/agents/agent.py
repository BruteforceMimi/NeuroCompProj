from abc import ABC, abstractmethod #needed to create abstract class 
from terrain import Terrain 
from distance import Distance

class Agent(ABC):
    """
    Abstract class for the agent.
    Is used in the force controlled agent and PPO-agent
    """
    def __init__(self, start_pos, radius):
        self.pos = start_pos 
        self.radius = radius 
        self.left_motor = 0 
        self.right_motor = 0 
        self.left_terrain = Terrain.PLAIN
        self.right_terrain = Terrain.PLAIN
        self.left_target = Distance.EQUAL
        self.right_target = Distance.EQUAL 

    @abstractmethod
    def step(self, action):
        pass 


