import numpy as np 
import pygame 

class Env():
    """
    """
    def __init__(self, agent):
        self.size = np.asarray([600, 600])
        self.display = pygame.display.set_mode(self.size)
        self.agent = agent
        self.map = np.ones((self.size)) * 255

    def step(self, action):
        return 0,0,0,0

    def render(self):
        self._draw_terrain()
        self._draw_agent()
        pygame.display.update()

    def reset(self):
        pass 

    def _generate_terrain(self):
        pass 

    def _draw_terrain(self):
        surf = pygame.surfarray.make_surface(self.map)
        self.display.blit(surf, (0, 0))

    def _draw_agent(self):
        pygame.draw.circle(self.display, pygame.Color(255, 0, 0), self.agent.pos, self.agent.radius)



    
