import numpy as np 
import pygame 
from terrain import Terrain 
from distance import Distance

class Env():
    """
    """
    def __init__(self, agent):
        self.size = np.asarray([600, 600])
        self.display = pygame.display.set_mode(self.size)
        self.agent = agent
        self.map = np.ones((self.size)) * 255

        self.clock = pygame.time.Clock()
        self.speed = 2

        self.goal = np.array([400,400])

    def step(self, action):
        self.agent.step(action)
        d_l, d_r = self._distance()
        if d_l == d_r:
            self.agent.left_target = Distance.EQUAL
            self.agent.right_target = Distance.EQUAL 
        elif d_l < d_r: 
            self.agent.left_target = Distance.WEAK 
            self.agent.right_target = Distance.STRONG
        else: 
            self.agent.left_target = Distance.STRONG 
            self.agent.right_target = Distance.WEAK 
        print("left: ", self.agent.left_target)
        print("right: ", self.agent.right_target)

        return 0,0,0,0

    def render(self):
        self._draw_terrain()
        self._draw_agent()
        self._draw_goal()
        self.clock.tick(self.speed)
        pygame.display.update()

    def reset(self):
        pass 

    def _generate_terrain(self):
        pass 

    def _draw_terrain(self):
        surf = pygame.surfarray.make_surface(self.map)
        self.display.blit(surf, (0, 0))

    def _draw_agent(self):
        pygame.draw.circle(self.display, pygame.Color(0, 0, 0), self.agent.pos, self.agent.radius)
        closest = np.argmin(self._distance())
        print(closest)
        if closest == 0: #left sensor
            pygame.draw.circle(self.display, pygame.Color(255, 0, 0), self.agent.left_target_pos, 2)
            pygame.draw.circle(self.display, pygame.Color(0, 0, 255), self.agent.right_target_pos, 2)
        if closest == 1: #right sensor
            pygame.draw.circle(self.display, pygame.Color(0, 0, 255), self.agent.left_target_pos, 2)
            pygame.draw.circle(self.display, pygame.Color(255, 0, 0), self.agent.right_target_pos, 2)


    def _draw_goal(self):
        pygame.draw.circle(self.display, pygame.Color(0,255,0), self.goal, 9)    

    def _distance(self):
        g_x, g_y = self.goal 
        l_x, l_y = self.agent.left_target_pos 
        r_x, r_y = self.agent.right_target_pos 
        return np.sqrt((g_x-l_x)**2 + (g_y-l_y)**2), np.sqrt((g_x-r_x)**2 + (g_y-r_y)**2)