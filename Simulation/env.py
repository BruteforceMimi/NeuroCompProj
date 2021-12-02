import numpy as np 
import pygame 
from terrain import Terrain 
from distance import Distance
from PIL import Image, ImageOps
import math

class Env():
    """
    """
    def __init__(self, agent):
        pygame.init()
        self.size = np.asarray([600, 600])
        self.display = pygame.display.set_mode(self.size)
        self.agent = agent

        self.map = self._load_map('env2.png')

        self.clock = pygame.time.Clock()
        self.speed = 15

        self.goal = np.array([500,80])

    def step(self, action):
        """

        """
        obs = []
        reward = 0
        done = False 
        info = {}
        print(self._check_terrain(self.agent.left_terrain_pos, self.agent.right_terrain_pos))
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

        return obs, reward, done, info

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
        if closest == 0: #left sensor
            pygame.draw.circle(self.display, pygame.Color(255, 0, 0), self.agent.left_target_pos, 5)
            pygame.draw.circle(self.display, pygame.Color(255, 0, 0), self.agent.right_target_pos, 3)
        if closest == 1: #right sensor
            pygame.draw.circle(self.display, pygame.Color(255, 0, 0), self.agent.left_target_pos, 3)
            pygame.draw.circle(self.display, pygame.Color(255, 0, 0), self.agent.right_target_pos, 5)
            
        pygame.draw.circle(self.display, pygame.Color(150, 75, 0), self.agent.left_terrain_pos, 3)
        pygame.draw.circle(self.display, pygame.Color(150, 75, 0), self.agent.right_terrain_pos, 3)


    def _draw_goal(self):
        pygame.draw.circle(self.display, pygame.Color(0,255,0), self.goal, 9)    

    def _distance(self):
        g_x, g_y = self.goal 
        l_x, l_y = self.agent.left_target_pos 
        r_x, r_y = self.agent.right_target_pos 
        return np.sqrt((g_x-l_x)**2 + (g_y-l_y)**2), np.sqrt((g_x-r_x)**2 + (g_y-r_y)**2)

    def get_display(self):
        return self.display

    def _load_map(self, name):
        map = Image.open(f'Simulation/environments/{name}')
        map = ImageOps.grayscale(map)
        map = np.asarray(map)
        return map 


    def _check_terrain(self, sensorL_pos, sensorR_pos):
        return [self.map[math.ceil(sensorL_pos[0]), math.ceil(sensorL_pos[1])], self.map[math.ceil(sensorR_pos[0]), math.ceil(sensorR_pos[1])]] 