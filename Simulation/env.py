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
        self.speed = 1

        self.goal = np.array([500,80])
        self.C = np.array([[0.1],[0.01]])

    def step(self, action):
        """

        """
        obs = []
        reward = 0
        done = False 
        info = {}
        self.agent.step(action)
        self.agent.left_target, self.agent.right_target = self._distance()
        self.agent.left_terrain, self.agent.right_terrain = self._terrain()
        target = self._generate_target()
        print(target)
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

    def _terrain(self):
        l_t = self.map[math.ceil(self.agent.left_terrain_pos[0]), math.ceil(self.agent.left_terrain_pos[1])]
        r_t = self.map[math.ceil(self.agent.right_terrain_pos[0]), math.ceil(self.agent.right_terrain_pos[1])]
        return l_t, r_t 

    def get_display(self):
        return self.display

    def _load_map(self, name):
        map = Image.open(f'Simulation/environments/{name}')
        map = ImageOps.grayscale(map)
        map = np.asarray(map)
        return map 


    def _check_terrain(self, sensorL_pos, sensorR_pos):
        return [self.map[math.ceil(sensorL_pos[0]), math.ceil(sensorL_pos[1])], self.map[math.ceil(sensorR_pos[0]), math.ceil(sensorR_pos[1])]] 

    def _generate_target(self):
        sensors = np.array([[self.agent.left_terrain, self.agent.left_target],
                            [self.agent.right_terrain, self.agent.right_target]]) 

        return sensors @ self.C 