import numpy as np 
import pygame 
from terrain import Terrain 
from distance import Distance
from PIL import Image, ImageOps
import math


"""
TO DO:
-writing data to file(memory issues)
-terrain does not have influence on speed of agent 
-agent should speed up
"""

class Env():
    """
    """
    def __init__(self, agent):
        pygame.init()
        self.size = np.asarray([600, 600])
        self.display = pygame.display.set_mode(self.size)
        self.agent = agent

        self.map = self._load_map('env3.png')

        self.clock = pygame.time.Clock()
        self.speed = 15
        self.goal = np.array([500,80])
        self.C = np.array([[0.1],[0.01]])

        self.data =  {str(k): [] for k in range(1,13)}

        self.i = 0 

    def step(self, action):
        """
        Used to take one step in the enviornment. 
        returns:
            obs: Array, new observation of the environment
            reward: Int, reward(penalty in our implemenation) the agent gets
            done: Boolean, true if agents reaches goal, false otherwise
            info: dict, some additional information that can be used for debugging 
        """
        obs = []
        reward = 0
        done = False 
        info = {}
        self.agent.step(action)
        self.agent.left_target, self.agent.right_target = self._distance()
        self.agent.left_terrain, self.agent.right_terrain = self._terrain()

        #ugly way to only sample every 5 steps(avoids samples being to similair)

        if self.i % 5 == 0:
            self._generate_data()
        self.i+=1

        total = sum([len(self.data[str(k)]) for k in range(1,13)])
        if total == 12 * 25:
            self._save_csv()
            done = True 

        print([len(self.data[str(k)]) for k in range(1,13)])

        if self.agent.left_target < 20 or self.agent.right_target < 20:
            done = True 
        return obs, reward, done, info

    def render(self):
        """Draws all object to the screen. If not called nothing is rendered"""
        self._draw_terrain()
        self._draw_agent()
        self._draw_goal()
        self.clock.tick(self.speed)
        pygame.display.update()

    def reset(self):
        """Resets the environment. Puts agent back at begin position"""
        self.agent.pos = np.array([80.0, 550.0])

    def _draw_terrain(self):
        """Draws the terrain to the screen"""
        surf = pygame.surfarray.make_surface(self.map)
        self.display.blit(surf, (0, 0))

    def _draw_agent(self):
        """Draws the agent. The distance sensor that is closer to the goal is bigger"""
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
        """Draws the goal to the screen"""
        pygame.draw.circle(self.display, pygame.Color(0,255,0), self.goal, 9)    

    def _distance(self):
        """Calculates the distance from the distance sensors to the goal"""
        g_x, g_y = self.goal 
        l_x, l_y = self.agent.left_target_pos 
        r_x, r_y = self.agent.right_target_pos 
        return np.sqrt((g_x-l_x)**2 + (g_y-l_y)**2), np.sqrt((g_x-r_x)**2 + (g_y-r_y)**2)

    def _terrain(self):
        """Calculates the terrain values for the terrain sensors"""
        l_t = self.map[math.ceil(self.agent.left_terrain_pos[0]), math.ceil(self.agent.left_terrain_pos[1])]
        r_t = self.map[math.ceil(self.agent.right_terrain_pos[0]), math.ceil(self.agent.right_terrain_pos[1])]
        return l_t, r_t 

    def get_display(self):
        return self.display

    def _load_map(self, name):
        """
        Loads the map that is displayed. All map should be in the Simulation/environments/ folder
        name : String, filename 
        """
        map = Image.open(f'Simulation/environments/{name}')
        map = ImageOps.grayscale(map)
        map = np.asarray(map)
        return map 


    # def _check_terrain(self, sensorL_pos, sensorR_pos):
    #     """"""
    #     return [self.map[math.ceil(sensorL_pos[0]), math.ceil(sensorL_pos[1])], self.map[math.ceil(sensorR_pos[0]), math.ceil(sensorR_pos[1])]] 

    def _generate_target(self):
        """ Generators target frequencies that the agent should use """
                            #255 - is used to map black terrain to rough terrain 
        sensors = np.array([[255 - self.agent.left_terrain, self.agent.left_target],
                            [255 - self.agent.right_terrain, self.agent.right_target]]) 

        return sensors @ self.C 

    def _equal(self, sensor1, sensor2):
        """checks if the sensors are close to equal"""
        return abs(sensor1-sensor2) < 5


    def _generate_data(self):
        """
        Generates data used for training according to the 12 cases.
        """
        l_ter = self.agent.left_terrain
        r_ter = self.agent.right_terrain
        l_tar = self.agent.left_target
        r_tar = self.agent.right_target

        l_target, r_target = self._generate_target().flatten()
        
        max_points = 25

        if l_ter == 255 and r_ter == 255 and l_tar > r_tar:
            if len(self.data['1']) < max_points:
                self.data['1'].append([l_ter, r_ter, l_tar, r_tar, l_target, r_target])
        elif l_ter == 255 and r_ter < 255 and l_tar > r_tar:
            if len(self.data['2']) < max_points:
                self.data['2'].append([l_ter, r_ter, l_tar, r_tar, l_target, r_target])
        elif l_ter < 255 and r_ter == 255 and l_tar > r_tar:
            if len(self.data['3']) < max_points:
                self.data['3'].append([l_ter, r_ter, l_tar, r_tar, l_target, r_target])
        elif l_ter < 255 and r_ter < 255 and l_tar > r_tar:
            if len(self.data['4']) < max_points:
                self.data['4'].append([l_ter, r_ter, l_tar, r_tar, l_target, r_target])
        elif l_ter == 255 and r_ter == 255 and self._equal(l_tar, r_tar):
            if len(self.data['5']) < max_points:
                self.data['5'].append([l_ter, r_ter, l_tar, r_tar, l_target, r_target])
        elif l_ter == 255 and r_ter < 255 and self._equal(l_tar, r_tar):
            if len(self.data['6']) < max_points:
                self.data['6'].append([l_ter, r_ter, l_tar, r_tar, l_target, r_target])
        elif l_ter < 255 and r_ter == 255 and self._equal(l_tar, r_tar):
            if len(self.data['7']) < max_points:
                self.data['7'].append([l_ter, r_ter, l_tar, r_tar, l_target, r_target])
        elif l_ter < 255 and r_ter < 255 and self._equal(l_tar, r_tar):
            if len(self.data['8']) < max_points:
                self.data['8'].append([l_ter, r_ter, l_tar, r_tar, l_target, r_target])
        elif l_ter == 255 and r_ter == 255 and l_tar < r_tar:
            if len(self.data['9']) < max_points:
                self.data['9'].append([l_ter, r_ter, l_tar, r_tar, l_target, r_target])
        elif l_ter == 255 and r_ter < 255 and l_tar < r_tar:
            if len(self.data['10']) < max_points:
                self.data['10'].append([l_ter, r_ter, l_tar, r_tar, l_target, r_target])
        elif l_ter < 255 and r_ter == 255 and l_tar < r_tar:
            if len(self.data['11']) < max_points:
                self.data['11'].append([l_ter, r_ter, l_tar, r_tar, l_target, r_target])
        elif l_ter < 255 and r_ter < 255 and l_tar < r_tar:
            if len(self.data['12']) < max_points:
                self.data['12'].append([l_ter, r_ter, l_tar, r_tar, l_target, r_target])
        else:
            print("something went wrong, this should not be executed!")


    def _save_csv(self):
        with open('data.csv', 'a') as f:
            for key, values in self.data.items(): 
                for value in values:    
                    for elm in value:   
                        f.write(str(elm) + " ")
                    f.write("\n")