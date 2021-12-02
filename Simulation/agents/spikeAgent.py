import agents.agent 
import numpy as np

class SpikeAgent(agents.agent.Agent):
    """
    Agent that is controlled with a SNN 
    """
    def __init__(self, start_pos, radius):
        super().__init__(start_pos, radius)

        #used for moving the agent
        self.vx = 0 #velocity x--axis
        self.vy = 0 #veolicty y- axis
        self.v = 0 #linear speed 
        self.theta = 0 #angle between agent and x-axis 
        self.vl = 0 #speed of left engine 
        self.vr = 0 #speed of right motor 

        #hyper paramters
        self.l = 1
        self.eta = 1 
        self.tau = 1.3 

    def step(self, action):
        l_spike, r_spike = action 
        self.vx = self.v * np.cos(self.theta)
        self.vy = self.v * np.sin(self.theta)
        self.v = (self.vl + self.vr) / 2
        self.theta += (self.vr - self.vl) / self.l 
        self.vl += -self.vl / self.tau + self.eta * l_spike
        self.vr += -self.vr / self.tau + self.eta * r_spike
        update_pos = np.array([self.vx, self.vy])
        self.pos += update_pos 

        self.left_target_pos = np.array([np.cos(self.theta + np.pi / 4), np.sin(self.theta + np.pi / 4)]) * 12
        self.left_target_pos += self.pos  

        self.right_target_pos = np.array([np.cos(self.theta - np.pi / 4), np.sin(self.theta - np.pi / 4)]) * 12
        self.right_target_pos += self.pos  

        self.left_terrain_pos = np.array([np.cos(self.theta + np.pi / 2), np.sin(self.theta + np.pi / 2)])  * (self.radius)+ self.pos 
        self.right_terrain_pos = np.array([np.cos(self.theta - np.pi / 2), np.sin(self.theta - np.pi / 2)]) * (self.radius) + self.pos 
