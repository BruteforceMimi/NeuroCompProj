from agents import spikeAgent
from env import Env 
import numpy as np 
import pygame 

insect = spikeAgent.SpikeAgent(np.array([80.0, 550.0]), 15)
env = Env(insect) 
done = False 

display = env.get_display()

while True:
    env.render()
    pygame.event.pump()

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        action = [1, 0]
    elif keys[pygame.K_RIGHT]:
        action = [0, 1]
    elif keys[pygame.K_UP]:
        action = [1, 1]
    else:
        action = [0, 0]
    
    obs, reward, done, info = env.step(action)

    if done:
        env.reset()
     