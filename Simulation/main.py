from agents import spikeAgent
from env import Env 
import numpy as np 

insect = spikeAgent.SpikeAgent(np.array([200.0, 200.0]), 15)
env = Env(insect) 

while True:
    env.render()
    action = np.random.randint(2, size=2)
    obs,_,done,_ = env.step(action)
    if done:
        env.reset()
     