from agents import manualAgent
from env import Env 

insect = manualAgent.ManualAgent([200,200], 3)
env = Env(insect) 

while True:
    env.render()
    obs,_,done,_ = env.step(None)
    if done:
        env.reset()
     