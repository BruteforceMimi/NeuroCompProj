import agents.agent 

class ManualAgent(agents.agent.Agent):
    """
    Agent that is controlled with a keyboard 
    """
    def __init__(self, start_pos, radius):
        super().__init__(start_pos, radius)

    def step(self, move):
        pass