from random import randint, choice
from gc import collect

#
# author:        Reiji SUZUKI et al.
# refactoring:   Clément BARRIERE
#

class Agent:
    """ 
    An agent is an individual in the simulation that has a position (x, y) and carry a word.
    
    Attributes
    ----------
    id : int
        The unique identifier of the agent.
        
    x : int
        The x-coordinate of the agent's position.
        
    y : int
        The y-coordinate of the agent's position.
        
    word : str
        The word that the agent carries.
        
    Static Attributes
    ----------------
    dxy : list
        A list of possible movements in the grid.
        
    active_agents : list
        A list of all active agents in the simulation.
        
    W : int
        The width of the grid in which the agents exist.
    
    agents_pos : list
        A 2D list representing the positions of agents in the grid.
    """
    
    dxy: list[list[int]] = [[-1,-1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
    active_agents: list = []
    W: int = None
    agents_pos: list = None
    
    id: int
    x: int
    y: int
    word: str
    
    def __init__(self, w: str, id: int, W: int =None):
        """ 
        Initializes an agent with a word and a unique identifier.
        
        Parameters
        ----------
        w : str
            The word that the agent carries.
            
        id : int
            The unique identifier of the agent.
            
        W : int
            The width of the grid in which the agents exist.
            Since W is froze after being set, if it's not None this parameter will be ignored.
        """
        if Agent.W is None: 
            if W is None:
                raise ValueError("Grid's must be known by agents. Please set W at least for the first agent.")
            else:
                #the first agent specifies the grid size in which it appears
                Agent.W = W 
                #the posotion of the agents is initialized to -1 (no agent)
                Agent.agents_pos = [[-1 for _ in range(Agent.W)] for _ in range(Agent.W)]
        
        #the new agent is randomly placed in the grid
        self.x = randint(0, Agent.W-1)
        self.y = randint(0, Agent.W-1)
        
        self.word = w
        self.id = id
        
        #the new agent is added to the active agents list
        Agent.active_agents.append(self)

    @staticmethod
    def _clip(x: int) -> int:
        """
        Clip the x-coordinate to wrap around the grid.
        
        Parameters
        ----------
        x : int
            The x-coordinate to clip.
            
        Returns
        -------
        int
            The clipped x-coordinate.
        """
        if x<0:
            return(x + Agent.W)
        
        elif x >= Agent.W:
            return(x - Agent.W)
        
        else:
            return(x)
    
    def random_walk_2(self, verbose: bool = False) -> None:
        """ 
        Perform a random walk in the grid.
        
        Parameters
        ----------
        verbose : bool, optional
            If True, print the updated position of the agent.
        """
        
        #Randomly select a movement from the list of possible movements
        dfxy = choice(Agent.dxy)
        px = Agent._clip(self.x + dfxy[0])
        py = Agent._clip(self.y + dfxy[1])
        
        #Do not move if the new position is already occupied by another agent
        if Agent.agents_pos[px][py] == -1:
            Agent.agents_pos[self.x][self.y] = -1
            self.x = px
            self.y = py
            Agent.agents_pos[px][py] = self
            
        if verbose: print(f"Upated pos: {px},{py}: {Agent.agents_pos[px][py]}, {Agent.agents_pos[px][py].id}")

    def get_neighbors(self, verbose: bool = False) -> list | None:
        """ 
        Get the neighboring agents of the current agent.
        
        Parameters
        ----------
        verbose : bool, optional
            If True, print the neighboring agents.
            
        Returns
        -------
        list[Agent]
            A list of neighboring agents.
            
        None
            If there are no neighboring agents.
        """
        nlist= []
        
        #Check the 8 possible neighboring positions
        for xd in [-1, 0, 1]:
            for yd in [-1, 0, 1]:
                posagent = Agent.agents_pos[Agent._clip(self.x + xd)][Agent._clip(self.y + yd)]
                
                #If the position is occupied by another agent and it's not the current agent
                if posagent != -1 and posagent.id != self.id:
                    nlist.append(posagent)
                    
                    if verbose: print(f"{posagent}:{posagent.id}")
                    
        if nlist != []:
            return(nlist)
        else:
            return(None)
        
    def compete(self, judge: callable, verbose: bool = False) -> None:
        """ 
        Compete with neighboring agents to determine the dominant word.
        
        Parameters
        ----------
        judge : function
            The callable instance that judges the competition between two words.
            
        verbose : bool, optional
            If True, print the competition details.
        """
        #neighbors= [a for a in agents if a!= self and abs(a.x- self.x)<=1 and abs(a.y- self.y)<=1]
        
        neighbors = self.get_neighbors()
        #if there are neighbours to compete with
        if neighbors != None:
            if verbose: print(f"Compete: {self.id}:{neighbors}")
            
            a = choice(neighbors)
            #competition between same words is not allowed
            if self.word != a.word:
                #if the current agent wins the competition 
                if judge(self.word, a.word) == self.word:
                    #the other agent's word is overwritten
                    a.word = self.word
                    
    def __repr__(self):
        return f"Agent {self.id}: {self.word} at ({self.x}, {self.y})"
    
    @staticmethod
    def clear():
        """
        Remove all agents data.
        """
        for a in Agent.active_agents:
            del a
        
        if Agent.agents_pos is not None:
            for a in Agent.agents_pos:
                del a
            
        collect()