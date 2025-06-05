from random import sample, choice
from gc import collect

#
# author:        Reiji SUZUKI et al.
# refactoring:   Clément BARRIERE
#

class Agent:
    """ 
    An agent is an individual in the simulation that has a position (x, y) and carry a word.
    
    Objects Attributes
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
    ----------
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
    agents_pos: list = []
    
    def __init__(self, w: str, id: int):
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
        self.word: str = w
        self.id: int = id
        self.x: int = None
        self.y: int = None
        
        #the new agent is added to the active agents list
        Agent.active_agents.append(self)
    
    @staticmethod
    def set_W(W: int) -> None:
        if (Agent.W is None):
            Agent.W = W
    
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
        
    @staticmethod
    def set_agents_pos(agents_count: int =None) -> None:
        """
        Set each agent's position in the grid and dispatch them.
        
        Parameters
        ----------
        agents_count : int, optional
            The expected total number of agents in the grid.
            May be used if the number of active agents is not equal to the total number of agents.
            
        Raises
        -------
        ValueError
            If the width of the grid (W) is not set.
        """
        if Agent.W is None:
            raise ValueError("Width of the grid (W) is not set. Please set W before calling this method.")
        
        Agent.agents_pos = [[-1 for _ in range(Agent.W)] for _ in range(Agent.W)]
        
        all_positions = sample([
            (x, y) for y in range(Agent.W) 
            for x in range(Agent.W)
            
        ], agents_count if agents_count is not None else len(Agent.active_agents))

        for i, a in enumerate(Agent.active_agents):
            a.x = all_positions[i][0]
            a.y = all_positions[i][1]
            Agent.agents_pos[a.x][a.y] = a
    
    def random_walk_2(self, n: int =1, verbose: bool = False) -> None:
        """ 
        Perform a random walk in the grid.
        
        Parameters
        ----------
        n : int, optional
            The number of random walk to proceed.
            Default is 1.
        verbose : bool, optional
            If True, print the updated position of the agent.
        """
        for _ in range(n):
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

    def get_neighbors(self, verbose: bool = False) -> list:
        """ 
        Get the neighboring agents of the current agent.
        
        Parameters
        ----------
        verbose : bool, optional
            If True, print the neighboring agents.
            
        Returns
        -------
        list[Agent]
            A list of neighboring agents. Empty list if no neighbors are found.
        """
        nlist= []
        
        #Check the 8 possible neighboring positions
        for xd in [-1, 0, 1]:
            for yd in [-1, 0, 1]:
                posagent = Agent.agents_pos[Agent._clip(self.x + xd)][Agent._clip(self.y + yd)]
                
                #If the position is occupied by another agent and it's not the current agent
                if posagent != -1 and posagent.id != self.id:
                    nlist.append(posagent)
                    
        if verbose: print(f"{self.id} neighbors: {nlist}")            
        return(nlist)
        
    def compete(self, judge: callable, gen: int, verbose: bool = False) -> tuple[tuple[object, object], object]:
        """ 
        Compete with neighboring agents to determine the dominant word.
        
        Parameters
        ----------
        judge : function
            The callable instance that judges the competition between two words.
        verbose : bool, optional
            If True, print the competition details.
        gen : int
            The generation number of the competition.
            
        Returns
        -------
        tuple[tuple[Agent, Agent], Agent]
            A tuple containing the two competing agents and the winning agent.
        """
        neighbors = self.get_neighbors()
        res = ((self, None), None)
        
        #if there are neighbours to compete with
        if neighbors != []:
            a = choice(neighbors)
            res = ((self, a), None)
            #competition between same words is not allowed
            if verbose: print(f"Compete: {self.id}_vs_{a.id}")
            if self.word != a.word:
                tmp = judge(self.word, a.word, gen=gen)
                #if the current agent wins the competition 
                if tmp == self.word:
                    #the other agent's word is overwritten
                    a.word = self.word
                    res = (res[0], self)
                else:
                    #if the other agent wins the competition
                    if tmp == a.word:
                        #the overwritting of the current agent's word is not supported
                        #but the result is still archived in the judge history anyway
                        res = (res[0], a)
                        
                    else:
                        res = (res[0], None)
                        
                if verbose: print(f"Result: {self.id}_vs_{a.id} => {res[1].id if res[1] is not None else 'None'}")
                
        return res
        
    def __repr__(self) -> str:
        return f"Agent {self.id} [{self.word}] at ({self.x}, {self.y})"
    
    @staticmethod
    def _clear():
        """
        Remove all agents data.
        """
        for a in Agent.active_agents:
            del a
        
        if Agent.agents_pos is not None:
            for a in Agent.agents_pos:
                del a

        Agent.active_agents = []
        Agent.agents_pos = []
        Agent.W = None
        collect()