from os import path, makedirs
from random import shuffle, uniform, seed as rnd_seed, choice, sample
from numpy import random as np_rnd
import torch
from json import load
from Agent import Agent
from Judge import Judge
from io import TextIOWrapper
from gc import collect
from datetime import datetime

#
# author:       Reiji SUZUKI et al.
# refactor:     Clément BARRIERE
#

class Simulation:
    """ 
    A class to run a Ecology of Words simulation.
    
    Objects attributes
    ----------
    simul_config: dict
        The configuration, extracted from the given json file, for the simulation.
        
    config_path: str
        The path to the json configuration file.
    
    judge: Judge
        Judge object to manage the interactions between simulated population and LLM.
    
    words: list
        The list of words currently in the simulation.
    
    frequency: dict
        The frequency of each word in the simulation.
        
    logs_file: TextIOWrapper
        The logs file IO wrapper.
        
    trial_res_logs: TextIOWrapper | None
        The IO wrapper to log the results of each trial.
    """
    
    def __init__(self, 
        config_path: str, 
        logs_path: str =None,
    ) -> None:
        """
        Initializes the simulation with the given configuration and run it.

        Parameters
        ----------
        config_path : str
            The path to the json configuration file.
            
        logs_path : str
            The path to the logs file.
        """
        self.config_path: str = path.abspath(config_path)
        self.logs_path: str = path.abspath(logs_path) if logs_path is not None else None
        
        self.judge: Judge = None
        self.words: list = []
        self.frequency: dict = {}
        
        self._logs_file: TextIOWrapper = None
        self._trial_res_logs: TextIOWrapper | None = None
        
        with open(self.config_path, "r") as f:
            self.simul_config = load(f)
            self.verify_config()
            
        self.run()
        
    def step(self, s: int) -> None:
        """
        Performs a step in the simulation.

        Parameters
        ----------
        s : int
            The current step of the simulation.
        """
        self.log_event(event=f"Step {s} started.", indent="\t")
        
        shuffle(Agent.active_agents) #improve globality
        for a in Agent.active_agents:
            a.random_walk_2(
                n=self.simul_config["simulation"]["N_WALK"],
                verbose=self.simul_config["wrokspace"]["verbose"]
            )
        
        shuffle(Agent.active_agents)
        for a in Agent.active_agents:
            a.compete(judge=self.judge.judge, verbose=self.simul_config["wrokspace"]["verbose"])
            #a.compete3()
        
        shuffle(Agent.active_agents)
        for a in Agent.active_agents:
            if uniform(0, 1) < self.simul_config["simulation"]["MUT_RATE"]:
                a.word = self.judge.mutate(
                    a.word,
                    verbose=self.simul_config["wrokspace"]["verbose"]
                )
                
        self.log_event(f"Step {s} completed.")
        
    def update(self, s: int) -> None:
        """
        Updates the frequency dictionary and logs the results of the current step.
                
        Parameters
        ----------
        s : int
            The current step of the simulation.
        """
        current_unique_words= set([a.word for a in Agent.active_agents])
        
        for w in current_unique_words:
            if w not in self.frequency:
                self.frequency[w]= [0] * (s+1)
                self.words.append(w)
        for w in self.words:
            self.frequency[w].append(len([a for a in Agent.active_agents if a.word==w]))
            
        if self.simul_config["wrokspace"]["log_trial_results"]:
            for i, a in enumerate(Agent.active_agents):
                self._trial_res_logs.write(str(s)+","+str(a.id)+","+str(a.x)+","+str(a.y)+","+'"'+a.word.replace('"','')+'"'+"\n")

    @staticmethod
    def set_seed(SEED: int) -> None:
        """
        Sets the seed for random number generators to ensure reproducibility.
        Concerned generators are: random, numpy and pytorch.

        Parameters
        ----------
        SEED : int
            The seed to set for the random number generators.
        """
        # Random
        rnd_seed(SEED)
        # Numpy
        np_rnd.seed(SEED)
        # Pytorch
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark=False

    def clear(self) -> None:
        """
        Cleans up the simulation by closing the log files and force freeing memory.
        """
        del self
        collect()
        
        
    def clean_after_trial(self) -> None:
        """
        Closes the opened files and frees memory.
        """
        self._trial_res_logs.close()
        self.judge.clear()
        Agent.clear()
        
    def log_event(self, 
        event: str, 
        source: str =None, 
        indent: str ="", 
        underline: bool =False,
        type: str ="",
    ) -> None:
        """
        Logs an event to the log file with some 
        additional information.

        Parameters
        ----------
        event : str
            The event to log.
            
        source : str, optional
            The source file of the event.
            
        indent : str, optional
            The indentation to use for the log message. 
            Use "/t" for each indentation level (relpace the "/" with a backslash).
            
        underline : bool, optional
            Whether to underline the log message or not.
            
        type : str, optional
            Specific type of the event (FATAL, WARNING, etc...)
        """
        if self._logs_file is None:
            #if the logs path is not set
            if self.logs_path is None:
                print("--- WARNING - No logs path set for the simulation. No information will be logged. ---")
                return
            
            #create and/or open the log file
            self._logs_file = open(
                path.join(path.dirname(__file__), self.simul_config["wrokspace"]["exp_dir"], self.logs_path),
                "w+",
                encoding="utf-8"
            )
            #if the file is not empty, overwrite the content
            if self._logs_file.readlines() != []:
                self._logs_file.write("\n")
            
            self.log_event(event="Log file initialized.", underline= True)
        
        self._logs_file.writelines(\
            f"{indent}{datetime.now().hour}:{datetime.now().minute}:{datetime.now().second}\
            {f" - {type}" if type is not None else ""}\
            {f" - from {source}" if source is not None else ""}\
            - {event}\n{indent}{"------------\n" if underline else ""}"
        )
        self._logs_file.flush()
    
    def _init_population(self, initial_word_list: list) -> None:
        """ 
        Initializes the population of agents with the given initial word list.
        
        Parameters
        ----------
        initial_word_list : list
            A list of words to be initially assigned to the agents.
        """
        try:
            #set the grid size for future agents
            Agent.set_W(self.simul_config["simulation"]["W"])
            #create the agents and assign them a word
            for i in range(0, self.simul_config["simulation"]["N"], 1):
                #because A <= N
                if len(Agent.active_agents) < self.simul_config["simulation"]["A"]:
                    #each word is at least assigned to one agent
                    word = initial_word_list[i]
                else:
                    #the rest of the agents get a random word
                    word = choice(initial_word_list)
                    
                Agent(
                    w= word,
                    id= i,
                )
            assert len(Agent.active_agents) == self.simul_config["simulation"]["N"], f"Wrong number of agents created: should be {len(Agent.active_agents)} but got {self.simul_config["simulation"]["N"]}."
            
        except Exception as e:
            self.log_event(
                event=f"Error creating agents: {e}",
                source="Simulation",
                type="FATAL"
            )
            raise e

        #agents get a random position in the grid
        all_positions = sample([
            (x, y) for y in range(self.simul_config["simulation"]["W"]) 
            for x in range(self.simul_config["simulation"]["W"])
        ], self.simul_config["simulation"]["N"])

        for i, a in enumerate(Agent.active_agents):
            a.x = all_positions[i][0]
            a.y = all_positions[i][1]
            Agent.agents_pos[a.x][a.y] = a
            
        self.log_event(f"Agents created and dispatched.")
    
    def run(self) -> None:
        """Initializes the simulation and lauches it."""
        
        #set up the experience output directory
        if not path.exists(path.abspath(self.simul_config["wrokspace"]["exp_dir"])):
            makedirs(path.abspath(self.simul_config["wrokspace"]["exp_dir"]))
        self.log_event(event=f"Experience output directory found/created: {self.simul_config["wrokspace"]["exp_dir"]}.")
        
        #for T trials
        for current_trial in range(0, self.simul_config["simulation"]["T"]):
            self.log_event(event=f"Trial {current_trial} started.", underline= True)
            
            self.judge = Judge(
                criteria= self.simul_config["simulation"]["CRITERIA"],
                config= self.simul_config,
                log_event= self.log_event
            )
            
            self.set_seed(self.simul_config["simulation"]["SEED"] + current_trial)
            self.log_event(event=f"Seed set to {self.simul_config["simulation"]["SEED"] + current_trial}.")

            #initialize the base word list for the base population
            words = self.judge.create_word_list(
                verbose= self.simul_config["wrokspace"]["verbose"],
            )
            self.log_event(event=f"Initial words list created: {words}.")
        
            self._init_population(words)
            
            #initialize the frequency dict 
            for w in words:
                self.frequency[w]= [len([i for i in words if i==w])]
            self.log_event(f"Frequency dict initialized.")
            
            if self.simul_config["wrokspace"]["log_trial_results"]:
                self._trial_res_logs = open(path.join(path.dirname(__file__), f"{self.simul_config["wrokspace"]["exp_dir"]}", f"results_trial{current_trial}.csv"), "w+", encoding="utf-8")
                
                #the header of the csv file
                self._trial_res_logs.write("gen,id,x,y,word\n")
                
                #the first line of the csv file is the initial population
                for i, a in enumerate(Agent.active_agents):
                    self._trial_res_logs.write(str(0)+","+str(a.id)+","+str(a.x)+","+str(a.y)+","+'"'+a.word.replace('"','')+'"'+"\n")
                
                self.log_event(f"Trial {current_trial} results file created.")

            self._run_trial(current_trial)
            
            if self.simul_config["wrokspace"]["log_judgement_history"]:
                self.judge.log_case_law(log_label=current_trial)
            
            self.clean_after_trial()
            self.log_event(f"Trial {current_trial} data cleaned.")
        
        self.clear()
            
    def _run_trial(self, t: int) -> None:
        """
        Runs a trial of the simulation.
        
        Parameters
        ----------
        t : int
            The current trial number.
        """
        self.log_event(f"Starting main loop for trial {t}.")
        
        #for S steps
        for i in range(self.simul_config["simulation"]["S"]):
            self.step(i)
            self.update(i)
            
    def verify_config(self) -> None:
        """
        Verifies that the configuration file is valid.
        May proceed to some adjustments to the configuration according to the inputed values.
        """
        self.log_event(f"Chcking requirements from config...", "Simulation", "\t")
        
        try:
            assert self.simul_config["simulation"]["N"] is not None, "N is not defined in the 'simulation' section of the config file."
            assert self.simul_config["simulation"]["N_WALK"] is not None, "N_WALK is not defined in the 'simulation' section of the config file."
            assert self.simul_config["simulation"]["A"] is not None, "A is not defined in the 'simulation' section of the config file."
            assert self.simul_config["simulation"]["T"] is not None, "T is not defined in the 'simulation' section of the config file."
            assert self.simul_config["simulation"]["S"] is not None, "S is not defined in the 'simulation' section of the config file."
            assert self.simul_config["simulation"]["W"] is not None, "W is not defined in the 'simulation' section of the config file."
            assert self.simul_config["simulation"]["CRITERIA"] is not None, "CRITERIA is not defined in the 'simulation' section of the config file."
            
            assert self.simul_config["workspace"]["exp_dir"] is not None, "Missing exp_dir parameter in the 'workspace' section of the config file."
            assert self.simul_config["workspace"]["log_trial_results"] is not None, "Missing log_trial_results parameter in the 'workspace' section of the config file."
            assert self.simul_config["workspace"]["log_judgement_history"] is not None, "Missing log_judgement_history parameter in the 'workspace' section of the config file."
            
            assert self.simul_config["simulation"]["N"] >= self.simul_config["simulation"]["A"], "N must be greater than or equal to A."
            
            if self.simul_config["simulation"]["N_WALK"] <= 0:
                self.log_event(
                    event=f"N_WALK is set to {self.simul_config["simulation"]["N_WALK"]} in the config file, which means the agents will not move.",
                    source="Simulation",
                    type="WARNING",
                    indent="\t"
                )
            
            if self.simul_config["workspace"]["verbose"] is None:
                self.simul_config["workspace"]["verbose"] = False
                
            if self.simul_config["simulation"]["SEED"] is None:
                self.simul_config["simulation"]["SEED"] = 42
            
        except Exception as e:
            self.log_event(
                event=f"Error verifying configuration file: {e}",
                source="Simulation",
                type="FATAL"
            )
            raise e
