from os import path, makedirs
from random import shuffle, uniform, seed as rnd_seed, choice
from numpy import random as np_rnd
import torch
from Agent import Agent
from Judge import Judge
from io import TextIOWrapper
from gc import collect
from json import dump
from timeit import default_timer as timer
from WemActor import WemActor

#
# author:       Reiji SUZUKI et al.
# refactor:     Clément BARRIERE
#

class Simulation(WemActor):
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
    emerged_words: list
        The list of words that emerged in the simulation.
    frequency: dict
        The frequency of each word in the simulation.
    logs_path: str
        The path to the logs file.
    _logs_file: TextIOWrapper
        The logs file IO wrapper.
    _trial_res_logs: TextIOWrapper | None
        The IO wrapper to log the results of each trial.
    _trial_competition_history_logs: TextIOWrapper | None
        The IO wrapper to log the competition history of each trial.
    _trial_mutation_history_logs: TextIOWrapper | None
        The IO wrapper to log the mutation history of each trial.
    competition_history: dict
        The history of the competition between agents.
    mutation_history: dict[int, dict[str, tuple[str, list[str]]]]
        The history of the mutations of words accross generations.
        {generation: {source word: [new word, [mutations]]}}
    average_step_duration: float
        The average duration of a step in the simulation.
    simul_advance: list[tuple[int, int, int, int, str]]
        The list of tuples containing the simulation progress.
        Each tuple contains (generation, agent id, x position, y position, word).
    """
    
    def __init__(self, 
        config: dict, 
        enable_logs: bool =True,
    ) -> None:
        """
        Initializes the simulation with the given configuration and run it.

        Parameters
        ----------
        config_path : str
            The path to the json configuration file.
        enable_logs : bool, optional
            Whether to enable logging file or not. Default is True.
        """
        self.simul_config: dict = config
        self.verify_config()
        
        self.logs_path = path.join(path.abspath(self.simul_config["workspace"]["exp_dir"]), "logs.txt") if enable_logs else None
        self._logs_file: TextIOWrapper = None
        self._log_event(event="Log file initialized.", underline= True)
        
        self.judge: Judge = None
        self.emerged_words: list = []
        self.frequency: dict = {}
        self.competition_history: dict[int, dict[tuple[int, int], int]] = {}
        self.mutation_history: dict[int, dict[str, dict[str, list[list[str]]]]] = {}
        self.simul_advance: list[tuple[int, int, int, int, str]] = []
        self.average_step_duration: float = 0.0
        
        self._trial_res_logs: TextIOWrapper | None = None
        self._trial_competition_history_logs: TextIOWrapper | None = None
        self._trial_mutation_history_logs: TextIOWrapper | None = None
            
        #set up the experience output directory
        if not path.exists(path.abspath(self.simul_config["workspace"]["exp_dir"])):
            makedirs(path.abspath(self.simul_config["workspace"]["exp_dir"]))
        self._log_event(event=f"Experience output directory found/created: {self.simul_config["workspace"]["exp_dir"]}.")
            
        self.run()
    
    def _log_event(self, 
        event: str, 
        source: str =None, 
        indent: str ="", 
        underline: bool =False,
        type: str =None,
    ) -> None:
        self._logs_file = super()._log_event(
            event=event, source=source, indent=indent, underline=underline, type=type, 
            logs_path=self.logs_path, logs_file=self._logs_file
        )
    
    def _step(self, s: int) -> None:
        """
        Performs a step in the simulation.

        Parameters
        ----------
        s : int
            The current step of the simulation.
        """
        self._log_event(event=f"Step {s} started.", indent="\t")
        
        self._log_event(event="Starting random walks phase.", indent="\t", underline=True)
        shuffle(Agent.active_agents) #improve globality
        for a in Agent.active_agents:
            a.random_walk_2(
                n=self.simul_config["simulation"]["N_WALK"],
                verbose=self.simul_config["workspace"]["verbose"]
            )
        
        self._log_event(event="Starting competition phase.", indent="\t", underline=True)
        self.competition_history[s] = {}
        shuffle(Agent.active_agents)
        for a in Agent.active_agents:
            res = a.compete(judge=self.judge.judge, gen=s, verbose=self.simul_config["workspace"]["verbose"])
            self.competition_history[s][(res[0][0].id if res[0][0] is not None else None, res[0][1].id if res[0][1] is not None else None)] = res[1].id if res[1] is not None else None
        
        self._log_event(event="Starting mutation phase.", indent="\t", underline=True)
        self.mutation_history[s] = {}
        shuffle(Agent.active_agents)
        for a in Agent.active_agents:
            if uniform(0, 1) < self.simul_config["simulation"]["MUT_PROB"]:
                mutation = self.judge.mutate(
                    a.word,
                    verbose=self.simul_config["workspace"]["verbose"]
                )
                
                if not self.mutation_history[s].keys().__contains__(a.word):
                    self.mutation_history[s][a.word] = {}
                if not self.mutation_history[s][a.word].keys().__contains__(mutation[0]):
                    self.mutation_history[s][a.word].update({mutation[0]: [mutation[1]]})
                else:
                    self.mutation_history[s][a.word][mutation[0]].extend(mutation[1])
                a.word = mutation[0]
                
        self._log_event(f"Step {s} completed.")
        
    def _update(self, s: int, t: int) -> None:
        """
        Updates the frequency dictionary and logs the results of the current step.
                
        Parameters
        ----------
        s : int
            The current step of the simulation.
        t : int
            The current trial of the simulation.
        """
        current_words = []
        
        for a in Agent.active_agents:
            current_words.append(a.word)
            
            self.simul_advance.append((
                s, a.id, a.x, a.y, a.word.replace('"','')
            ))
        
        current_unique_words= set(current_words)
        
        for w in current_unique_words:
            if w not in self.frequency:
                self.frequency[w]= [0] * (s+1)
                self.emerged_words.append(w)
                
        for w in current_unique_words:
            self.frequency[w].append(len([a for a in Agent.active_agents if a.word==w]))
            
        if self.simul_config["workspace"]["log_trial_results"]:
            self.log_trial_results(t=t)
            
        if self.simul_config["workspace"]["log_judgement_history"]:
            self.judge.log_case_law(t=t)
            
        if self.simul_config["workspace"]["log_mutation_history"]:
            self.log_mutation_history(t=t)
            
        if self.simul_config["workspace"]["log_competition_history"]:
            self.log_competition_history(t=t)
        
        self._log_event(event=f"Current population ({len(Agent.active_agents)}): {current_words}.")
        self._log_event(event=f"Current number of words: {len(current_unique_words)}.")
        self._log_event(event=f"Current emergence rate: {len(self.emerged_words)}.", underline= True)

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

    def _clear(self) -> None:
        """
        Cleans up the simulation by closing the log files and force freeing memory.
        """
        if self._logs_file is not None:
            self._logs_file.flush()
            self._logs_file.close()
            self._logs_file = None
            
        del self
        collect()
        
    def _clean_after_trial(self) -> None:
        """
        Closes the opened files and frees memory.
        """
        if self._trial_res_logs is not None:
            self._trial_res_logs.flush()
            self._trial_res_logs.close()
            self._trial_res_logs = None
            
        if self._trial_competition_history_logs is not None:
            self._trial_competition_history_logs.flush()
            self._trial_competition_history_logs.close()
            self._trial_competition_history_logs = None
            
        if self._trial_mutation_history_logs is not None:
            self._trial_mutation_history_logs.flush()
            self._trial_mutation_history_logs.close()
            self._trial_mutation_history_logs = None
        
        self.simul_advance = []
        self.competition_history = {}
        self.mutation_history = {}
        self.frequency = {}
        self.emerged_words = []
        
        self.judge._clear()
        Agent._clear()
    
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
                if len(Agent.active_agents) < len(initial_word_list):
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
            self._log_event(
                event=f"Error creating agents: {e}",
                source="Simulation",
                type="FATAL"
            )
            raise e
        
        try:
            Agent.set_agents_pos(agents_count= self.simul_config["simulation"]["N"])
            
        except Exception as e:
            self._log_event(
                event=f"Error setting agents positions: {e}",
                source="Simulation",
                type="FATAL"
            )
            raise e
        
        self._log_event(f"Agents created and dispatched.")
    
    def run(self) -> None:
        """Initializes the simulation and lauches it."""
        #for T trials
        for current_trial in range(0, self.simul_config["simulation"]["T"]):
            self._log_event(event=f"Trial {current_trial} started.", underline= True)
            
            self.judge = Judge(
                config= self.simul_config,
                enable_logs= (self.logs_path != None)
            )
            
            self.set_seed(self.simul_config["simulation"]["SEED"] + current_trial)
            self._log_event(event=f"Seed set to {self.simul_config["simulation"]["SEED"] + current_trial}.")

            #initialize the base word list for the base population
            words = self.judge.create_word_list(
                verbose= self.simul_config["workspace"]["verbose"],
            )
            self._log_event(event=f"Initial words list created: {words}.")
        
            self._init_population(words)
            
            #initialize the frequency dict 
            for w in words:
                self.frequency[w]= [len([i for i in words if i==w])]
            self._log_event(f"Frequency dict initialized.")

            self._run_trial(current_trial)
            
            self._clean_after_trial()
            self._log_event(f"Trial {current_trial} data cleaned.")
        
        self.average_step_duration /= self.simul_config["simulation"]["S"]*self.simul_config["simulation"]["T"]
        self._log_event(f"Average step duration: {round(self.average_step_duration, 2)} seconds.")
        self._log_event(f"Simulation reached the end. Bravo!")
        print(f"---Simulation reached the end.---")
        self._clear()
            
    def _run_trial(self, t: int) -> None:
        """
        Runs a trial of the simulation.
        
        Parameters
        ----------
        t : int
            The current trial number.
        """
        self._log_event(f"Starting main loop for trial {t}.")
        
        #for S steps
        for i in range(self.simul_config["simulation"]["S"]):
            step_start_time = timer()
            
            self._step(i)
            self._update(i, t)
            
            step_end_time = timer()
            self.average_step_duration += step_end_time - step_start_time
            
    def log_competition_history(self, t: int) -> None:
        """ 
        Logs the competition history to a file.
        
        Parameters
        ----------
        t : int
            The current trial number.
        """
        if self._trial_competition_history_logs is None:
            self._trial_competition_history_logs = open(f"{path.abspath(self.simul_config["workspace"]["exp_dir"])}/competition_history_{t}.json", "w", encoding="utf-8")
            self._log_event(f"Competition history file created.", "Simulation")
        
        self._trial_competition_history_logs.seek(0)
        self._trial_competition_history_logs.truncate()
        dump({
            gen : {
                str(competitors) : winner for competitors, winner in competitions.items()
            } for gen, competitions in self.competition_history.items()
        }, self._trial_competition_history_logs, indent=2)
        
        self._log_event(f"Competition history logs updated.", "Simulation")
        self._trial_competition_history_logs.flush()
        
    def log_trial_results(self, t: int) -> None:
        """ 
        Logs the trial results to a file.
        
        Parameters
        ----------
        t : int
            The current trial number.
        """
        if self._trial_res_logs is None:
            self._trial_res_logs = open(path.join(path.abspath(self.simul_config["workspace"]["exp_dir"]), f"results_{t}.csv"), "w+", encoding="utf-8")
            self._log_event(f"Results file created.", "Simulation")
        
        self._trial_res_logs.seek(0)
        self._trial_res_logs.truncate()
        #the header of the csv file
        self._trial_res_logs.write("gen,id,x,y,word\n")
        for line in self.simul_advance:
            self._trial_res_logs.write(f"{line[0]},{line[1]},{line[2]},{line[3]},{line[4]}\n")
                
        self._log_event(f"Results logs updated.", "Simulation")
        self._trial_res_logs.flush()
        
    def log_mutation_history(self, t: int) -> None:
        """ 
        Logs the mutation history to a file.
        
        Parameters
        ----------
        t : int
            The current trial number.
        """
        if self._trial_mutation_history_logs is None:
            self._trial_mutation_history_logs = open(f"{path.abspath(self.simul_config["workspace"]["exp_dir"])}/mutation_history_{t}.json", "w", encoding="utf-8")
            self._log_event(f"Mutation history file created.", "Simulation")
        
        self._trial_mutation_history_logs.seek(0)
        self._trial_mutation_history_logs.truncate()
        dump({
            gen : { 
                source : derivatives for source, derivatives in mutations.items() 
            } for gen, mutations in self.mutation_history.items()
        }, self._trial_mutation_history_logs, indent=2)
        
        self._log_event(f"Mutation history logs updated.", "Simulation")
        self._trial_mutation_history_logs.flush()
            
    def verify_config(self) -> None:
        """
        Verifies that the configuration file is valid.
        May proceed to some adjustments to the configuration according to the inputed values.
        """
        try:
            assert self.simul_config["simulation"]["N"] is not None, "N is not defined in the 'simulation' section of the config file."
            assert self.simul_config["simulation"]["N_WALK"] is not None, "N_WALK is not defined in the 'simulation' section of the config file."
            assert self.simul_config["simulation"]["A"] is not None, "A is not defined in the 'simulation' section of the config file."
            assert self.simul_config["simulation"]["T"] is not None, "T is not defined in the 'simulation' section of the config file."
            assert self.simul_config["simulation"]["S"] is not None, "S is not defined in the 'simulation' section of the config file."
            assert self.simul_config["simulation"]["W"] is not None, "W is not defined in the 'simulation' section of the config file."
            assert self.simul_config["simulation"]["MUT_PROB"] is not None, "MUT_PROB is not defined in the 'simulation' section of the config file."
            
            assert self.simul_config["workspace"]["exp_dir"] is not None, "Missing exp_dir parameter in the 'workspace' section of the config file."
            
            assert self.simul_config["simulation"]["N"] >= self.simul_config["simulation"]["A"], "N must be greater than or equal to A."
            
            if self.simul_config["simulation"]["N_WALK"] <= 0:
                self._log_event(
                    event=f"N_WALK is set to {self.simul_config["simulation"]["N_WALK"]} in the config file, which means the agents will not move.",
                    source="Simulation",
                    type="WARNING",
                    indent="\t"
                )
            
            try:
                self.simul_config["workspace"]["verbose"]
            except KeyError:
                self.simul_config["workspace"]["verbose"] = False
                
            try:
                self.simul_config["simulation"]["SEED"]
            except KeyError:
                self.simul_config["simulation"]["SEED"] = 42
                
            try:
                assert self.simul_config["workspace"]["log_judgement_history"]
            except KeyError:
                self.simul_config["workspace"]["log_judgement_history"] = False
                
            try:
                assert self.simul_config["workspace"]["log_competition_history"]
            except KeyError:
                self.simul_config["workspace"]["log_competition_history"] = False
                
            try:
                self.simul_config["workspace"]["log_mutation_history"]
            except KeyError:
                self.simul_config["workspace"]["log_mutation_history"] = False
                
            try:
                self.simul_config["workspace"]["log_trial_results"]
            except KeyError:
                self.simul_config["workspace"]["log_trial_results"] = True
            
        except Exception as e:
            print(f"--- FATAL - Error in the config file ---")
            raise e
