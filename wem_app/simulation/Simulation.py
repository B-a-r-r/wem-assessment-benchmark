from os import path, makedirs, remove
from random import shuffle, uniform, seed as rnd_seed, choice
from numpy import random as np_rnd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import Counter
import torch
import numpy as np
from .Agent import Agent
from .Judge import Judge
from io import TextIOWrapper
from gc import collect
from json import dump
from timeit import default_timer as timer
from .WemActor import WemActor

#
# author:       Reiji SUZUKI et al.
# refactor:     Clément BARRIERE
#

class Simulation(WemActor):
    """ 
    A class to run a Ecology of Words simulation.
    
    Objects attributes
    ----------
    simul_config : dict
        The configuration of the simulation.
    logs_path : str
        The path to the log file.
    _logs_file : TextIOWrapper
        The log file object.
    judge : Judge
        The judge of the simulation (in charge of competitions, mutation).
    emerged_words : list
        The list of words that emerged during the current trial.
    gens_words_frequency : dict[int, Counter]
        A dictionary mapping generations to a Counter of words to track frequency.
    competition_history : dict[int, dict[tuple[int, int], int]]
        A dictionary mapping generations to a dictionary of competitions, where each key is a tuple of agent IDs and the value is the winner's ID.
    mutation_history : dict[int, dict[str, dict[str, list[list[str]]]]]
        A dictionary mapping generations to a dictionary of mutations, where each key is the source word and the value is a dictionary of mutations with their derivatives.
    simul_advance : list[tuple[int, int, int, int, str]]
        A list of tuples representing the simulation's progress, where each tuple contains the generation, agent ID, x position, y position, and word.
    steps_duration : np.ndarray
        An array to store the duration of each step along the current trial.
    current_gen : int
        The current generation number in the current trial.
    current_trial : int
        The current trial number in the simulation.
    is_running : bool
        A flag indicating whether the simulation is currently running.
    real_time_image_path : str, exists if enable_real_time_views is True
        The path to the real-time views image file.
    fig : matplotlib.figure.Figure, exists if enable_real_time_views is True
        The figure object for real-time views of the simulation.
    _trial_res_logs : TextIOWrapper | None
        The log file for trial results, exists if logging is enabled.
    _trial_competition_history_logs : TextIOWrapper | None
        The log file for competition history, exists if logging is enabled.
    _trial_mutation_history_logs : TextIOWrapper | None
        The log file for mutation history, exists if logging is enabled.
    """
    
    def __init__(self, 
        config: dict, 
        enable_logs: bool =True,
        enable_real_time_views: bool =True,
    ) -> None:
        """
        Initializes the Simulation object with the given configuration.
        
        Parameters
        ----------
        config : dict
            The configuration dictionary for the simulation.
        enable_logs : bool, optional
            Whether to enable logging (default is True).
        enable_real_time_views : bool, optional
            Whether to enable real-time views of the simulation (default is True).
        """
        self.simul_config: dict = config
        self.verify_config()
        
        self.logs_path = path.join(path.abspath(self.simul_config["workspace"]["exp_dir"]), "logs.txt") if enable_logs else None
        self._logs_file: TextIOWrapper = None
        self._log_event(event="Log file initialized.", underline= True)
        
        self.judge: Judge = None
        self.emerged_words: list = []
        self.gens_words_frequency: dict[int, Counter] = {}
        self.competition_history: dict[int, dict[tuple[int, int], int]] = {}
        self.mutation_history: dict[int, dict[str, dict[str, list[list[str]]]]] = {}
        self.simul_advance: list[tuple[int, int, int, int, str]] = []
        self.steps_duration: np.ndarray = np.array([], dtype=np.float64)
        self.current_gen: int = 0
        self.current_trial: int = 0
        self.is_running: bool = True
        
        self._trial_res_logs: TextIOWrapper | None = None
        self._trial_competition_history_logs: TextIOWrapper | None = None
        self._trial_mutation_history_logs: TextIOWrapper | None = None
            
        #set up the experience output directory
        if not path.exists(path.abspath(self.simul_config["workspace"]["exp_dir"])):
            makedirs(path.abspath(self.simul_config["workspace"]["exp_dir"]))
        self._log_event(event=f"Experience output directory found/created: {self.simul_config["workspace"]["exp_dir"]}.")
            
        if enable_real_time_views:
            self.real_time_image_path = path.join(path.abspath(self.simul_config["workspace"]["exp_dir"]), "real_time_views.png")
            self.fig: plt.Figure
            self._create_real_time_views()
            self._log_event(event="Real-time views initiated.")
            
        self.run()
    
    def _log_event(self, 
        event: str, 
        source: str =None, 
        indent: str ="", 
        underline: bool =False,
        type: str =None,
    ) -> None:
        """
        Logs an event to the log file and prints it to the console.
        
        Parameters
        ----------
        event : str
            The event message to log.
        source : str, optional
            The source of the event (default is None).
        indent : str, optional
            The indentation to apply to the event message (default is an empty string).
        underline : bool, optional
            Whether to underline the event message (default is False).
        type : str, optional
            The type of the event (e.g., "INFO", "WARNING", "FATAL") (default is None).
        """
        self._logs_file = super()._log_event(
            event=event, source=source, indent=indent, underline=underline, type=type, 
            logs_path=self.logs_path, logs_file=self._logs_file
        )
    
    def _create_real_time_views(self) -> None:
        """
        Creates the figure and axes for real-time views of the simulation.
        This method sets up the figure with a grid layout and adds subplots for agent positions and word frequency views.
        """
        self.fig = plt.figure(figsize=(8*3, 6))
        grid_spec = self.fig.add_gridspec(1, 3)
        self.fig.suptitle("Real-time views of the simulation", fontsize=16)
        
        self.fig.add_subplot(grid_spec[0, 0])
        self.fig.add_subplot(grid_spec[0, 1:])
        
        #There is currently a bug with pyplot, the windows does not update and freezes.
        #Cannot use this method for now.
        #self._real_time_window()
    
    def _real_time_image(self):
        """
        Generates a real-time image of the simulation and saves it to the specified path.
        This method updates the agent positions and word frequency views, then saves the figure as an image.
        """
        ax = self.fig.get_axes()
        
        self._agent_pos_view(ax[0])
        self._word_freq_view(ax[1])

        self.fig.savefig(
            self.real_time_image_path,
            dpi=300
        )
        
        ax = None
        
    def _real_time_window(self): 
        """
        Creates a real-time updated plt window for the simulation.
        """
        ax = self.fig.get_axes()
        
        def update(frame):
            self._agent_pos_view(ax[0])
            self._word_freq_view(ax[1])
            self.fig.canvas.draw()
            
        anim = FuncAnimation(self.fig, update, interval=200)
        plt.ion()
        plt.show()
        
    def _agent_pos_view(self, ax: plt.Axes) -> None:
        """
        Creates a real-time view of the agents' positions on a grid.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to plot the agents' positions on.
        """
        ax.clear()
        
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.gens_words_frequency[self.current_gen].keys())))
        color_map = {word: colors[i] for i, word in enumerate(self.gens_words_frequency[self.current_gen].keys())}
        
        ax.set_title(f"Current Agents Positions\nTrial {self.current_trial} Gen {self.current_gen}", fontsize=12)
        ax.set_xlim(0, self.simul_config["simulation"]["W"])
        ax.set_ylim(0, self.simul_config["simulation"]["W"])
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, color='gray', linestyle='--', linewidth=0.2)
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12*0.8)
        
        for agent in Agent.active_agents:
            if agent.word in color_map:
                ax.add_artist(plt.Circle((agent.x, agent.y), 0.3, fill=True, color=color_map[agent.word], alpha=0.7))
                ax.text(agent.x, agent.y, agent.word, fontsize=12*0.5, ha='center', va='center')
        
    def _word_freq_view(self, ax: plt.Axes) -> None:
        """
        Creates a real-time view of the top B words frequency over generations.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to plot the word frequency on.
        """
        ax.clear()
        
        top_B_words = self.gens_words_frequency[self.current_gen].most_common(self.simul_config["workspace"]["top_B"])
        
        for word, _ in top_B_words:
            present_at_gens = []
            word_freq = []
            for gen, counter in self.gens_words_frequency.items():
                if counter.keys().__contains__(word):
                    present_at_gens.append(gen)
                    word_freq.append(counter[word])
            ax.plot(present_at_gens, word_freq, label=word)
            
        ax.set_title(f"Current Top{self.simul_config["workspace"]["top_B"]} Words Frequency\nTrial {self.current_trial} Gen {self.current_gen}", fontsize=12)
        ax.set_xlabel("Generation", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12*0.8)
        ax.set_xlim(0, self.simul_config["simulation"]["S"])
        ax.set_ylim(0, self.simul_config["simulation"]["N"])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12*0.8)
    
    def _step(self) -> None:
        """
        Performs a step in the simulation.
        """
        self._log_event(event=f"Step {self.current_gen} started.", indent="\t")
        
        self._log_event(event="Starting random walks phase.", indent="\t", underline=True)
        shuffle(Agent.active_agents) #improve globality
        for a in Agent.active_agents:
            a.random_walk_2(
                n=self.simul_config["simulation"]["N_WALK"],
                verbose=self.simul_config["workspace"]["verbose"]
            )
        
        self._log_event(event="Starting competition phase.", indent="\t", underline=True)
        self.competition_history[self.current_gen] = {}
        shuffle(Agent.active_agents)
        for a in Agent.active_agents:
            res = a.compete(judge=self.judge.judge, gen=self.current_gen, verbose=self.simul_config["workspace"]["verbose"])
            self.competition_history[self.current_gen][(res[0][0].id if res[0][0] is not None else None, res[0][1].id if res[0][1] is not None else None)] = res[1].id if res[1] is not None else None
        
        self._log_event(event="Starting mutation phase.", indent="\t", underline=True)
        self.mutation_history[self.current_gen] = {}
        shuffle(Agent.active_agents)
        for a in Agent.active_agents:
            if uniform(0, 1) < self.simul_config["simulation"]["MUT_PROB"]:
                mutation = self.judge.mutate(
                    a.word,
                    verbose=self.simul_config["workspace"]["verbose"]
                )
                
                if not self.mutation_history[self.current_gen].keys().__contains__(a.word):
                    self.mutation_history[self.current_gen][a.word] = {}
                if not self.mutation_history[self.current_gen][a.word].keys().__contains__(mutation[0]):
                    self.mutation_history[self.current_gen][a.word].update({mutation[0]: [mutation[1]]})
                else:
                    self.mutation_history[self.current_gen][a.word][mutation[0]].extend(mutation[1])
                a.word = mutation[0]
                
        self._log_event(f"Step {self.current_gen} completed.")
        
    def _update(self) -> None:
        """
        Updates the computed variables and trigger current state logging after a step.
        """
        current_words = []
        
        for a in Agent.active_agents:
            current_words.append(a.word)
            
            self.simul_advance.append((
                self.current_gen, a.id, a.x, a.y, a.word.replace('"','')
            ))
        
        current_unique_words= set(current_words)
        self.gens_words_frequency[self.current_gen] = Counter(current_words)
            
        if self.simul_config["workspace"]["log_trial_results"]:
            self.log_trial_results(t=self.current_trial)
            
        if self.simul_config["workspace"]["log_judgement_history"]:
            self.judge.log_case_law(t=self.current_trial)
            
        if self.simul_config["workspace"]["log_mutation_history"]:
            self.log_mutation_history(t=self.current_trial)
            
        if self.simul_config["workspace"]["log_competition_history"]:
            self.log_competition_history(t=self.current_trial)
        
        self._real_time_image() if hasattr(self, 'fig') else None
        
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
        
        plt.close(self.fig) if hasattr(self, 'fig') else None
        remove(self.real_time_image_path) if hasattr(self, 'real_time_image_path') and path.exists(self.real_time_image_path) else None
        
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
        self.gens_words_frequency = {}
        self.emerged_words = []
        self.steps_duration = np.array([], dtype=np.float64)
        
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
        """
        Runs the simulation; for each trials, initializes the required varaibles before launching it.
        """
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

            self._log_event(event=f"Frequency dict initialized.")
            
            self.current_trial = current_trial
            self._run_trial()
            self._log_event(f"Average step duration: {round(self.steps_duration.sum()/len(self.steps_duration) if len(self.steps_duration) > 0 else 1, 2)} seconds.")
            
            self._clean_after_trial()
            self._log_event(f"Trial {current_trial} data cleaned.")
        
        self._log_event(f"Simulation reached the end. Bravo!")
        print(f"---Simulation reached the end.---")
        self.is_running = False
        self._clear()
            
    def _run_trial(self) -> None:
        """
        Runs a trial of the simulation.
        """
        self._log_event(f"Starting main loop for trial {self.current_trial}.")
        
        #for S steps
        for i in range(self.simul_config["simulation"]["S"]):
            step_start_time = timer()
            self.current_gen = i
            
            self._step()
            self._update()
            
            step_end_time = timer()
            np.append(self.steps_duration, step_end_time - step_start_time)
            
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
                assert self.simul_config["workspace"]["top_B"] > 0, "top_B must be greater than 0."
            except KeyError:
                self.simul_config["workspace"]["top_B"] = 10
            
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
