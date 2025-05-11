from matplotlib import pyplot as plt, gridspec, figure
from os import path, makedirs
from random import shuffle, uniform, seed as rnd_seed, choice, sample
from numpy import random as np_rnd
import torch
from json import load, dump
from Agent import Agent
from Judge import Judge
from utils import count_non_negative_one, encode_dict, decode_dict
from LanguageModelHandler import LanguageModelHandler
from dotenv import dotenv_values
from io import TextIOWrapper
from gc import collect
from datetime import datetime
from sys import exit

#
# author:       Reiji SUZUKI et al.
# refactor:     Clément BARRIERE
#

class Simulation:
    """ 
    A class to run a Ecology of Words simulation.
    
    Attributes
    ----------
    simul_config: dict
        The configuration for the simulation.
        
    config_path: str
        The path to the configuration file.
        
    log_trial_res: bool
        Whether to log each trial final results or not.
    
    judge: Judge
        The judge.
    
    words: list
        The list of words currently in the simulation.
    
    frequency: dict
        The frequency of each word in the simulation.
    
    gstep: int
        The step of the animation.
    
    fig: matplotlib.figure.Figure
        The figure of the animation.
    
    gs: matplotlib.gridspec.GridSpec
        The grid of the animation.
        
    logs_file: TextIOWrapper
        The logs file.
        
    trail_res_logs: TextIOWrapper
        The current trial results logs file.
        
    final_trial_res_logs: TextIOWrapper
        The final trial results logs file.
    """
    simul_config: dict
    config_path: str
    log_trial_res: bool
    
    judge: Judge
    words: list
    frequency: dict
    gstep: int
    fig: figure.Figure
    gs: gridspec.GridSpec
    
    logs_file: TextIOWrapper
    trail_res_logs: TextIOWrapper
    final_trial_res_logs: TextIOWrapper
    
    def __init__(self, 
        config_path: str, 
        logs_path: str, 
        log_trial_res: bool =True
        
    ) -> None:
        """
        Initializes the simulation with the given configuration and run it.

        Parameters
        ----------
        config_path : str
            The path to the configuration file.
            
        logs_path : str
            The path to the logs file.
            
        log_trial_res : bool, optional
            Whether to log each trial results or not.
        """
        self.config_path = path.join(path.dirname(__file__), config_path)
        self.logs_path = logs_path
        self.log_trial_res = log_trial_res
        
        self.frequency= {}
        self.words= []
        self.gstep= 0
        self.fig = plt.figure(figsize=(10, 5))
        self.gs = gridspec.GridSpec(2, 3)
        
        self.judge = None
        self.logs_file = None
        self.trail_res_logs = None
        self.final_trial_res_logs = None
        
        try:
            self.simul_config = load(open(self.config_path, "r"))
            
            assert self.simul_config["simulation"] is not None, "Check your config file. Simulation configuration is missing."
            assert self.simul_config["model"] is not None, "Check your config file. Model configuration is missing."
        
        except Exception as e:
            self.log_event(event=f"Error loading configuration file: {e}", color="")
            exit(1)
        
        self.run()
        
    def step(self, t: int) -> None:
        """
        Performs a step in the simulation.

        Parameters
        ----------
        t : int
            The current step of the simulation.
        """
        self.log_event(event=f"Step {t} started.", indent="\t")
        
        self.log_event(event=f"Shuffling agents.", indent="\t\t")
        shuffle(Agent.active_agents) #improve globality
        for a in Agent.active_agents:
            a.random_walk_2(verbose=self.simul_config["simulation"]["VERBOSE"])
        
        self.log_event(event=f"Agents competing.", indent="\t\t")
        shuffle(Agent.active_agents)
        for a in Agent.active_agents:
            a.compete(judge=self.judge.judge, verbose=self.simul_config["simulation"]["VERBOSE"])
            #a.compete3()
        
        self.log_event(event=f"Agents mutating.", indent="\t\t")
        shuffle(Agent.active_agents)
        for a in Agent.active_agents:
            if uniform(0, 1) < self.simul_config["simulation"]["MUT_RATE"]:
                a.word = self.judge.mutate(
                    a.word,
                    verbose=self.simul_config["simulation"]["VERBOSE"]
                )
    
    def update(self, t: int, verbose: bool =False) -> None:
        """
        Updates the simulation by plotting the current state.
        
        Parameters
        ----------
        t : int
            The current step of the simulation.
            
        verbose : bool, optional
            Whether to print the current state of the simulation or not.
        """
        self.gstep= t
        self.fig.clear()
        ax1= self.fig.add_subplot(self.gs[1,0])
        x= [a.x for a in Agent.active_agents]
        y= [a.y for a in Agent.active_agents]
        
        ax1.scatter(x, y, color='brown')
        ax1.axis([-1, self.simul_config["simulation"]["W"], -1, self.simul_config["simulation"]["W"]])

        #overlays the words on the graph
        for a in Agent.active_agents:
            ax1.text(a.x, a.y, a.word, fontsize=8)
    
        #shows the frequency of the words with a histogram
        ax3= self.fig.add_subplot(self.gs[0,1])
        wds= [a.word for a in Agent.active_agents]
        ax3.hist(wds, self.simul_config["simulation"]["N"])
        ax3.tick_params(axis='x', rotation=90)
        ax3.set_title('word frequency')

        current_unique_words= set(wds)
        for w in current_unique_words:
            if w not in self.frequency:
                self.frequency[w]= [0] * (t+1)
                self.words.append(w)
        for w in self.words:
            self.frequency[w].append(len([a for a in Agent.active_agents if a.word==w]))

        ax4= self.fig.add_subplot(self.gs[0,2])
        for w in self.words:
            ax4.plot(self.frequency[w], label=w)
        ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=2, columnspacing=0.05)
        ax4.set_title('word frequency')
        ax4.set_xlabel('step')
        
        if verbose: print(t, [a.word for a in Agent.active_agents])
        if self.log_trial_res:
            for i, a in enumerate(Agent.active_agents):
                self.trail_res_logs.write(str(t)+","+str(a.id)+","+str(a.x)+","+str(a.y)+","+'"'+a.word.replace('"','')+'"'+"\n")

    def set_seed(self, SEED: int) -> None:
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
        self.clean_after_trial()
        Agent.clear()
        del self
        collect()
        
        
    def clean_after_trial(self) -> None:
        """
        Closes the log files and frees memory.
        """
        self.trail_res_logs.close()
        self.final_trial_res_logs.close()
        self.judge.clear()
        
    def log_event(self, 
        event: str, 
        source: str =None, 
        indent: str ="", 
        underline: bool =False, 
        color: str =""
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
            
        color : str, optional
            The color to use for the log message.
        """
        color = ""
        if self.logs_file is None:
            #create and/or open the log file
            self.logs_file = open(
                path.join(path.dirname(__file__), self.simul_config["simulation"]["EXP_DIR"], self.logs_path),
                "w+",
                encoding="utf-8"
            )
            #if the file is not empty, overwrite the content
            if self.logs_file.readlines() != []:
                self.logs_file.write("\n")
            
            self.log_event(event="Log file initialized.", underline= True)
        
        self.logs_file.writelines(f"{color}{indent}{datetime.now().hour}:{datetime.now().minute}:{datetime.now().second}{f" - from {source}" if source is not None else ""} - {event}\n{indent}{"------------\n" if underline else ""}")#\033[0m")
        self.logs_file.flush()
    
    def run(self) -> None:
        """Runs the simulation."""
        
        #set up the xperience output directory
        if not path.exists(path.join(path.dirname(__file__), self.simul_config["simulation"]["EXP_DIR"])):
            makedirs(path.join(path.dirname(__file__), self.simul_config["simulation"]["EXP_DIR"]), exist_ok=True)
        
        #for the set number of trials
        for current_trial in range(0, self.simul_config["simulation"]["T"]):
            self.log_event(event=f"Trial {current_trial} started.") 
            
            try:
                auth_token = dotenv_values(path.abspath("../.env"))["HF_AUTH_TOKEN"]
            except KeyError:
                auth_token = dotenv_values(path.abspath("./.env"))["HF_AUTH_TOKEN"]
            
            try:
                model = LanguageModelHandler(
                    model_name= self.simul_config["model"]["name"],
                    auth_token= auth_token,
                    log_event= self.log_event,
                )
            except Exception as e:
                self.log_event(
                    event=f"Error loading the model: {e}",
                    color="\033[91m"
                )
                raise e
                exit(1)    
            self.log_event(event=f"Model {self.simul_config["model"]["name"]} initialized.")
            
            self.judge = Judge(
                model= model,
                criteria= self.simul_config["simulation"]["CRITERIA"],
                config_path= self.config_path,
                log_event= self.log_event
            )
            self.log_event(event=f"Judge '{self.simul_config["simulation"]["CRITERIA"]}' loaded.")
            
            self.set_seed(self.simul_config["simulation"]["SEED"] + current_trial)

            #initialize the base word list for the base population
            words = self.judge.create_word_list(
                verbose= self.simul_config["simulation"]["VERBOSE"],
            )
            self.log_event(event=f"Words list created: {words}.")
            
            try:
                #create the agents and assign them a word
                for i in range(0, self.simul_config["simulation"]["N"], 1):
                    #because A <= N
                    if len(Agent.active_agents) < self.simul_config["simulation"]["A"]:
                        #each word is at least assigned to one agent
                        word = words[i]
                    else:
                        #the rest of the agents get a random word
                        word = choice(words)
                        
                    Agent(
                        w= word,
                        id= i,
                        W= self.simul_config["simulation"]["W"],
                    )
                assert len(Agent.active_agents) == self.simul_config["simulation"]["N"], "Wrong number of created agents."
                
            except Exception as e:
                self.log_event(
                    event=f"Error creating agents: {e}",
                    source="Simulation",
                    color="\033[91m"
                )
                exit(1)

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
            
            #initialize the frequency dict 
            for w in words:
                self.frequency[w]= [len([a for a in Agent.active_agents if a.word == w])]
            self.log_event(f"Frequency dict initialized.")
            
            if self.log_trial_res:
                self.trail_res_logs = open(path.join(path.dirname(__file__), f"{self.simul_config["simulation"]["EXP_DIR"]}", f"results_trial{current_trial}.csv"), "w+", encoding="utf-8")
                self.trail_res_logs.write("gen,id,x,y,word\n")
                
                for i, a in enumerate(Agent.active_agents):
                    self.trail_res_logs.write(str(0)+","+str(a.id)+","+str(a.x)+","+str(a.y)+","+'"'+a.word.replace('"','')+'"'+"\n")
                
                self.log_event(f"Trial {current_trial} results file created.")
            
            self.log_event(f"Simulation initialized. Starting main loop.")
            for i in range(self.simul_config["simulation"]["T"]):
                self.step(i)
                self.update(i)
                self.fig.savefig(f'trial_{current_trial}_step_{i}.png')
                self.log_event(f"Step {i} completed.")
            
            #at the end, saving the results, the close the files and free the memory
            self.final_trial_res_logs = open(f"{self.simul_config["simulation"]["EXP_DIR"]}/final_results_trial-{current_trial}.json", "w", encoding="utf-8")
            dump(encode_dict(self.judge.judgments_history), self.trail_res_logs, indent=2)
            self.log_event(f"Trial {current_trial} results saved.")
            
            self.clean_after_trial()
            self.log_event(f"Trial {current_trial} data cleaned.")
            
        self.clear()
