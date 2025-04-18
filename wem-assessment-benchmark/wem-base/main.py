import re
import matplotlib
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import os
from random import shuffle, uniform, seed as rnd_seed, choice, sample
from numpy import random as np_rnd
import torch
from json import load, dump
from sys import argv
from Agent import Agent
from Judge import Judge
from ..utils import count_non_negative_one
from ..LanguageModelHandler import LanguageModelHandler
from dotenv import dotenv_values



def step(t) -> None:
    
    shuffle(Agent.active_agents)
    for a in Agent.active_agents:
        a.random_walk_2()
        
    shuffle(Agent.active_agents)
    for a in Agent.active_agents:
        a.compete()
        #a.compete3()
        
    shuffle(Agent.active_agents)
    for a in Agent.active_agents:
        if uniform(0, 1) < simul_config["simulation"]["MUT_RATE"]:
            a.word = mutate(a.word)
   
# update function for graph
def update(t):
    global gstep
    gstep= t
    fig.clear()
    ax1= fig.add_subplot(gs[1,0])
    x= [a.x for a in agents]
    y= [a.y for a in agents]
    
    ax1.scatter(x, y, color='brown')
    ax1.axis([-1, W, -1, W])

    #overlays the words on the graph
    for a in agents:
        ax1.text(a.x, a.y, a.word, fontsize=8)
  
    #shows the frequency of the words with a histogram
    ax3= fig.add_subplot(gs[0,1])
    wds= [a.word for a in agents]
    ax3.hist(wds, N)
    ax3.tick_params(axis='x', rotation=90)
    ax3.set_title('word frequency')

    current_unique_words= set(wds)
    for w in current_unique_words:
        if w not in frequency:
            frequency[w]= [0] * (t+1)
            words.append(w)
    for w in words:
        frequency[w].append(len([a for a in agents if a.word==w]))

    ax4= fig.add_subplot(gs[0,2])
    for w in words:
        ax4.plot(frequency[w], label=w)
    ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=2, columnspacing=0.05)
    ax4.set_title('word frequency')
    ax4.set_xlabel('step')
    
    print(t, [a.word for a in agents])
    if isOutData:
        for i, a in enumerate(agents):
            f.write(str(t)+","+str(a.id)+","+str(a.x)+","+str(a.y)+","+'"'+a.word.replace('"','')+'"'+"\n")

    return(fig)

def set_seed(SEED: int) -> None:
    """
    Sets the seed for random number generators to ensure reproducibility.

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
    
def encode_dict(d):
    return {str(k): v for k, v in d.items()}

def decode_dict(d):
    return {tuple(eval(k)): v for k, v in d.items()}

def main() -> None:
    config_path: str
    logs_path: str
    #allow to input the config path and the logs path from the command line
    if len(argv) >= 2:
        config_path = argv[0]
        logs_path = argv[1]

    else:
        config_path = "config.json"
        logs_path = "logs.txt"

    simul_config: dict = load(open(config_path, "r")) #load the parameters from the JSON config file
    log_trial_res: bool = True #does the user want to log out the trial results?

    #set up the xperience output directory
    if not os.path.exists(simul_config["simulation"]["EXP_DIR"]):
        os.makedirs(simul_config["simulation"]["EXP_DIR"])
    
    
    #create and/or open the log file
    logs_file = open(
        f"{simul_config["simulation"]["EXP_DIR"]}/{logs_path}",
        "w+",
        encoding="utf-8",
    )
    #if the file is not empty, overwrite the content
    if logs_file.readlines() != []:
        logs_file.write("\n")
    
    gstep= 0
    
    #for the set number of trials
    for current_trial in range(0, simul_config["simulation"]["T"]):
        # initialize variables
        if 'model' in globals():
            model.__del__()

        model = LanguageModelHandler(
            model_name= simul_config["model"]["name"],
            auth_token= dotenv_values(".env")["HF_AUTH_TOKEN"],
        )
        
        judge = Judge(
            model= model,
            criteria= simul_config["judge"]["CRITERIA"],
            config_path= config_path,
        )

        set_seed(simul_config["simulation"]["SEED"] + current_trial)

        words = judge.create_word_list(simul_config["simulation"]["A"])
        
        for i in range(0, simul_config["simulation"]["N"]-1, 1):
            Agent(
                w= choice(words),
                id= i,
                W= simul_config["simulation"]["W"],
            )
            assert len(Agent.active_agents) == i + 1, f"Agent {i} not created."
            assert Agent.agents_pos.count(Agent) == len(Agent.active_agents), f"Agent {i} not in agents_pos."

        all_positions = sample([
            (x, y) for y in range(simul_config["simulation"]["W"]) 
            for x in range(simul_config["simulation"]["W"])
        ], simul_config["simulation"]["N"])

        for i, a in enumerate(Agent.active_agents):
            a.x = all_positions[i][0]
            a.y = all_positions[i][1]
            Agent.agents_pos[a.x][a.y] = a

        frequency= {}
        for w in words:
            frequency[w]= [len([a for a in Agent.active_agents if a.word == w])]

        fig = plt.figure(figsize=(10, 5))
        gs = gridspec.GridSpec(2, 3)

        if log_trial_res:
            trail_res_logs = open(f"{simul_config["simulation"]["EXP_DIR"]}/results_trial-{current_trial}.csv", "w+")
            trail_res_logs.write("gen,id,x,y,word\n")
            
            for i, a in enumerate(Agent.active_agents):
                f.write(str(0)+","+str(a.id)+","+str(a.x)+","+str(a.y)+","+'"'+a.word.replace('"','')+'"'+"\n")

        for i in range(simul_config["simulation"]["T"]):
            step(i)
            update(i)
        
        #at the end, saving the results, the close the files and free the memory
        final_trial_res_logs = open(f"{simul_config["simulation"]["EXP_DIR"]}/final_results_trial-{current_trial}.json", "w")
        dump(encode_dict(judge.judgments_history), f, indent=2)
        
        logs_file.close()
        trail_res_logs.close()
        final_trial_res_logs.close()
        judge.__del__()
        