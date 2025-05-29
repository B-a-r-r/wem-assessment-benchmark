# wem-assessment-bechmark
This project is **a specialized application made for the artificial life simulation model Evolutionary Ecology of Words**, in the context of its assesment as *an open-endedness driven benchmark for LLMs*.
The simulation parameters are full costummizable, even more abstract ones such as prompt engineering. At the end multiple visuals are generated with the output data, allowing to analyze the results under different perspectives.

Evolutionary Ecology of Words is a simulation model developed by Dr. Reiji SUZUKI and Dr. Takaya ARITA, both from the ALIFE-CORE laboratory at Nagoya University.


## Project configuration
1. **Jump into the project root directory**.
<br>

2. Run the setup scripts:
    - `setup.sh` for Linux
    - `setup.bat` for Windows

    <ins>NB</ins>: this scripts work for CUDA 11.8, please check your GPU compatbility and CUDA version to modify the requirements if needed.
<br>

3. Create a `.env` file in the root directory of the project and add a variable named `HF_AUTH_TOKEN` with **your Hugging Face authentication token**.

## Config file
The simulation is conigured through a **JSON config file**. It includes all the parameters for the **simulation**, but also the **LLM** parameters and the **workspace** variables.
The config file is stuctured as follows:
```
{
    "prompts": {
        "create": "...",    #The prompt used to create the initial words list
        "judge": "...",     #The prompt used to judge competiting words
        "mutate": "...",    #The prompt used to mutate a word
        "prefix": "...",    #The prompt prefix used for all the LLM calls
    },
    "simulation" : {
        "N": 80,                    #Number of agents in the simulation (population size)
        "S": 300,                   #Number of generations in the simulation
        "W": 16,                    #Size of the simulated space, where agents move and compete
        "SEED" : 42,                #Random seed for reproducibility
        "A": 35,                    #Size of the initial words list
        "N_WALK": 1,                #Number of steps each agent takes in the simulated space at each generation
        "MUT_PROB": 0.05,           #Probability of mutation for each agent at the end of the generation
        "T": 10,                    #Number of trials to run the simulation
        "B": 10,                    #Size of the mutation posibilities list, among which the actual mutation is chosen
        "CRITERIA": "stronger",     #Criteria for the competition between words
        "WORD_MAX_LENGHT": 20,      #Maximum length of a word
        "WORD_MIN_LENGHT": 3,       #Minimum length of a word
        "A_ALLOWED_DELTA": 5,       #Acceptable error margin between the expected size A and the actual one
        "B_ALLOWED_DELTA": 3        #Acceptable error margin between the expected size B and the actual one
    },
    "workspace": {
        "verbose": true,                    #Whether to print the progress of the simulation in the terminal    
        "exp_dir": "...",                   #The path to the (existing or not) directory in the project to store the experiment data
        "log_judgement_history": true,      #Whether to log the judgement history of the competitions
        "log_trial_results": true,          #Whether to log the results of each trial
        "log_competition_history": true,    #Whether to log the history of the competitions
        "log_mutation_history": true        #Whether to log the history of the mutations
    },
    "model" : {
        "name": "...",               #The name of the LLM model to use
        "local_offload": false,      #Whether to offload the model to the local disk
        "quantization": 8,           #The quantization level to apply (8, 4, anything else is not considered)
        "temperature": 0.8,          #The temperature to use for the LLM generation
        "use_gpu": true,             #Whether to use the GPU for the LLM generation
        "rep_penalty": 2.0,          #The repetition penalty to apply to the LLM generation
        "max_tokens_per_word": 5     #The maximum number of tokens to generate for each word
    }
}
```
<ins>NB</ins>: the file `config.json` is the default configuration file provided with the app.


## App usage
1. Create your own configuration **JSON file** containing your custom parameters, you can use the dedicated directory `custom_configs`. You can also use/modify `config.json` in the app directory, which is the default one provided.
<br>

2. Run the app by executing the scripts **from the root directory**:
    - `wem.sh [config_path] [enable_logs]` for Linux
    - `wem.bat [config_path] [enable_logs]` for Windows

- where `config_path` is the absolute path to your configuration file. *If you don't provide a path, the default one will be used*.
- where `enable_logs` is a boolean value that enables or disables the logging file of the running simulation. If you don't provide a value, set to *True by default*.

<ins>NB</ins>: if any trouble with the scripts, you can run the app directly with Python:
```bash

python wem_app/wem_main.py [config_path] [enable_logs]
```
<br>

While the simulation is runnning, the `verbose` parameter in the config file allows to track the progress of the operations in the terminal. If you enabed the logs, the log file will contain more detailed information about errors, warnings, current state, etc.. .

All the data from the simulation is stored in the experiment directory specified in the config file.

## About the data
**Four types of data** resulting from the simlation are logged (<ins>NB</ins>: *x* is the number of the trial):
- `results_x.csv` contains the listing of all the agents at each generation, and their properties.
    **<ins>Header</ins>: gen, id, x, y, word**
<br>

- `competition_history_x.json` contains the history of the competition between agents, with the following structure:
    ```json
    {
        "gen": {
            "(id1, id2)": "winner_id | null"
        }
    }
    ```
    <ins>NB</ins>: if the reslt of a competition is `null`, it means the LLM could not provide proper decision.
<br>

- `judgement_history_x.json` contains the case law for all the competitions combinaisons that happened, with the following structure:
    ```json
    {
        "gen": {
            "(word1, word2)": "winner_word"
        }
    }
    ```
<br>

- `mutation_history_x.json` contains the history of the mutations that happened at each generation, with the following structure:
    ```json
    {
        "gen": {
            "source_word": {
                "mutated_word": [
                    [
                        "mutation_possibilities..."
                    ],
                    [
                        "mutation_possibilities..."
                    ],
                ],
            }
        }
    }
    ```
    <ins>NB</ins>: **a source word with multiple mutations** indicates that **several agents carried it and mutated**. If **a mutated word has multiple lists of possibilities** (among which it was chosen) it indicates that **the related source word mutated into this word multiple times**, with a certain range of possibilities each time.

## About the visuals
At the end of the simulation, these default visuals are generated:
- **TopB words** animations collection for each trial (`topB_animations_trialX.mp4`)
- **The UMAP trajectory graph** of the overall simulation (`trajectory_with_top.pdf`)
<br>

Howerver, you can customize the generation of visuals very easily by modifying the parameters in the `visuals_maker.py` file. For example, you can create a trajectory graph for each trial separately, or you can create an animation of the trajectory graph by calling the related method.
The visuals are stored in the experiment directory specified in the config file.