# wem-assessment-bechmark
This project is **a specialized application made for the artificial life simulation model Evolutionary Ecology of Words**, in the context of its assesment as *an open-endedness driven benchmark for LLMs*.
The simulation parameters are full costummizable, even more abstract ones such as prompt engineering. At the end multiple visuals are generated with the output data, allowing to analyze the results under different perspectives.

Evolutionary Ecology of Words is an artificial life experiment model developed by Pr. Reiji SUZUKI and Pr. Takaya ARITA, both from the ALIFE-CORE laboratory at Nagoya University.


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
        "verbose": true,                        #Whether to print the progress of the simulation in the terminal    
        "exp_dir": "...",                       #The path to the (existing or not) directory in the project to store the experiment data
        "log_judgement_history": true,          #Whether to log the judgement history of the competitions
        "log_trial_results": true,              #Whether to log the results of each trial
        "log_competition_history": true,        #Whether to log the history of the competitions
        "log_mutation_history": true            #Whether to log the history of the mutations
        "lang": "en",                           #The language of the prompts
        "label": "default",                     #The label of the experiment, used to identify it and name the files for instance
        "top_B": 10,                            #The number of top words to consider for the visuals 
        "sentence_transformer_model": "..."     #The name of the sentence transformer model to use to ompute UMAP data
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
    - `wem.sh <config_path> <enable_logs> <enable_real_time_views>` for Linux
    - `wem.bat <config_path> <enable_logs> <enable_real_time_views>` for Windows

- where `config_path` (optional) is the absolute path to your configuration file. *If you don't provide a path, the default one will be used*.
- where `enable_logs` (optional) is a boolean value that enables or disables the logging file of the running simulation. If you don't provide a value, set to *True by default*.
- where `enable_real_time_views` (optional) is a boolean value that enables or disables the real-time updated graphs of the simulation. If you don't provide a value, set to *False by default*.

<ins>NB</ins>: if any trouble with the scripts, you can run the app directly with Python:
```

python wem_app/wem_main.py <config_path> <enable_logs> <enable_real_time_views>
```
<br>

While the simulation is runnning, the `verbose` parameter in the config file allows to track the progress of the operations in the terminal. If you enabed the logs, the log file will contain more detailed information about errors, warnings, current state, etc.. .

All the data from the simulation is stored in the experiment directory specified in the config file.

#### Generate visuals
At the end of the experiment, once the data files are logged, you can generate visuals. To do so, you can run `visuals_maker.py` as follow:
```
python wem_app/visuals_maker.py <config_path> [--ui]
```
- where `config_path` (mandatory) is the absolute path to your configuration file. *If you don't provide a path, the default one will be used*.
- where `--ui` (optional) is the option to enable the *user interface* for the visuals generation, allowing to select the files and parameters interactively.

## About the data
**Four types of data** resulting from the experiment are logged (<ins>NB</ins>: *x* is the number of the trial):
- `results_x.csv` contains the listing of all agents at each generation, and their properties.
    **<ins>Header</ins>: gen, id, x, y, word**
<br>

- `competition_history_x.json` contains the history of the competition between agents, with the following structure:
    ```json
    {
        "gen": {
            "(competitor1_id, competitor2_id2)": "winner_id | null"
        }
    }
    ```
    <ins>NB</ins>: if the result of a competition is `null`, it means the LLM could not provide proper decision.
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
The application includes several `analyzer` classes which are used to generate different types of visuals.
Each type of visuals is saved in a dedicated folder, created in the experiement directory:
- `topB-analysis`
- `emergence-analysis`
- `trajectory-analysis`
- `spatial-analysis`

#### Top B analysis
This analysis generates visuals based on the top B words at each generation, allowing to track their evolution over time.

- **TopB Words Histogram**: shows the distribution of the top B words at each generation.
![top_B_histogram](./assets/topB_hist.png)

- **TopB Words Frequency**: shows the frequency evolution of each word in the final top B.
![top_B_frequency](./assets/topB_freq_plot.png)

- **Agent Position**: shows the position of each agent in the simulated space at each generation.
![agent_pos_plot](./assets/agent_pos_plot.png)

#### Emergence analysis
This analysis generates visuals based on the emergence of new words along the simulation.

- **Emergence Score Evo**: the emergence score is the ratio of the number of emergences and the number of mutations at each generation. This metric allows to track one aspect of the effectiveness of the trajectory, as a high scores are correlated to the discover of new semantic fields and low scores to the return to a previously explored one. The behavior of the average of this score is also relevant: a stagnation is showed by a quickly decreasing curve, while a stable score is showed by a flat curve.
![emergence_score_evo](./assets/emergence_score_evo_plot.png)

#### Trajectory analysis
This analysis generates visuals based on the semantic trajectory, draw by the plot of the average semantic vector of the simulation every x steps. The average semantic vector of the simulation is optained by averaging the umap semantic representation of each words in the simulation at each generation.

- **Trajectory Plot**: shows the trajectory of the average semantic vector of the simulation.
![trajectory_plot](./assets/trajectory_plot.png)

- **Animated Trajectory Plot**: for a specific trial, the animation of the trajectory along the generations.

##### Spatial analysis
This analysis generates visuals based on the spatial properties of the umap representation. It allows to measure metrics such as exploration, exploitation, and redondancy of the average semantic vectors in the semantic space.

- **Areas Contours**: shows the areas containing the average semantic vectors, following certain percentiles.
![areas_contours](./assets/areas_plot.png)

- **Exploration Coverage**: show the convex hull of the plotted average semantic vectors, allowing to visualize the exploration of the semantic space.
![exploration_coverage](./assets/exploration_coverage_plot.png)

- **Density Plot**: shows the density of the average semantic vectors in the semantic space, allowing to visualize the exploitation of the semantic space.
![density_plot](./assets/density_plot.png)

- **Metrics Textblock**: shows the metrics computed from the spatial analysis, such as the exploration, exploitation, and redondancy scores.
![metrics_textblock](./assets/spatial_metrics_textblock.png)