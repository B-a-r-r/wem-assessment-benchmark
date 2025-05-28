# wem-assessment-bechmark
This project is **a specialized application made for the artificial life simulation model Evolutionary Ecology of Words**, in the context of its assesment as *an open-endedness driven benchmark for LLMs*.
The simulation parameters are full costummizable, even more abstract ones such as prompt engineering. At the end multiple visuals are generated with the output data,
allowing to analyze the results under different perspectives.

Ecology of Words is a simulation model developed by Dr. Reiji SUZUKI and Dr. Takaya ARITA, both from the ALIFE-CORE laboratory at Nagoya University.


## Project configuration
1. **Jump into the project root directory**.
<br>

2. Run the setup scripts:
    - `setup.sh` for Linux
    - `setup.bat` for Windows

    <ins>NB</ins>: this scripts work for CUDA 11.8, please check your GPU compatbility and CUDA version to modify the requirements if needed.
<br>

3. Create a `.env` file in the root directory of the project and add a variable named `HF_AUTH_TOKEN` with **your Hugging Face authentication token**.

## App usage
1. Create you own configuration **JSON file** containing your custom parameters, you can use the dedicated directory `custom_configs`. You can also use/modify `config.json` in the app directory, which is the default one provided.
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

## About the visuals
At the end of the simulation, these default visuals are generated:
- **TopB words** animations collection for each trial (`topB_animations_trialX.mp4`)
- **The UMAP trajectory graph** of the overall simulation (`trajectory_with_top.pdf`)
<br>

Howerver, you can customize the generation of visuals very easily by modifying the parameters in the `visuals_maker.py` file. For example, you can create a trajectory graph for each trial separately, or you can create an animation of the trajectory graph by calling the related method.
The visuals are stored in the experiment directory specified in the config file.

## About the data
**Four types of data** resulting from the simlation are logged (<ins>NB</ins>: *x* is the number of the trial):
- `results_x.csv` contains the listing of all the agents at each generation, and their properties.
    **<ins>Header</ins>: gen, id, x, y, word**
<br>

- `competition_history_x.json` contains the history of the competition between agents, with the following structure:
    ```json
    {
        "gen": {
            "(id1, id2)": winner_id | null 
        }
    }
    ```
    <ins>NB</ins>: if the reslt of a competition is `null`, it means the LLM could not provide proper decision.
<br>

- `judgement_history_x.json` contains the case law for all the competitions combinaisons that happened, with the following structure:
    ```json
    {
        "gen": {
            "(word1, word2)": winner_word
        }
    }
    ```
<br>

- `mutation_history_x.json` contains the history of the mutations that happened at each generation, with the following structure:
    ```json
    {
        "gen": {
            "source_word": {
                mutated_word: [
                    [
                        mutation_possibilities...
                    ],
                ]
            }
        }
    }
    ```
    <ins>NB</ins>: **a source word with multiple mutations** indicates that **several agents carried it and mutated**. If **a mutated word has multiple lists of possibilities** (among which it was chosen) it indicates that **the related source word mutated into this word multiple times**, with a certain range of possibilities each time.