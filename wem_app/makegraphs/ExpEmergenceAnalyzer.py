from typing import Literal
from warnings import warn
from .WemExpData import WemExpData
from os import path, listdir, remove
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import savgol_filter
from json import loads
from .WemVisualsMaker import WemVisualsMaker

#
# author: Clément BARRIERE
# github: B-a-r-r
#

class ExpEmergenceAnalyzer(WemVisualsMaker):
    """
    Class to compute the emergence score for each generation in the experiment.
    
    Objects attributes
    -------------------
    exp_data : WemExpData
        An instance of WemExpData containing the experiment data.
    trials_gens_emergences : dict[int, dict[int, list]]
        A dictionary mapping each trial to its generations and the list of words that emerged in each generation.
    trials_gens_emergence_score : dict[int, dict[int, tuple[float, float]]]
        A dictionary mapping each trial to its generations and the emergence score for each generation.
    trials_gens_mutation_history : dict[int, dict[int, dict[str, dict[str, list[list[str]]]]]]
        A dictionary mapping each trial to its generations and the mutation history for each word. 
        A source word is mapped to its mutations and the lists of possibilities that led to each mutation.
    skip_no_mutations_gen : bool
        Whether to skip generations with no mutations when computing emergence scores.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the ExpEmergencesAnalyzerGraph object.
        
        Parameters
        ----------
        folder_path : str
            The path to the folder containing the experiment data files.
        label : str
            A label for the experiment, used for identification.
        lang : str, optional
            The language of the experiment data, default is 'en'.
        seed : int, optional
            The random seed for reproducibility, default is 42.
        skip_no_mutations_gen : bool, optional
            Whether to skip generations with no mutations when computing emergence scores, default is True.
        exp_data : WemExpData, optional
            An instance of WemExpData containing the experiment data. If provided, it will be used instead of reading from files.
        """
        assert 'folder_path' in kwargs or 'exp_data' in kwargs, "Either provide an experiment folder path or an instance of existing experiment data."
        
        exp_data = kwargs.get('exp_data', None)
        if exp_data is None:
            exp_data = WemExpData(
                folder_path=kwargs.get('folder_path'),
                label=kwargs.get('label', ''),
                seed=kwargs.get('seed', 42),
                lang=kwargs.get('lang', 'en'),
                top_B=kwargs.get('top_B', 10),
            )
        
        self.exp_data: WemExpData = exp_data
        self.trials_gens_emergences: dict[int, dict[int, list]] = {}
        self.trials_gens_emergence_score: dict[int, dict[int, tuple[float, float]]] = {}
        self.trials_gens_mutation_history: dict[int, dict[int, dict[str, dict[str, list[list[str]]]]]] = {}
        self.skip_no_mutations_gen: bool = kwargs.get('skip_no_mutations_gen', True)
        
        self._plt_font_for_lang() 
        self._read_data()
        self._process_data()
        self._compute_emergence_scores()
        self.exp_data.folder_path = self._create_dedicated_dir(path.join(self.exp_data.folder_path, "emergence-analysis"))
    
    def _plt_font_for_lang(self) -> None:
        """
        Set the matplotlib font for the specified language.
        """
        super()._plt_font_for_lang(lang=self.exp_data.lang)
    
    def import_exp_data(self, exp_data: WemExpData):
        """
        Import existing experiment data into the analyzer.
        
        Parameters
        ----------
        exp_data : WemExpData
            An instance of WemExpData containing the experiment data to import.
        """
        assert isinstance(exp_data, WemExpData), "Provided data to import must be an instance of WemUmapData."
        self.exp_data = exp_data

    def export_exp_data(self) -> WemExpData:
        """
        Export the processed experiment data as a WemUmapData object.
        
        Returns
        -------
        WemExpData:
            An instance of WemExpData containing the current object's experiment data.
        """
        return self.exp_data
    
    def _create_dedicated_dir(self, folder_path) -> str:
        """
        Create a dedicated directory for saving visuals if it does not exist.
        If the directory already exists, it does nothing.
        
        Parameters
        ----------
        folder_path : str
            The path to the folder where the visuals will be saved.
            
        Returns
        -------
        str:
            The absolute path to the created or existing directory.
        """
        return super()._create_dedicated_dir(folder_path)
    
    def _read_data(self):
        """
        Read all experiment data files.
        If experiment data has been imported, it checks the type of the first element in the data.
        Here the unitary type of data has to be a string representing words.
        If the data is not in the expected format, it rereads the CSV and JSON files from the experiment folder.
        """
        first_element = None
        if not self.exp_data.is_blank():
            first_element = self.exp_data.all_data[0][0][0]
            
        tmp = type(first_element)
        if not self.exp_data.is_blank() and not tmp == str:
            warn("The units in the provided experiment data must be string " \
                + f"representing words to create emergence visuals, found {tmp} instead.\n" \
                + "NB: dict type is used to create topB animation visuals.\n" \
                + "---> Rereading experiment data...\n"
            )
        
        if self.exp_data.is_blank() or not tmp == str:
            csv_idx = 0
            json_idx = 0
            for file in listdir(path.join(path.abspath(self.exp_data.folder_path))):
                
                if file.endswith('csv'):
                    if file.startswith('result'):
                        file_path = path.join(path.abspath(self.exp_data.folder_path), file)
                        self.exp_data.all_data[csv_idx] = (\
                            pd.read_csv(file_path, usecols=['gen', 'word'], encoding='utf-8')\
                            .groupby('gen')['word'].apply(list).to_dict()
                        )
                        csv_idx += 1
                    
                if file.endswith('json'):
                    if file.startswith('mutation_history'):
                        file_path = path.join(path.abspath(self.exp_data.folder_path), file)
                        self.trials_gens_mutation_history[json_idx] = {
                            int(gen): {
                                source: {
                                    mut: possibilities
                                    for mut, possibilities in mutations.items()
                                } for source, mutations in data.items()
                            } for gen, data in loads(open(file_path, 'r', encoding='utf-8').read(), parse_int=True, parse_float=True).items()
                        }
                        json_idx += 1
                        
            if csv_idx == 0:
                warn("No CSV files found in the provided folder path.")
            if json_idx == 0:
                warn("No JSON files found in the provided folder path.")

    def _process_data(self):
        """
        Process the experimental data to compute attributes.
        """
        for file, data in self.exp_data.all_data.items():
            self.exp_data.trials_gens_unique_words[file] = set()
            self.trials_gens_emergences[file] = {}
            ancestors = set() #The words ecoutered from the beginning of the trial till the current generation
            
            #Retablish words from the initial list that mutated in gen 0 and
            #so that are no more visible in the all_data first generation. 
            #And remove the gen 0 emergences from the list of words
            sister_muts = set()
            for source, res in self.trials_gens_mutation_history[file][0].items():
                data[0].append(source)
                
                for mut in res.keys():
                    sister_muts.add(mut)
                    data[0].pop(data[0].index(mut))
            
            ancestors.update(data[0])
            
            for mut in sister_muts:
                if mut not in ancestors:
                    data[0].append(mut)

            for gen, words in data.items():
                self.trials_gens_emergences[file][gen] = [w for w in words if w not in ancestors]
                ancestors.update(words)
                
            self.exp_data.trials_gens_unique_words[file].update(ancestors)
    
    def _compute_emergence_scores(self):
        """
        Compute the emergence score for each generation, and the average one till that generation.
        The emergence score is defined as the ratio of emerged words to the total number of mutations that happened.
        It can be computed from local data, relative the one generation, or from the average data, relative to all generations.
        """
        for file, data in self.exp_data.all_data.items():
            #Initializing computation variables
            total_encountered_emergences = 0
            total_mutations_happened = 0
            
            self.trials_gens_emergence_score[file] = {}
            for gen in data.keys():
                
                mutations_now = 0
                #Each source word can have multiple mutations
                for mut in self.trials_gens_mutation_history[file][gen].values():
                    for obtained_from in mut.values():
                        mutations_now += len(obtained_from)
                
                emergences_now = len(self.trials_gens_emergences[file][gen])
                mutations_now += emergences_now - mutations_now if emergences_now > mutations_now else 0
                
                assert emergences_now <= mutations_now, f"Emergences should't be greater than mutations. {emergences_now} > {mutations_now} at gen {gen} in trial {file}."
                
                #Updating computation variables
                total_mutations_happened += mutations_now
                total_encountered_emergences += emergences_now
                
                #Computing score for this generation and omputing the average score taking into account the data of this generation
                average_emergence_score = total_encountered_emergences / (total_mutations_happened if total_mutations_happened > 0 else 1) #avoid division by zero
                local_emergence_score = emergences_now / (mutations_now if mutations_now > 0 else 1) #avoid division by zero
                
                self.trials_gens_emergence_score[file][gen] = (round(local_emergence_score, 2), round(average_emergence_score, 2))
    
    def __repr__(self):
        return f"""
        Emergence Score Graph By Language
        --------------------------------
        Emergence Score: {self.trials_gens_emergence_score.__repr__()}
        
        """
              
    def create_score_evo_anim(self,
        output_prefix: str ="score_evo_anim",
        output_extension: Literal['gif', 'mp4'] ="gif",
        fig_size: tuple[int, int] =(20, 8),
        fontsize: int =12,
        only_trials: list[str] =None,
        uncommon_plot_interval: int =1,
        local_plot_interval: int =1,
        average_plot_interval: int =1,
        plateau_threshold: int =50,
        plot_std_dev_inner_scores: bool =True,
        animation_speed: int =200,
        subplot: plt.Axes =None,
        save: bool =True,
    ) -> list[FuncAnimation]:
        """
        Create an animation of the emergence score evolution over generations for each trial.
        
        Parameters
        ----------
        output_prefix : str
            Prefix for the output file name.
        output_extension : Literal['gif', 'mp4']
            File format for saving the plot, either 'gif' or 'mp4'.
        fig_size : tuple[int, int]
            Size of the figure in inches (width, height).
        fontsize : int
            Font size for the plot labels and title.
        only_trials : list[str], optional
            A list of trial indices to include in the animation. If None, all trials are included.
        uncommon_plot_interval : int
            Interval for plotting uncommon local emergence scores (those outside the standard deviation range).
        local_plot_interval : int
            Interval for plotting common local emergence scores (those within the standard deviation range).
        average_plot_interval : int
            Interval for plotting the average emergence score.
        plateau_threshold : int
            The number of generations to consider a plateau in the average emergence score.
        plot_std_dev_inner_scores : bool
            Whether to plot the common local emergence scores within the standard deviation range.
        animation_speed : int
            Speed of the animation in milliseconds.
        subplot : plt.Axes, optional
            A matplotlib Axes object to plot on. If None, a new figure and axes will be created.
        save : bool
            Whether to save the animation to a file. If True, the animation will be saved with the specified output prefix and extension.
        
        Returns
        -------
        list[FuncAnimation]:
            A list of FuncAnimation objects for each trial's emergence score evolution animation.
        """
        assert uncommon_plot_interval > 0, "Uncommon plot interval must be greater than 0."
        assert local_plot_interval > 0, "Local plot interval must be greater than 0."
        assert average_plot_interval > 0, "Average plot interval must be greater than 0."
        assert plateau_threshold > 0, "Plateau threshold must be greater than 0."
        
        only_trials = None if only_trials == [] else only_trials
        
        returned_animations = []
        
        for idx, (file, data) in enumerate(self.trials_gens_emergence_score.items()):
            if only_trials and idx not in only_trials:
                continue
            
            fig = plt.figure(figsize=fig_size) if subplot is None else subplot.figure
            subplot = fig.add_subplot(111, label='emergence_score_anim') if subplot is None else subplot
            generations = sorted(list(data.keys()))
            average_scores = [data[gen][1] for gen in generations]
            local_scores = [data[gen][0] for gen in generations]
            
            #Store the standard deviation of the local emergence scores for each generation
            std_dev_maps = {}
            #Smoothed curve y of the std. dev. of the local emergence scores around the average emergence score
            average_score_y_pos = []
            average_score_y_neg = []
            #Identify the common and uncommon local emergence scores (in and out of the standard deviation range from the average emergence score)
            common_local_scores_idx = []
            uncommonly_low_local_scores_idx = []
            uncommonly_high_local_scores_idx = []
            #Calculate the plateaux, which can be defined by the maintaining of the average emergence score for a certain number of generations
            plateaux = []
            is_plateau = [-1]
            
            for gen in generations:
                std_dev_maps[gen] = round(np.std(local_scores[:gen+1]), 3)
                avg_pos = average_scores[gen] + std_dev_maps[gen]
                avg_neg = average_scores[gen] - std_dev_maps[gen]
                
                average_score_y_pos.append(avg_pos)
                average_score_y_neg.append(avg_neg)
                
                if (local_scores[gen] <= avg_neg):
                    uncommonly_low_local_scores_idx.append(gen)
                
                elif (local_scores[gen] >= avg_pos):
                    uncommonly_high_local_scores_idx.append(gen)
                    
                else:
                    common_local_scores_idx.append(gen)
                    
                if np.isclose(average_scores[gen], is_plateau[-1], rtol=1e-02):
                    is_plateau.append(gen)
                    
                else:
                    is_plateau = [gen]
                
                if len(is_plateau) >= plateau_threshold:
                    plateaux.append(is_plateau)
            
            #Smoothed curve y of the average emergence score evolution across generations
            average_score_y_smooth = savgol_filter(average_scores, 30, 5)
            average_score_y_pos_smooth = savgol_filter(average_score_y_pos, 30, 5)
            average_score_y_neg_smooth = savgol_filter(average_score_y_neg, 30, 5)
            
            #merge uncommonly high and low local scores indices
            uncommon_local_scores_idx = sorted(uncommonly_high_local_scores_idx + uncommonly_low_local_scores_idx)
            
            def update(frame):
                subplot.clear()
                    
                #The minimum and maximum emergence scores encoutered till this generation
                min_score = min(local_scores[:frame+1])
                max_score = max(local_scores[:frame+1])
                
                #Only plot every average_plot_interval frames
                plot_frames = [i for i in range(0, frame+1, average_plot_interval)]
                subplot.plot([generations[i] for i in plot_frames], [average_score_y_smooth[i] for i in plot_frames], label=f"Average Emergence Score", color='blue', alpha=0.6)
                
                #Plot the positive and negative standard deviation curves of the local scores around the average score
                subplot.plot([generations[i] for i in plot_frames], [average_score_y_pos_smooth[i] for i in plot_frames], color='green', alpha=0.1, linestyle='--')
                subplot.plot([generations[i] for i in plot_frames], [average_score_y_neg_smooth[i] for i in plot_frames], color='red', alpha=0.1, linestyle='--')
                subplot.fill_between(generations[:frame+1], average_score_y_smooth[:frame+1], average_score_y_pos_smooth[:frame+1], label=f"Std. Dev. +", color='green', alpha=0.05)
                subplot.fill_between(generations[:frame+1], average_score_y_smooth[:frame+1], average_score_y_neg_smooth[:frame+1], label=f"Std. Dev. -", color='red', alpha=0.05)
                
                #Plot the uncommon local emergence scores
                plot_frames = [i for i in range(0, frame+1, uncommon_plot_interval)]
                subplot.plot([generations[i] for i in uncommon_local_scores_idx if (i in plot_frames and i <= frame)], [local_scores[i] for i in uncommon_local_scores_idx if (i in plot_frames and i <= frame)], linestyle="none", marker='o', color='red', alpha=0.4, label=f"Local Emergence Scores outside Std. Dev.", markersize=4)
                
                #Plot the common local emergence scores
                if plot_std_dev_inner_scores:
                    plot_frames = [i for i in range(0, frame+1, local_plot_interval)]
                    subplot.plot([generations[i] for i in common_local_scores_idx if (i in plot_frames and i <= frame)], [local_scores[i] for i in common_local_scores_idx if (i in plot_frames and i <= frame)], linestyle="none", marker='o', color='orange', alpha=0.4, label=f"Local Emergence Scores within Std. Dev.", markersize=4)
                
                #Plot plateaux
                for plateau in plateaux:
                    if frame in plateau:
                        subplot.plot([generations[i] for i in plateau], [average_scores[i] for i in plateau], color='black', linestyle='--', alpha=0.6, label=f"Average Score Plateau (at least {plateau_threshold} generations)")
                
                subplot.set_title(f"Emergence Score Over Generations\nExperiment '{self.exp_data.label}' trial {file}", fontsize=fontsize)
                subplot.set_xlabel("Generation", fontsize=fontsize)
                subplot.set_ylabel("Emergence Score", fontsize=fontsize)
                subplot.grid(True, linestyle='--', alpha=0.2)
                subplot.tick_params(axis='both', which='major', labelsize=fontsize*0.8)
                subplot.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fontsize*0.8)
                subplot.text(
                    1.03, 
                    0.6, 
                    f"Generation {generations[frame]}\n----------------\nCurrent local score:  {local_scores[frame]}\nMin score:  {min_score}\nMax score:  {max_score}\nStd. dev.:  {round(std_dev_maps[generations[frame]], 2)}", 
                    transform=subplot.transAxes, 
                    fontsize=fontsize*0.8,
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.6'),
                )
                
                fig.tight_layout()
        
            anim = FuncAnimation(fig, update, frames=len(generations), interval=animation_speed, repeat=False)
            returned_animations.append(anim)
            
            if save:
                self._save_visual(anim, f"{output_prefix}_{file}", output_extension)
            
        return returned_animations

    def _save_visual(self, fig: plt.Figure | FuncAnimation, output_prefixe: str, output_extension: str):
        """
        Save the created visual to a file.
        
        Parameters
        -----------
        fig : plt.Figure | FuncAnimation
            The figure or animation to save.
        output_prefixe : str
            Prefix for the output file name
        output_extension : Literal['pdf', 'png']
            File format for saving the plot
        """
        filename = path.join(self.exp_data.folder_path, f"{output_prefixe}{"_" if self.exp_data.label is not None else ""}{self.exp_data.label}.{output_extension}")
        super()._save_visual(fig, filename, output_extension)
    
    def convert_to_video(self, targeted_extensions: list[str], remove_input: bool =True) -> None:
        """
        Convert all files with specified extensions in the experiment folder to video format (.mp4).
        
        Parameters
        ----------
        targeted_extensions : list[str]
            A list of file extensions to target for conversion (e.g., ['.gif', '.avi']).
        remove_input : bool, optional
            If True, the input files will be removed after conversion. Default is True.
        """
        for file in listdir(path.abspath(self.exp_data.folder_path)):
            for ext in targeted_extensions:
                if file.endswith(ext):
                    input_path = path.join(self.exp_data.folder_path, file)
                    output_path = path.join(self.exp_data.folder_path, file.replace(ext, '.mp4'))
                    super().convert_to_video(input_path, output_path, remove_input)


if __name__ == "__main__":
    e = ExpEmergenceAnalyzer(
        folder_path="makegraph-en-llama-2",
        label="label",
        lang="en",
        seed=42,
        skip_no_mutations_gen=True,
    )
    
    e.create_score_evo_anim(only_trials=[0, 1])
    e.convert_to_video(targeted_extensions=['.gif'], remove_input=True)