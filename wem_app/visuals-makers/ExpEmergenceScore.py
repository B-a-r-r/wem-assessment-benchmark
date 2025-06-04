from WemExpData import WemExpData
from os import path, listdir, remove
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from moviepy import VideoFileClip
from scipy.signal import savgol_filter
from json import loads


class ExpEmergenceScore(WemExpData):
    """
    Class to compute the emergence score for each generation in the experiment.
    
    Objects attributes
    -------------------
    trials_emergences: dict[str, dict[int, list]]
        Dictionary containing each trial's emerged words for each generation.
    trials_emergence_score: dict[str, dict[int, float]]
        Dictionary containing each trial's emergence score for each generation and the average emergence score at each generation.
    local_score_plot_interval: int
        Interval for plotting local emergence scores.
    average_score_plot_interval: int
        Interval for plotting average emergence score.
    skip_frst_gen: bool
        Whether to skip the first generation when computing emergence scores.
    """
    
    def __init__(self, 
        folder_path: str, 
        label: str, 
        lang: str ="en", 
        seed: int =42,
        skip_no_mutations_gen: bool =True
    ):
        """
        Initialize the ExpEmergenceScoreGraph object.
        
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
        """
        super().__init__(folder_path, label, lang, seed, top_B=0)
        
        self.trials_emergences: dict[int, dict[int, list]] = {}
        self.trials_emergence_score: dict[int, dict[int, tuple[float, float]]] = {}
        self.trials_mutation_history: dict[int, dict[int, dict[str, dict[str, list[list[str]]]]]] = {}
        self.skip_no_mutations_gen: bool = skip_no_mutations_gen
        
        self._read_data()
        self._process_data()
        self._compute_emergence_scores()
    

    def _read_data(self):
        """
        Read all experiment data files and store them in all_data.
        Maps each trial to all its generation mapped to a list of words.
        The data is expected to be in CSV format, containing 'gen' and 'word' columns.
        """
        csv_idx = 0
        json_idx = 0
        for file in listdir(path.join(path.dirname(__file__), self.folder_path)):
            
            if file.endswith('csv'):
                if file.startswith('result'):
                    file_path = path.join(path.dirname(__file__), self.folder_path, file)
                    self.all_data[csv_idx] = (\
                        pd.read_csv(file_path, usecols=['gen', 'word'], encoding='utf-8')\
                        .groupby('gen')['word'].apply(list).to_dict()
                    )
                    csv_idx += 1
                
            if file.endswith('json'):
                if file.startswith('mutation_history'):
                    file_path = path.join(path.dirname(__file__), self.folder_path, file)
                    self.trials_mutation_history[json_idx] = {
                        int(gen): {
                            source: {
                                mut: possibilities
                                for mut, possibilities in mutations.items()
                            } for source, mutations in data.items()
                        } for gen, data in loads(open(file_path, 'r', encoding='utf-8').read(), parse_int=True, parse_float=True).items()
                    }
                    json_idx += 1

    def _process_data(self):
        """
        Process the experimental data to calculate the emergence of words in each trial.
        Computes unique words across all trials.
        """
        for file, data in self.all_data.items():
            self.trials_unique_words[file] = set()
            self.trials_emergences[file] = {}
            ancestors = set() #The words ecoutered from the beginning of the trial till the current generation
            
            #Retablish words from the initial list that mutated in gen 0 and
            #so that are no more visible in the all_data first generation. 
            #And remove the gen 0 emergences from the list of words
            sister_muts = set()
            for source, res in self.trials_mutation_history[file][0].items():
                data[0].append(source)
                
                for mut in res.keys():
                    sister_muts.add(mut)
                    data[0].pop(data[0].index(mut))
            
            ancestors.update(data[0])
            
            for mut in sister_muts:
                if mut not in ancestors:
                    data[0].append(mut)

            for gen, words in data.items():
                self.trials_emergences[file][gen] = [w for w in words if w not in ancestors]
                ancestors.update(words)
                
            self.trials_unique_words[file].update(ancestors)
    
    def _compute_emergence_scores(self):
        """
        Compute the emergence score for each generation, and the average one till that generation.
        The emergence score is defined as the ratio of emerged words to the total number of mutations that happened.
        It can be computed from local data, relative the one generation, or from the average data, relative to all generations.
        """
        for file, data in self.all_data.items():
            #Initializing computation variables
            total_encountered_emergences = 0
            total_mutations_happened = 0
            
            self.trials_emergence_score[file] = {}
            for gen in data.keys():
                
                mutations_now = 0
                #Each source word can have multiple mutations
                for mut in self.trials_mutation_history[file][gen].values():
                    for obtained_from in mut.values():
                        mutations_now += len(obtained_from)
                
                emergences_now = len(self.trials_emergences[file][gen])
                mutations_now += emergences_now - mutations_now if emergences_now > mutations_now else 0
                
                assert emergences_now <= mutations_now, f"Emergences should't be greater than mutations. {emergences_now} > {mutations_now} at gen {gen} in trial {file}."
                
                #Updating computation variables
                total_mutations_happened += mutations_now
                total_encountered_emergences += emergences_now
                
                #Computing score for this generation and omputing the average score taking into account the data of this generation
                average_emergence_score = total_encountered_emergences / (total_mutations_happened if total_mutations_happened > 0 else 1) #avoid division by zero
                local_emergence_score = emergences_now / (mutations_now if mutations_now > 0 else 1) #avoid division by zero
                
                self.trials_emergence_score[file][gen] = (round(local_emergence_score, 2), round(average_emergence_score, 2))
    
    def __repr__(self):
        return f"""
        Emergence Score Graph By Language
        --------------------------------
        Emergence Score: {self.trials_emergence_score.__repr__()}
        
        """
              
    def create_emergence_score_anim(self,
        output_file: str ="emergence_score",
        output_extension: str ="gif",
        figsize: tuple[int, int] =(20, 8),
        fontsize: int =12,
        only_trials: list[str] =None,
        uncommon_plot_interval: int =1,
        local_plot_interval: int =1,
        average_plot_interval: int =1,
        plateau_threshold: int =50,
        plot_std_dev_inner_scores: bool =True,
        animation_speed: int =200
    ):
        """
        Create an animated graph showing the evolution of emergence scores over generations for each trial.
        
        Parameters
        ----------
        output_file : str
            The base name of the output file (without extension).
        output_extension : str
            The file extension for the output animation. Must be either '.mp4' or '.gif'.
        figsize : tuple[int, int]
            The size of the figure for the animation.
        fontsize : int
            The font size for the text in the animation.
        only_trials : list[str], optional
            A list of trial filenames to include in the animation. If None, all trials are included.
        local_plot_interval : int
            The interval for plotting local emergence scores.
        average_plot_interval : int
            The interval for plotting average emergence scores.
        plateau_threshold : int
            The threshold for considering a plateau in the emergence score.
        plot_std_dev_inner_scores : bool
            Whether to plot the local emergence scores that are within the standard deviation range of the average score.
        animation_speed : int
            The speed of the animation in milliseconds between frames.
        """
        assert output_extension in ['mp4', 'gif'], "Output extension must be either 'mp4' or 'gif'."
        
        for idx, (file, data) in enumerate(self.trials_emergence_score.items()):
            if only_trials and idx not in only_trials:
                continue
            
            fig, subplot = plt.subplots(figsize=figsize)
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
                
                subplot.set_title(f"Emergence Score Over Generations", fontsize=fontsize)
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
            
            filename = path.join(path.dirname(__file__), self.folder_path, f"{output_file}_trial{file}.{output_extension}")
            anim.save(filename, writer='ffmpeg', fps=10)
            print(f"Animation saved as {filename}")
            
            plt.close(fig)

if __name__ == "__main__":
    def convert_gif_to_video(input_path: str, output_path: str, remove_input: bool =False):
        if not path.exists(input_path):
            raise FileNotFoundError(f"Input file {input_path} does not exist.")
        
        if not input_path.endswith('.gif'):
            raise ValueError("Input file must be a GIF.")
        
        clip = VideoFileClip(input_path)
        clip.write_videofile(output_path, codec='libx264')
        
        if remove_input:
            clip.close()
            remove(input_path)
    
    exps = ["makegraph-en-llama-2"]
    lang = "en"
    seed = 42
    
    for folder_path in exps:
        from ExpTopBAnimations import ExpTopBAnimations
        from ExpTrajectoryUmap import ExpTrajectoryUmap
        
        exp_top_b_animations = ExpTopBAnimations(folder_path, lang, seed)
        exp_top_b_animations.create_animatons_boards()
        
        exp_trajectory_umap_graph = ExpTrajectoryUmap(folder_path, lang, seed)
        exp_trajectory_umap_graph.create_trajectory_graph(output_extension="png")
        exp_trajectory_umap_graph.create_contour_plot(output_extension="png")
        exp_trajectory_umap_graph.create_trajectory_animation(only_trials=[0,1,2])
        
        emergence_score_graph = ExpEmergenceScore(folder_path, lang, seed)
        emergence_score_graph.create_emergence_score_anim(only_trials=[0,1])

        for file in listdir(path.join(path.dirname(__file__), folder_path)):
            if file.endswith('.gif'):
                convert_gif_to_video(
                    input_path=path.join(path.dirname(__file__), folder_path, file),
                    output_path=path.join(path.dirname(__file__), folder_path, file.replace('.gif', '.mp4')),
                    remove_input=True
                )