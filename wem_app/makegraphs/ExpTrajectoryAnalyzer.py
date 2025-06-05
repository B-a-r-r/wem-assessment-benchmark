from os import listdir, path
from typing import Literal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import Counter
from WemVisualsMaker import WemVisualsMaker
from WemUmapData import WemUmapData
from warnings import warn

#
# author: Reiji SUZUKI et al.
# refactor: Clément BARRIERE
#

class ExpTrajectoryAnalyzer(WemVisualsMaker):
    """
    Class to create trajectory graphs and metrics from the data of an experiment.
    
    Object Attributes
    ----------------
    exp_data : WemExpData
        An object containing the experiment loaded data and umap representations.
    """
    
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        folder_path : str
            The path to the folder containing the experiment data. Mendatory if exp_data is not provided.
        label : str, default ''
            The label of the experiment. Added to output file names.
        lang : str, default 'en'
            The natural language used in the experiment (two letters code, e.g., 'en' for English, 'ch' for Chinese).
        sentence_transformer_model : str, default 'all-MiniLM-L6-v2'
            The sentence transformer model to use, by default 'all-MiniLM-L6-v2'.
        seed : int, default 42
            The seed to use for random operations, by default 42.
        top_B : int, default 10
            The number of top B words to extract for each file, by default 10.
        exp_data : WemUmapData
            An existing instance of WemUmapData, by default None.
            If provided everything else is ignored.
        """
        assert 'folder_path' in kwargs or 'exp_data' in kwargs, "Either provide an experiment folder path or an instance of existing experiment data."
        
        exp_data = kwargs.get('exp_data', None)
        if exp_data is None:
            exp_data = WemUmapData(
                folder_path=kwargs.get('folder_path'),
                label=kwargs.get('label', ''),
                seed=kwargs.get('seed', 42),
                lang=kwargs.get('lang', 'en'),
                top_B=kwargs.get('top_B', 10),
                model=kwargs.get('sentence_transformer_model', 'all-MiniLM-L6-v2')
            )
        
        self.exp_data: WemUmapData = exp_data
        
        self._plt_font_for_lang() 
        self._read_data()
        self._process_data()
        self.exp_data.folder_path = self._create_dedicated_dir(path.join(self.exp_data.folder_path, "trajectory-analysis"))
    
    def _plt_font_for_lang(self) -> None:
        """Set the matplotlib font for the specified language."""
        super()._plt_font_for_lang(lang=self.exp_data.lang)
    
    def import_exp_data(self, exp_data: WemUmapData):
        """
        Import existing experiment data into the analyzer.
        
        Parameters
        ----------
        exp_data : WemUmapData
            An instance of WemUmapData containing the experiment data to import.
        """
        assert isinstance(exp_data, WemUmapData), "Provided data to import must be an instance of WemUmapData."
        self.exp_data = exp_data

    def export_exp_data(self) -> WemUmapData:
        """Export the processed experiment data as a WemUmapData object."""
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
        
    def _read_data(self) -> None:
        """
        Read all experiment data files and store them in all_data.
        The data is should be in CSV format, containing 'gen' and 'word' columns.
        If an existing experiment data object has been provided, ensure its format 
        match this class requirements.
        """
        first_element = None
        if not self.exp_data.is_blank():
            first_element = self.exp_data.all_data[0][0][0]
            
        tmp = type(first_element)
        if not self.exp_data.is_blank() and not tmp == str:
            warn("The units in the provided experiment data must be string " \
                + f"representing words to create trajectory visuals, found {tmp} instead.\n" \
                + "NB: dict type is used to create topB animation visuals.\n" \
                + "---> Rereading experiment data...\n"
            )
            
        if self.exp_data.is_blank() or not tmp == str:
            csv_idx = 0
            for file in listdir(path.abspath(self.exp_data.folder_path)):
                if file.endswith('csv'):
                    file_path = path.join(self.exp_data.folder_path, file)
                    self.exp_data.all_data[csv_idx] = (\
                        pd.read_csv(file_path, usecols=['gen', 'word'], encoding='utf-8')\
                        .groupby('gen')['word'].apply(list).to_dict()
                    )
                    csv_idx += 1
            
            if csv_idx == 0:
                warn("No CSV files found in the provided folder path.")
                    
    def _process_data(self) -> None:
        """
        From the read data, compute average semantic verctor coordinates for each gen in each trial, 
        the top B words for each gen in each trial, the top B words for each trial, and the color for 
        each unique word in a trial.
        """
        if self.exp_data.is_empty():
            for file, data in self.exp_data.all_data.items():
                
                self.exp_data.trials_gens_unique_words[file] = {}
                self.exp_data.trials_gens_count_words[file] = {}
                word_counts = None
                
                for gen, words in data.items():
                    self.exp_data.trials_gens_unique_words[file][gen] = set()
                    self.exp_data.trials_gens_unique_words[file][gen].update(words)
                    
                    word_counts = Counter(words)
                    
                    self.exp_data.trials_gens_count_words[file] = {}
                    self.exp_data.trials_gens_count_words[file][gen] = word_counts
            
            if self.exp_data.is_umap_blank():
                self.exp_data.vectorize_words()
                self.exp_data.umap_reduce()
                
            for file, data in self.exp_data.all_data.items():
                self.exp_data.avg_coords[file] = {}
                for gen, words in data.items():
                    coords = np.array([self.exp_data.word_to_umap[word] for word in words if word in self.exp_data.word_to_umap])
                    self.exp_data.avg_coords[file][gen] = coords.mean(axis=0) if len(coords) > 0 else np.array([np.nan, np.nan])
    
    def create_trajectory_graph(self, 
        fig_size: tuple =(9, 9), 
        font_size: int =12, 
        output_prefix: str ="trajectory",
        plot_interval: int =20,
        only_trials: list[int] =None,
        output_extension: Literal['pdf', 'png'] ="pdf",
        save: bool =True
    ) -> plt.Axes:
        """
        Create a plot of the experiment data, showing the trajectory of each file
        over generations. The plot is saved to a PDF file.

        Parameters
        ----------
        fig_size : tuple, optional
            The size of the figure, by default (9, 9)
        font_size : int, optional
            By default 12
        output_prefix : str, optional
            The file name to save the plot with (without extension and label), by default "trajectory_top_words.pdf"
        plot_interval : int, optional
            The interval at which to plot the coordinates, by default 20
        only_trials : list[int], optional
            A list of indices of trials to plot. If None, all trials are plotted, by default None
        output_extension : str, optional
            The file extension for the output plot, either 'pdf' or 'png', by default 'pdf'
        """
        assert plot_interval > 0, "plot_interval must be a positive integer less than or equal to the number of generations in the simulation."
        only_trials = None if only_trials == [] else only_trials
            
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.exp_data.avg_coords.keys())))
        
        for idx, (file, coords) in enumerate(self.exp_data.avg_coords.items()): 
            if only_trials is not None and file not in only_trials:
                continue
            
            generations = sorted(coords.keys())
            x_coords = [coords[gen][0] for gen in generations]
            y_coords = [coords[gen][1] for gen in generations]
            color = colors[idx]
            
            for j in range(0, len(generations), plot_interval):
                end = min(j + plot_interval, len(generations)) + 1
                plt.plot(x_coords[j:end], y_coords[j:end], '-o', color=color, alpha=0.6, markersize=5)

            for word in self.exp_data.get_trial_topB_words(file):
                if word in self.exp_data.word_to_umap:
                    x, y = self.exp_data.word_to_umap[word]
                    plt.scatter(x, y, color="gray", s=50, alpha=0.7, marker='s')
                    plt.annotate(word, (x, y), xytext=(3, 3),  
                                textcoords='offset points', fontsize=font_size*0.6, color="gray")
            
            plt.scatter(x_coords[0], y_coords[0], color=color, s=100, zorder=5, label=f'Trial {file} ({len(self.exp_data.get_trial_unique_words(file))})')
            plt.annotate('Initial', (x_coords[0], y_coords[0]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=font_size*0.8, fontweight='bold', color='gray')

            plt.scatter(x_coords[-1], y_coords[-1], color=color, s=100, zorder=5)
            final_word = self.exp_data.trials_gens_count_words[file][generations[-1]].most_common(1)[0][0]
            plt.annotate(final_word, (x_coords[-1], y_coords[-1]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=font_size*0.8, fontweight='bold', color="black")
        
        spec_trials = self._trials_specifier_for_title(only_trials)
        plt.title(f"Semantic trajectory of Experiment '{self.exp_data.label}' for {spec_trials}", fontsize=font_size)
        
        plt.xlabel("UMAP Dimension 1", fontsize=font_size)
        plt.ylabel("UMAP Dimension 2", fontsize=font_size)
        plt.legend(loc='lower left', fontsize=font_size*0.6)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            self._save_visual(fig, output_prefix, output_extension)
        
        return ax
    
    def _save_visual(self, fig: plt.Figure | FuncAnimation, output_prefixe: str, output_extension: str):
        """
        Save the created visual to a file.
        
        Parameters
        -----------
        fig : plt.Figure
            The matplotlib figure to save
        output_prefixe : str
            Prefix for the output file name
        output_extension : Literal['pdf', 'png']
            File format for saving the plot
        """
        filename = path.join(self.exp_data.folder_path, f"{output_prefixe}{"_" if self.exp_data.label is not None else ""}{self.exp_data.label}.{output_extension}")
        super()._save_visual(fig, filename, output_extension)
    
    def create_trajectory_animation(self,
        output_prefix: str ="trajectory_anim",
        only_trials: list[int] =None,
        output_extension: Literal['gif'] ="gif",
        font_size: int = 12,
        fig_size: tuple = (9, 9),
        animation_speed: int = 200,
        save: bool = True,
        subplot: plt.Axes =None
    ) -> list[FuncAnimation]:
        """
        Create one animated trajectory plot for each trial (WARNING: one output file per trial).
        
        Parameters
        ----------
        output_prefix : str, optional
            The file name to save the animation with (without extension and label), by default "trajectory_animation"
        only_trials : list[int], optional
            A list of indices of trials to plot. If None, all trials are plotted, by default None
        output_extension : str, optional
            The file extension for the output animation, either 'gif' or 'mp4', by default 'gif'
        font_size : int, optional
            The font size for the plot, by default 12
        fig_size : tuple, optional
            The size of the figure, by default (9, 9)
        """
        only_trials = None if only_trials is [] else only_trials
        
        returned_animations = []
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.exp_data.all_data.keys())))
        
        for idx, file in enumerate(self.exp_data.all_data.keys()):
            if only_trials is not None and file not in only_trials:
                continue
            
            fig = plt.figure(figsize=fig_size) if subplot is None else subplot.get_figure()
            subplot = fig.add_subplot(111, label=f"trajectory_anim") if subplot is None else subplot
            generations = sorted(list(self.exp_data.all_data[file].keys()))
            coords = self.exp_data.avg_coords[file]
            color = colors[idx]
            total_emergences = set()
            
            def update(frame):
                subplot.clear()
                
                x_coords = [coords[gen][0] for gen in generations[:frame+1]]
                y_coords = [coords[gen][1] for gen in generations[:frame+1]]
                total_emergences.update(self.exp_data.trials_gens_unique_words[file][generations[frame]])
                
                subplot.plot(x_coords[:frame+1], y_coords[:frame+1], '-o', color=color, alpha=0.6, markersize=5)
                
                subplot.scatter(x_coords[0], y_coords[0], color=color, s=100, zorder=5, label=f'Trial {file} ({len(total_emergences)})')
                subplot.annotate('Initial', (x_coords[0], y_coords[0]), xytext=(5, 5),textcoords='offset points', fontsize=font_size*0.8, fontweight='bold', color=color)

                if frame == len(generations) - 1:
                    final_word = self.exp_data.get_trial_topB_words(file)[0]
                    subplot.scatter(x_coords[-1], y_coords[-1], color=color, s=100, zorder=5)
                    subplot.annotate(final_word, (x_coords[-1], y_coords[-1]), xytext=(5, 5), textcoords='offset points', fontsize=font_size*0.8, fontweight='bold', color="black")
                
                subplot.set_title(f"Animated Semantic Trajectory for Trial {file} from Experiment '{self.exp_data.label}'\nCurrent Generation: {generations[frame-1]}", fontsize=font_size)
                subplot.set_xlabel("UMAP Dimension 1", fontsize=font_size)
                subplot.set_ylabel("UMAP Dimension 2", fontsize=font_size)
                subplot.grid(True, alpha=0.3)
                subplot.set_xlim([min(x_coords) - 1, max(x_coords) + 1])
                subplot.set_ylim([min(y_coords) - 1, max(y_coords) + 1])
                subplot.legend(loc='lower left', fontsize=font_size*0.8)
            
            anim = FuncAnimation(fig, update, frames=len(generations), interval=animation_speed, repeat=False)
            returned_animations.append(anim)
            
            if save:
                self._save_visual(anim, f"{output_prefix}_{file}", output_extension)
            
        return returned_animations
    
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
    a = ExpTrajectoryAnalyzer(folder_path='makegraph-en-llama-2', label='xx', lang='en')
    #a.create_trajectory_graph(output_extension="png")
    a.create_trajectory_animation(only_trials=[0], output_extension="gif")
    a.convert_to_video(['.gif'], remove_input=True)