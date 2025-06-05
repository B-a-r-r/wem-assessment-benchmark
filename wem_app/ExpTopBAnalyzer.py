from os import listdir, path
from typing import Literal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import Counter
from WemExpData import WemExpData
from WemVisualsBoarding import WemVisualsBoarding
from WemVisualsMaker import WemVisualsMaker
from warnings import warn

#
# author: Reiji SUZUKI et al.
# refactor: Clément BARRIERE
#

class ExpTopBAnalyzer(WemVisualsMaker):
    """
    Class to create animations for the top B words in the experiment data.
    
    Objects Attributes
    -----------------
    borad_figure : plt.Figure
        The figure object for the animation board.
    grid_spec : plt.GridSpec
        The grid specification for the subplots in the animation board.
    subplots : list[plt.Axes]
        The list of subplots for the animation board.
    gen_topB_words : dict[int, dict[int, Counter]]
        A dictionary containing the word counts for each trial and generation.
    trial_topB_words : dict[int, list[str]]
        A dictionary containing the top B words for each trial.
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
        seed : int, default 42
            The seed to use for random operations, by default 42.
        top_B : int, default 10
            The number of top B words to extract for each file, by default 10.
        exp_data : WemExpData
            An existing instance of WemExpData, by default None.
            If provided everything else is ignored.
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
            
        self._plt_font_for_lang()
        self._read_data()
        self._process_data()
        self.exp_data.folder_path = self._create_dedicated_dir(path.join(self.exp_data.folder_path, "topB-analysis"))

    def _plt_font_for_lang(self) -> None:
        """Set the matplotlib font for the specified language."""
        super()._plt_font_for_lang(lang=self.exp_data.lang)
    
    def import_exp_data(self, exp_data: WemExpData):
        """
        Import existing experiment data into the analyzer.
        
        Parameters
        ----------
        exp_data : WemExpData
            An instance of WemExpData containing the experiment data to import.
        """
        assert isinstance(exp_data, WemExpData), "Provided data to import must be an instance of WemExpData."
        self.exp_data = exp_data

    def export_exp_data(self) -> WemExpData:
        """Export the processed experiment data as a WemExpData object."""
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
        Read all experiment data files and store them in all_data.
        The data is should be in CSV format, containing 'gen', 'id', 'x', 'y' and 'word' columns.
        """
        first_element = None
        if not self.exp_data.is_blank():
            first_element = self.exp_data.all_data[0][0][0]
            
        tmp = type(first_element)
        if not self.exp_data.is_blank() and not tmp == dict:
            warn("The units in the provided experiment data must be dict " \
                + f"representing agents to create trajectory visuals, found {tmp} instead.\n" \
                + "NB: string type is used to create trajectory or emergence score visuals for exemple.\n" \
                + "---> Rereading experiment data...\n"
            )
            
        if self.exp_data.is_blank() or not tmp == dict:
            csv_idx = 0
            for file in listdir(path.join(path.dirname(__file__), self.exp_data.folder_path)):
                if file.endswith('csv'):
                    data = pd.read_csv(path.join(path.dirname(__file__), self.exp_data.folder_path, file), encoding='utf-8')
                    data = data.groupby('gen')[['id', 'x', 'y', 'word']].apply(lambda x: x.to_dict('records')).to_dict()
                    self.exp_data.all_data[csv_idx] = data
                    
                    csv_idx += 1
            
            if csv_idx == 0:
                warn("No CSV files found in the provided folder path.")
         
    def _process_data(self):
        """
        Process the retrieved data to sort and compute other specific varaibles.
        """
        if self.exp_data.is_empty():
            for file, data in self.exp_data.all_data.items():
                
                self.exp_data.trials_gens_unique_words[file] = {}
                for gen, agents in data.items():
                    self.exp_data.trials_gens_unique_words[file][gen] = set()
                    words = set(agent['word'] for agent in agents)
                    self.exp_data.trials_gens_unique_words[file][gen].update(words)
                    
                self.exp_data.trials_gens_count_words[file] = {}
                for gen, agents in data.items():
                    self.exp_data.trials_gens_count_words[file][gen] = Counter(agent['word'] for agent in agents)
    
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
    
    def create_agents_pos_anim(self,
        fig_size: tuple[int, int] =(12, 8),
        font_size: int =12,
        save: bool =True,
        animation_speed: int =200,
        only_trials: list[str] =None,
        output_extension: Literal['gif', 'mp4'] ='gif',
        output_prefix: str ='agents_pos_anim',
        subplot: plt.Axes =None
    ) -> list[FuncAnimation]:
        """"""
        only_trials = None if only_trials is [] else only_trials
        
        returned_animations = []
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.exp_data.all_data.values())))
        
        for idx, file in enumerate(self.exp_data.all_data.keys()):
            if only_trials is not None and file not in only_trials:
                continue
            
            fig = plt.figure(figsize=fig_size) if subplot is None else subplot.figure
            subplot = fig.add_subplot(111, label='agents_pos_anim') if subplot is None else subplot
            generations = sorted(list(self.exp_data.all_data[file].keys()))
            
            def update(frame):
                subplot.clear()
                
                agents = self.exp_data.all_data[file][generations[frame]]
                
                subplot.set_xlim(0, 15)
                subplot.set_ylim(0, 15)
                subplot.set_aspect('equal', adjustable='box')
                
                for i, agent in enumerate(agents):
                    subplot.add_artist(plt.Circle((agent['x'], agent['y']), 0.3, fill=True, color=colors[i], alpha=0.7))
                    subplot.text(agent['x'], agent['y'], agent['word'], fontsize=font_size*0.5, ha='center', va='center')
                
                subplot.grid(True, color='gray', linestyle='--', linewidth=0.2)       
                subplot.set_title(f"Positions and Words at Generation {generations[frame]}", fontsize=font_size*1)
                subplot.set_xlabel("x", fontsize=font_size)
                subplot.set_ylabel("y", fontsize=font_size)
                subplot.tick_params(axis='both', which='major', labelsize=font_size*0.8)
            
            anim = FuncAnimation(fig, update, frames=len(generations), interval=animation_speed, repeat=False)
            returned_animations.append(anim)
                
            if save:
                self._save_visual(anim, f"{output_prefix}_{file}", output_extension)
                
        return returned_animations
            
    def create_top_hist_anim(self, 
        fig_size: tuple[int, int] =(12, 8),
        font_size: int =12,
        save: bool =True,
        animation_speed: int =200,
        only_trials: list[str] =None,
        output_extension: Literal['gif', 'mp4'] ='gif',
        output_prefix: str ='top_hist_anim',
        subplot: plt.Axes =None
    ) -> list[FuncAnimation]:
        """"""
        only_trials = None if only_trials is [] else only_trials
        
        returned_animations = []
        
        for idx, file in enumerate(self.exp_data.all_data.keys()):
            if only_trials is not None and file not in only_trials:
                continue
            
            fig = plt.figure(figsize=fig_size) if subplot is None else subplot.figure
            subplot = fig.add_subplot(111, label='top_hist_anim') if subplot is None else subplot
            generations = sorted(list(self.exp_data.all_data[file].keys()))
            
            def update(frame):
                subplot.clear()
                
                words_count = self.exp_data.trials_gens_count_words[file][generations[frame]].most_common(self.exp_data.top_B)
        
                current_top_words = []
                counts = []
                for word, count in words_count:
                    current_top_words.append(word)
                    counts.append(count)
                
                subplot.barh(current_top_words, counts)
                subplot.set_title(f"Top {self.exp_data.top_B} Words at Generation {generations[frame]}", fontsize=font_size*1)
                subplot.set_xlabel("Count", fontsize=font_size)
                subplot.tick_params(axis='both', which='major', labelsize=font_size*0.8)
                subplot.invert_yaxis()
            
            anim = FuncAnimation(fig, update, frames=len(generations), interval=animation_speed, repeat=False)
            returned_animations.append(anim)
                
            if save:
                self._save_visual(anim, f"{output_prefix}_{file}", output_extension)
                
        return returned_animations
        
    def create_top_freq_anim(self, 
        fig_size: tuple[int, int] =(12, 8),
        font_size: int =12,
        save: bool =True,
        animation_speed: int =200,
        only_trials: list[str] =None,
        output_extension: Literal['gif', 'mp4'] ='gif',
        output_prefix: str ='top_freq_anim',
        subplot: plt.Axes =None
    ) -> list[FuncAnimation]:
        """"""
        only_trials = None if only_trials is [] else only_trials
        
        returned_animations = []
        
        for idx, file in enumerate(self.exp_data.all_data.keys()):
            if only_trials is not None and file not in only_trials:
                continue
            
            fig = plt.figure(figsize=fig_size) if subplot is None else subplot.figure
            subplot = fig.add_subplot(111, label='top_freq_anim') if subplot is None else subplot
            generations = sorted(list(self.exp_data.all_data[file].keys()))
            trial_top_words = self.exp_data.get_trial_topB_words(file)
            word_freq_data = {}
            
            for word in trial_top_words:
                word_freq_data[word] = []
                for g in generations:
                    word_freq_data[word].append(self.exp_data.trials_gens_count_words[file][g].get(word, 0))
            
            def update(frame):
                subplot.clear()
                
                for word in trial_top_words:
                    subplot.plot(generations[:frame+1], word_freq_data[word][:frame+1], label=word)
                
                subplot.set_title(f"Trial {file} Top {self.exp_data.top_B} Words Frequency Over Generations", fontsize=font_size*1)
                subplot.set_xlabel("Generation", fontsize=font_size)
                subplot.set_ylabel("Frequency", fontsize=font_size)
                subplot.tick_params(axis='both', which='major', labelsize=font_size*0.8)
                subplot.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=font_size*0.8)
            
            anim = FuncAnimation(fig, update, frames=len(generations), interval=animation_speed, repeat=False)
            returned_animations.append(anim)
                
            if save:
                self._save_visual(anim, f"{output_prefix}_{file}", output_extension)
                
        return returned_animations
    
    def create_additional_legend(self,
        fig_size: tuple[int, int] =(12, 8),
        font_size: int =12,
        save: bool =True,
        animation_speed: int =200,
        only_trials: list[str] =None,
        output_extension: Literal['gif', 'mp4'] ='gif',
        output_prefix: str ='topB_additional_legend',
        subplot: plt.Axes =None
    ) -> plt.Axes:
        """"""
        only_trials = None if only_trials is [] else only_trials
        
        fig = plt.figure(figsize=fig_size) if subplot is None else subplot.figure
        subplot = fig.add_subplot(111, label='topB_additional_legend') if subplot is None else subplot
        pass
        # subplot.clear()
        
        # agents = self.all_data[file][gen]
        # current_num_species = len(set(agent['word'] for agent in agents))
        
        # subplot.grid(False)
        # subplot.set_axis_off()
        # subplot.set_aspect('auto', adjustable='datalim')
        # subplot.text(
        #     0.5, 
        #     0.5, 
        #     f"Additional Information:\nTrial: {file}\nGeneration: {gen}\nTotal number of active species: {current_num_species}", 
        #     transform=subplot.transAxes, 
        #     fontsize=font_size,
        #     va='center',
        #     bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.5')
        # )
    
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
    a = ExpTopBAnalyzer(folder_path='makegraph-en-llama-2', label='', lang='en')
    a.create_agents_pos_anim(only_trials=[0])
    a.create_top_hist_anim(only_trials=[0])
    a.create_top_freq_anim(only_trials=[0])
    a.convert_to_video(['.gif'], remove_input=True)
    