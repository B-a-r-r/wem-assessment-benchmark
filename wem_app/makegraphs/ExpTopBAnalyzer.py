from os import listdir, path
from typing import Literal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import Counter
from .WemExpData import WemExpData
from .WemVisualsBoarding import WemVisualsBoarding
from .WemVisualsMaker import WemVisualsMaker
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
    exp_data : WemExpData
        An instance of WemExpData containing the experiment data.
    """

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        folder_path : str, optional
            The path to the experiment folder containing the data files.
        exp_data : WemExpData, optional
            An instance of WemExpData containing the experiment data to analyze.
            If provided, it will override all other parameter.
        label : str, optional
            A label for the experiment, used in the output file names.
        seed : int, optional
            A seed for random number generation, default is 42.
        lang : str, optional
            The language for the analysis, default is 'en'.
        top_B : int, optional
            The number of top words to consider in the analysis, default is 10.
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
        assert isinstance(exp_data, WemExpData), "Provided data to import must be an instance of WemExpData."
        self.exp_data = exp_data

    def export_exp_data(self) -> WemExpData:
        """
        Export the processed experiment data as a WemExpData object.
        
        Returns
        -------
        WemExpData
            The processed experiment data.
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
        Read the experiment data from the specified folder path.
        If the data is already loaded, it checks if the data is in the correct format.
        Here the correct format is a dictionary mapping trials, generations and agents.
        If the data is not in the correct format, it attempts to reread files from the folder.
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
            for file in listdir(path.abspath(self.exp_data.folder_path)):
                if file.endswith('csv'):
                    data = pd.read_csv(path.join(path.abspath(self.exp_data.folder_path), file), encoding='utf-8')
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
        subplot: plt.Axes =None,
        grid_shape: tuple[int, int] =(15, 15)
    ) -> list[FuncAnimation]:
        """
        Create an animation showing the positions of agents and their associated words over generations.
        
        Parameters
        ----------
        fig_size : tuple[int, int], optional
            The size of the figure for the animation, default is (12, 8).
        font_size : int, optional
            The font size for the text in the animation, default is 12.
        save : bool, optional
            If True, the animation will be saved to a file, default is True.
        animation_speed : int, optional
            The speed of the animation in milliseconds, default is 200.
        only_trials : list[str], optional
            A list of trial identifiers to filter the animations, default is None (no filtering).
        output_extension : Literal['gif', 'mp4'], optional
            The file format for saving the animation, default is 'gif'.
        output_prefix : str, optional
            The prefix for the output file name, default is 'agents_pos_anim'.
        subplot : plt.Axes, optional
            A matplotlib Axes object to use for the animation, default is None (a new Axes will be created).
        grid_shape : tuple[int, int], optional
            The shape of the grid for the animation, default is (15, 15).
        
        Returns
        -------
        list[FuncAnimation]
            A list of FuncAnimation objects for each trial in the experiment data.
        """
        only_trials = None if only_trials is [] else only_trials
        
        returned_animations = []
        
        for idx, file in enumerate(self.exp_data.all_data.keys()):
            if only_trials is not None and file not in only_trials:
                continue
            
            fig = plt.figure(figsize=fig_size) if subplot is None else subplot.figure
            subplot = fig.add_subplot(111, label='agents_pos_anim') if subplot is None else subplot
            generations = sorted(list(self.exp_data.all_data[file].keys()))
            
            def update(frame):
                subplot.clear()
                
                colors = plt.cm.rainbow(np.linspace(0, 1, len(self.exp_data.trials_gens_unique_words[file][generations[frame]])))
                color_map = {word: color for word, color in zip(self.exp_data.trials_gens_unique_words[file][generations[frame]], colors)}
                agents = self.exp_data.all_data[file][generations[frame]]
                
                subplot.set_xlim(0, grid_shape[0])
                subplot.set_ylim(0, grid_shape[1])
                subplot.set_aspect('equal', adjustable='box')
                
                for i, agent in enumerate(agents):
                    subplot.add_artist(plt.Circle((agent['x'], agent['y']), 0.3, fill=True, color=color_map[agent['word']], alpha=0.7))
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
        """
        Create an animation showing the top B words and their counts over generations.
        
        Parameters
        ----------
        fig_size : tuple[int, int], optional
            The size of the figure for the animation, default is (12, 8).
        font_size : int, optional
            The font size for the text in the animation, default is 12.
        save : bool, optional
            If True, the animation will be saved to a file, default is True.
        animation_speed : int, optional
            The speed of the animation in milliseconds, default is 200.
        only_trials : list[str], optional
            A list of trial identifiers to filter the animations, default is None (no filtering).
        output_extension : Literal['gif', 'mp4'], optional
            The file format for saving the animation, default is 'gif'.
        output_prefix : str, optional
            The prefix for the output file name, default is 'top_hist_anim'.
        subplot : plt.Axes, optional
            A matplotlib Axes object to use for the animation, default is None (a new Axes will be created).
        
        Returns
        -------
        list[FuncAnimation]
            A list of FuncAnimation objects for each trial in the experiment data.
        """
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
        """
        Create an animation showing the frequency of the top B words over generations.
        
        Parameters
        ----------
        fig_size : tuple[int, int], optional
            The size of the figure for the animation, default is (12, 8).
        font_size : int, optional
            The font size for the text in the animation, default is 12.
        save : bool, optional
            If True, the animation will be saved to a file, default is True.
        animation_speed : int, optional
            The speed of the animation in milliseconds, default is 200.
        only_trials : list[str], optional
            A list of trial identifiers to filter the animations, default is None (no filtering).
        output_extension : Literal['gif', 'mp4'], optional
            The file format for saving the animation, default is 'gif'.
        output_prefix : str, optional
            The prefix for the output file name, default is 'top_freq_anim'.
        subplot : plt.Axes, optional
            A matplotlib Axes object to use for the animation, default is None (a new Axes will be created).
        
        Returns
        -------
        list[FuncAnimation]
            A list of FuncAnimation objects for each trial in the experiment data.
        """
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
        """TODO: Implement the additional legend for top B analysis."""
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
    