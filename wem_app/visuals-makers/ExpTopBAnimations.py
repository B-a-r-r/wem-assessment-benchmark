from os import listdir, path, stat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import Counter
from WemExpData import WemExpData
from WemVisualsBoarding import WemVisualsBoarding
from WemVisualsMaker import WemVisualsMaker
from warnings import warn


class ExpTopBAnimations(WemVisualsMaker, WemVisualsBoarding):
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
    AVAILABLE_VISUALS = ['agents_pos_plot', 'top_words_hist', 'top_words_freq_graph', 'additional_legend']
    
    def __init__(self, 
        folder_path: str, 
        label: str, 
        lang: str, 
        seed: int =42, 
        top_B: int =10, 
        board_size: tuple =(16, 8),
        only_visuals: list[str] = None
    ):
        """
        Initialize the ExpTopBAnimations class, which is used to create animations for the top B words in the experiment data.
        
        Parameters
        ----------
        folder_path : str
            The path to the folder containing the experiment data files.
        label : str 
            The label for the experiment. Added to the output file names.
        lang : str
            The language of the experiment data.
        seed : int, optional
            The random seed for reproducibility, by default 42.
        top_B : int, optional
            The number of top words to consider for the animations, by default 10.
        board_size : tuple, optional
            The size of the board figure for the animations, by default (16, 8).
        only_visuals : list[str], optional
            A list of specific visuals to include in the animations. If None, all available visuals will be used.
        """
        if only_visuals is not None:
            assert len(only_visuals) > 0, "At least one visual must be specified."
        only_visuals = only_visuals if only_visuals is not None else ExpTopBAnimations.AVAILABLE_VISUALS
        
        super(WemVisualsMaker, self).__init__(self)
        super(WemVisualsBoarding, self).__init__(self)
        
        self.exp_data = WemExpData(folder_path=folder_path, label=label, seed=seed, lang=lang, top_B=top_B)
        
        self.borad_figure: plt.Figure = plt.figure(figsize=board_size)
        
        ref = (len(only_visuals) if only_visuals is not None else len(ExpTopBAnimations.AVAILABLE_VISUALS))//2
        self.grid_spec: plt.GridSpec = self.borad_figure.add_gridspec(ref, ref)
        
        self.subplots: list[plt.Axes] = []
        for visual in only_visuals:
            if visual in ExpTopBAnimations.AVAILABLE_VISUALS:
                for idx, available in enumerate(ExpTopBAnimations.AVAILABLE_VISUALS):
                    if visual == available:
                        self.subplots.append(self.borad_figure.add_subplot(self.grid_spec[idx // ref, idx % ref], label=visual))
            else:
                plt.close(self.borad_figure)
                raise ValueError(f"Visual {visual} is not available. Available options are {ExpTopBAnimations.AVAILABLE_VISUALS}.")
            
        if lang == 'ch':
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        
        self._read_data()
        self._process_data()

    
    def _read_data(self):
        """
        Read all experiment data files and store them in all_data.
        The data is should be in CSV format, containing 'gen', 'id', 'x', 'y' and 'word' columns.
        """
        tmp = type(self.exp_data.all_data[0][0][0])
        if not self.exp_data.is_empty() and not tmp == dict:
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

                all_words_last_gen = Counter([agent['word'] for agent in data[len(data)-1]])
                self.exp_data.get_trial_topB_words(file) = [word for word, _ in all_words_last_gen.most_common(self.exp_data.top_B)]
                
                self.exp_data.trial_colors[file] = plt.cm.rainbow(np.linspace(0, 1, len(self.exp_data.get_trial_unique_words(file))))
    
    def create_animatons_boards(self, font_size: int =12, output_extension: str ='gif', only_trials: list[str] =None):
        """
        For each trial, create an animation board with the specified visuals and save it as a file.
        
        Parameters
        ----------
        font_size : int, optional
            The font size for the plot labels and titles, by default 12.
        output_extension : str, optional
            The file extension for the output file, by default 'gif' (can be 'gif' or 'mp4').
        only_trials : list[str], optional
            A list of specific trials to create animations for. If None, all trials will be used.
        """
        assert output_extension in ['gif', 'mp4'], "Output extension must be either 'gif' or 'mp4'."
        only_trials = None if only_trials is [] else only_trials
        
        for file in self.all_data.keys():
            if only_trials is not None and file not in only_trials:
                continue
            
            generations = sorted(list(self.all_data[file].keys()))

            def update(frame):
                gen = generations[frame]
                
                for subplot in self.subplots:
                    getattr(self, f"create_{subplot.get_label()}")(font_size=font_size, gen=gen, file=file)
                
                self.borad_figure.tight_layout()
            
            anim = FuncAnimation(self.borad_figure, update, frames=len(generations), interval=200, repeat=False)
            
            filename = path.join(path.dirname(__file__), self.folder_path, f"anims_topB_trial{file}.{output_extension}")
            anim.save(filename, writer='ffmpeg', fps=10)
            print(f"Animations board saved as {filename}")

        plt.close(self.borad_figure)
        
    def create_agents_pos_plot(self, file: str, font_size: int =12, gen: int =0, subplot: plt.Axes =None, save: bool =False):
        """
        Create a plot of the simulated space with each agents displayed at their position for a given generation.
        
        Parameters
        ----------
        file : str
            The name of the file containing the data.
        font_size : int, optional
            The font size for the plot labels and titles, by default 12.
        gen : int, optional
            The generation number to plot, by default 0.
        subplot : plt.Axes, optional
            The subplot to plot on if external, by default None (using this object's subplot).
        save : bool, optional
            Whether to save the subplot as a file, by default False.
        """
        if subplot is not None:
            subplot = subplot
        else:
            try:
                subplot = [sub for sub in self.subplots if sub.get_label() == 'agents_pos_plot'][0]
            except IndexError:
                raise ValueError("No subplot with label 'agents_pos_plot' found. You are trying to create an unexpected visual of the board. If you want to use this method separately, please specify the external subplot in argument.")
        
        subplot.clear()
        
        subplot.set_xlim(0, 15)
        subplot.set_ylim(0, 15)
        subplot.set_aspect('equal', adjustable='box')
        
        agents = self.all_data[file][gen]
        
        for i, agent in enumerate(agents):
            subplot.add_artist(plt.Circle((agent['x'], agent['y']), 0.3, fill=True, color=self.trial_colors[file][i], alpha=0.7))
            subplot.text(agent['x'], agent['y'], agent['word'], fontsize=font_size*0.5, ha='center', va='center')
                    
        
        subplot.grid(True, color='gray', linestyle='--', linewidth=0.2)       
        subplot.set_title(f"Positions and Words at Generation {gen}", fontsize=font_size*1)
        subplot.set_xlabel("X", fontsize=font_size)
        subplot.set_ylabel("Y", fontsize=font_size)
        subplot.tick_params(axis='both', which='major', labelsize=font_size*0.8)
        
        if save:
            self.save_subplot(subplot, output_extension='png', gen=gen, file=file)
            
    def create_top_words_hist(self, file: str, font_size: int =12, gen: int =0, subplot: plt.Axes =None, save: bool =False):
        """ 
        Create a horizontal bar plot of the top B words at a given generation.
        
        Parameters
        ----------
        file : str
            The name of the file containing the data.
        font_size : int, optional
            The font size for the plot labels and titles, by default 12.
        gen : int, optional
            The generation number to plot, by default 0.
        subplot : plt.Axes, optional
            The subplot to plot on if external, by default None.
        save : bool, optional
            Whether to save the subplot as a file, by default False.
        """
        if subplot is not None:
            subplot = subplot
        else:
            try:
                subplot = [sub for sub in self.subplots if sub.get_label() == 'top_words_hist'][0]
            except IndexError:
                raise ValueError("No subplot with label 'top_words_hist' found. You are trying to create an unexpected visual of the board. If you want to use this method separately, please specify the external subplot in argument.")
            
        subplot.clear()
        
        words_count = self.gen_topB_words[file][gen].most_common(self.top_B)
        
        current_top_words = []
        counts = []
        for word, count in words_count:
            current_top_words.append(word)
            counts.append(count)
        
        subplot.barh(current_top_words, counts)
        subplot.set_title(f"Top {self.top_B} Words at Generation {gen}", fontsize=font_size*1)
        subplot.set_xlabel("Count", fontsize=font_size)
        subplot.tick_params(axis='both', which='major', labelsize=font_size*0.8)
        subplot.invert_yaxis()
        
        if save:
            self.save_subplot(subplot, output_extension='png', gen=gen, file=file)
        
    def create_top_words_freq_graph(self, file: str, gen: int =0, font_size: int =12, subplot: plt.Axes =None, save: bool =False):
        """
        Create a line plot of the frequency of the top B words over generations.
        
        Parameters
        ----------
        file : str
            The name of the file containing the data.
        gen : int, optional
            The generation number to plot, by default 0.
        font_size : int, optional
            The font size for the plot labels and titles, by default 12.
        subplot : plt.Axes, optional
            The subplot to plot on if external, by default None.
        save : bool, optional
            Whether to save the subplot as a file, by default False.
        """
        if subplot is not None:
            subplot = subplot
        else:
            try:
                subplot = [sub for sub in self.subplots if sub.get_label() == 'top_words_freq_graph'][0]
            except IndexError:
                raise ValueError("No subplot with label 'top_words_freq_graph' found. You are trying to create an unexpected visual of the board. If you want to use this method separately, please specify the external subplot in argument.")
        
        subplot.clear()
        
        generations = sorted(list(self.all_data[file].keys()))
        trial_top_words = self.trial_topB_words[file]
        word_freq_data = {}
        
        for word in trial_top_words:
            word_freq_data[word] = []
            
            for g in generations:
                word_freq_data[word].append(self.gen_topB_words[file][g].get(word, 0))
        
        for word in trial_top_words:
            subplot.plot(generations[:gen+1], word_freq_data[word][:gen+1], label=word)
        
        subplot.set_title(f"Trial {file} Top {self.top_B} Words Frequency Over Generations", fontsize=font_size*1)
        subplot.set_xlabel("Generation", fontsize=font_size)
        subplot.set_ylabel("Frequency", fontsize=font_size)
        subplot.tick_params(axis='both', which='major', labelsize=font_size*0.8)
        subplot.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=font_size*0.8)
        
        if save:
            self.save_subplot(subplot, output_extension='png', gen=gen, file=file)
        
    def create_additional_legend(self, file: str, gen: int =0, font_size: int =12, subplot: plt.Axes =None):
        """ 
        Create a legend for the additional information.
        
        Parameters
        ----------
        file : str
            The name of the file containing the data.
        gen : int, optional
            The generation number to plot, by default 0.
        font_size : int, optional
            The font size for the plot labels and titles, by default 12.
        subplot : plt.Axes, optional
            The subplot to plot on if external, by default None.
        """
        if subplot is not None:
            subplot = subplot
        else:
            try:
                subplot = [sub for sub in self.subplots if sub.get_label() == 'additional_legend'][0]
            except IndexError:
                raise ValueError("No subplot with label 'additional_legend' found. You are trying to create an unexpected visual of the board. If you want to use this method separately, please specify the external subplot in argument.")
        
        subplot.clear()
        
        agents = self.all_data[file][gen]
        current_num_species = len(set(agent['word'] for agent in agents))
        
        subplot.grid(False)
        subplot.set_axis_off()
        subplot.set_aspect('auto', adjustable='datalim')
        subplot.text(
            0.5, 
            0.5, 
            f"Additional Information:\nTrial: {file}\nGeneration: {gen}\nTotal number of active species: {current_num_species}", 
            transform=subplot.transAxes, 
            fontsize=font_size,
            va='center',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.5')
        )
    
    def save_subplot(self, subplot: plt.Axes, file: str, output_extension: str ='png', gen: int =0):
        """
        Save the current subplot as a file.
        
        Parameters
        ----------
        subplot : plt.Axes
            The subplot to save.
        file : str
            The name of the file containing the data.
        output_extension : str, optional
            The file extension for the output file, by default 'png'.
        gen : int, optional
            The generation number to plot, by default 0.
        """
        assert output_extension in ['png', 'jpg', 'gif'], "Output extension must be either 'png', 'jpg' or 'gif'."
        
        filename = path.join(path.dirname(__file__), self.folder_path, f"{subplot.get_label()}_gen{gen}_trial{file}.{output_extension}")
        subplot.figure.savefig(filename, bbox_inches='tight')
        print(f"Subplot saved as {filename}")

    def __repr__(self):
        return f"""
        ExpAnimationsBoard(folder_path={self.folder_path}, label={self.label}, lang={self.lang}, seed={self.seed})
        """
    
if __name__ == "__main__":
    a = ExpTopBAnimations(folder_path='makegraph-en-llama-2', label='', lang='en')
    
    # tmp_fig = plt.figure(figsize=(12, 8))
    # tmp_subplot = tmp_fig.add_subplot(111)
    # a.create_agents_pos_plot(file='result_0.csv', gen=0, save=True, subplot=tmp_subplot)
    
    #a.create_animatons_boards(output_extension="gif")
    
    # fig, ax = plt.subplots(figsize=(16, 8))
    # for gen in range(0, 100, 10):
    #     a.create_top_words_freq_graph(file=0, gen=gen, font_size=12, subplot=ax, save=True)