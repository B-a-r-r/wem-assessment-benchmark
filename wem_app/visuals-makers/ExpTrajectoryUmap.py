from os import listdir, path, makedirs
from typing import Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import Counter
from scipy.stats import gaussian_kde
from short.WemVisualsMaker import ExpVisualsMaker
from WemUmapData import WemUmapData
from warnings import warn

#
# author: Reiji SUZUKI et al.
# refactor: Clément BARRIERE
#

class ExpTrajectoryAnalysis(ExpVisualsMaker):
    """
    Class to create trajectory graphs and metrics from the data of an experiment.
    
    Object Attributes
    ----------------
    exp_data : WemExpData
        An object containing the experiment loaded data and umap representations.
    """
    
    def __init__(self, **kwargs):
        """
        Kwargs Parameters
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
        
        if kwargs.get('exp_data', None) is None:
            folder_path = path.abspath(kwargs.get('folder_path'))
            if not path.exists(path.join(folder_path, "trajectory-anlysis")):
                makedirs(folder_path, "trajectory-anlysis")
                folder_path = path.abspath(path.join(folder_path, "trajectory-anlysis"))
            exp_data = WemUmapData(
                folder_path, 
                label=kwargs.get('label', ''), 
                lang=kwargs.get('lang', 'en'), 
                seed=kwargs.get('seed', 42), 
                top_B=kwargs.get('top_B', 10),
                model=kwargs.get('sentence_transformer_model', 'all-MiniLM-L6-v2')
            )   
        
        self.exp_data: WemUmapData = exp_data
        ExpTrajectoryAnalysis._plt_font_for_lang(lang=self.exp_data.lang) 
        
        self._read_data()
        self._process_data()
    
    def _read_data(self) -> None:
        """
        Read all experiment data files and store them in all_data.
        The data is should be in CSV format, containing 'gen' and 'word' columns.
        If an existing experiment data object has been provided, ensure its format 
        match this class requirements.
        """
        tmp = type(self.exp_data.all_data[0][0][0])
        if not self.exp_data.is_empty() and not tmp == str:
            warn("The units in the provided experiment data must be string " \
                + f"representing words to create trajectory visuals, found {tmp} instead.\n" \
                + "NB: dict type is used to create topB animation visuals.\n" \
                + "---> Rereading experiment data...\n"
            )
            
        if self.exp_data.is_blank() or not tmp == str:
            csv_idx = 0
            for file in listdir(path.abspath(self.exp_data.folder_path)):
                if file.endswith('csv'):
                    file_path = path.abspath(self.exp_data.folder_path, file)
                    self.exp_data.all_data[csv_idx] = (\
                        pd.read_csv(file_path, usecols=['gen', 'word'], encoding='utf-8')\
                        .groupby('gen')['word'].apply(list).to_dict()
                    )
                    csv_idx += 1
                    
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
            
                self.exp_data.trial_colors[file] = plt.cm.rainbow(np.linspace(0, 1, len(self.exp_data.get_trial_unique_words(file))))
            
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
        output_extension: str ="pdf",
    ):
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
        assert output_extension in ['pdf', 'png'], "output_extension must be either 'pdf' or 'png'."
        only_trials = None if only_trials == [] else only_trials
            
        plt.figure(figsize=fig_size)
        
        for idx, (file, coords) in enumerate(self.exp_data.avg_coords.items()): 
            if only_trials is not None and file not in only_trials:
                continue
            
            generations = sorted(coords.keys())
            x_coords = [coords[gen][0] for gen in generations]
            y_coords = [coords[gen][1] for gen in generations]
            
            color = self.exp_data.trial_colors[0][idx]
            
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
                        textcoords='offset points', fontsize=font_size*0.8, fontweight='bold', color=color)

            plt.scatter(x_coords[-1], y_coords[-1], color=color, s=100, zorder=5)
            final_word = self.exp_data.trials_gens_count_words[file][generations[-1]].most_common(1)[0][0]
            plt.annotate(final_word, (x_coords[-1], y_coords[-1]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=font_size*0.8, fontweight='bold', color="black")
        
        spec_trials = "each trials"
        if only_trials is not None:
            spec_trials = "trials "
            for t in only_trials:
                spec_trials += f"{t}{", " if t != only_trials[-1] else ""}" 
        plt.title(f"Semantic trajectory of Experiment '{self.exp_data.label}' for {spec_trials}", fontsize=font_size)
        
        plt.xlabel("UMAP Dimension 1", fontsize=font_size)
        plt.ylabel("UMAP Dimension 2", fontsize=font_size)
        plt.legend(loc='lower left', fontsize=font_size*0.6)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = path.join(self.exp_data.folder_path, f"{output_prefix}{"_" if self.exp_data.label is not None else ""}{self.exp_data.label}.{output_extension}")
        plt.savefig(filename, format=output_extension, dpi=300, bbox_inches='tight')
        print(f"Trajectory graph saved as {filename}")
    
    def create_trajectory_animation(self,
        output_prefix: str = "trajectory_anim",
        only_trials: list[int] = None,
        output_extension: str = "gif",
        font_size: int = 12,
        fig_size: tuple = (9, 9),
        animation_speed: int = 200
    ):
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
        assert output_extension in ['gif', 'mp4'], "output_extension must be either 'gif' or 'mp4'."
        only_trials = None if only_trials is [] else only_trials
        
        fig, subplot = plt.subplots(figsize=fig_size)
        
        for file in self.exp_data.all_data.keys():
            if only_trials is not None and file not in only_trials:
                continue
        
            generations = sorted(list(self.exp_data.all_data[file].keys()))
            coords = self.exp_data.avg_coords[file]
            color = self.exp_data.trial_colors[file][0]
            
            def update(frame):
                subplot.clear()
                
                x_coords = [coords[gen][0] for gen in generations[:frame+1]]
                y_coords = [coords[gen][1] for gen in generations[:frame+1]]
                
                subplot.plot(x_coords[:frame+1], y_coords[:frame+1], '-o', color=color, alpha=0.6, markersize=5, label=f'{file[:-4]} ({len(self.exp_data.trials_gens_unique_words[file][generations[frame-1]])})')
                
                subplot.scatter(x_coords[0], y_coords[0], color=color, s=100, zorder=5)
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
                subplot.legend(loc='lower left', fontsize=font_size*0.6)
            
            anim = FuncAnimation(fig, update, frames=len(generations), interval=animation_speed, repeat=False)
            
            filename = path.join(self.exp_data.folder_path, f"{output_prefix}{"_" if self.exp_data.label is not None else ""}{self.exp_data.label}_{file}.{output_extension}")
            anim.save(filename, writer='ffmpeg', fps=10)
            print(f"Trajectory animation saved to {filename}")
        
    def create_contour_plot(self, 
        output_prefixe: str = "contour_plot", 
        fig_size: tuple = (9, 9),
        output_extension: str = 'pdf',
        only_trials: list[int] = None
    ):
        """ 
        Create a contour plot of the UMAP embeddings of the words in the experiment data.

        Parameters
        ----------
        output_prefix : str, optional
            The path to save the contour plot, by default "contour_plot"
        fig_size : tuple, optional
            The size of the figure, by default (9, 9)
        output_extension : str, optiona
            The file extension for the output plot, either 'pdf' or 'png', by default 'pdf'
        only_trials : list[int], optional
            A list of indices of trials to plot. If None, all trials are plotted, by default None            
        """
        assert output_extension in ['pdf', 'png'], "output_extension must be either 'pdf' or 'png'."
        only_trials = None if only_trials is [] else only_trials
        
        for file in self.exp_data.all_data.keys():
            if only_trials is not None and file not in only_trials:
                continue
            
            plt.figure(figsize=fig_size)
            umap_embeddings = np.array(list(self.exp_data.word_to_umap.values()))
            xy = np.vstack([umap_embeddings[:, 0], umap_embeddings[:, 1]])
            kde = gaussian_kde(xy)

            x, y = np.mgrid[umap_embeddings[:, 0].min():umap_embeddings[:, 0].max():100j,
                                umap_embeddings[:, 1].min():umap_embeddings[:, 1].max():100j]
            positions = np.vstack([x.ravel(), y.ravel()])
            density = kde(positions)

            plt.contourf(x, y, density.reshape(x.shape), levels=20, cmap='viridis', alpha=0.8)

        spec_trials = "all trials"
        if only_trials is not None:
            spec_trials = "trials "
            for t in only_trials:
                spec_trials += f"{t}{", " if t != only_trials[-1] else ""}" 
        plt.title(f"Density of Points in the Semantic Space of Experiment '{self.exp_data.label}' with {spec_trials}", fontsize=12)
        
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.colorbar(label='Density')
        plt.tight_layout()

        filename = path.join(self.exp_data.folder_path, f"{output_prefixe}{"_" if self.exp_data.label is not None else ""}{self.exp_data.label}.{output_extension}")
        plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
        print(f"Contour plot saved as {filename}")


if __name__ == "__main__":
    a = ExpTrajectoryAnalysis(folder_path='makegraph-en-llama-2', label='', lang='en')
    a.create_trajectory_graph(output_extension="png")
    # a.create_contour_plot(output_extension="png")
    #a.create_trajectory_animation(output_extension="gif")