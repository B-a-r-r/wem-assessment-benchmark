from os import listdir, path
from utils import process_file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import Counter
import random as rnd
import wem_assessment_benchmark.Animations as Animations


class Animations:
    """
    Class to create animations from simulation data. This class is not meant to be instantiated,
    use the static methods to create visuals.
    """

    @staticmethod
    def gen_multi_visuals_animation(
        folder_path: str, 
        B: int = 10, 
        default_font_size: int = 12,
        SEED: int = 42
    ) -> None:
        """
        Create multiple animated visuals from simulation data in a specified folder.
        
        Parameters
        ----------
        folder_path : str
            The path to the folder containing the simulation data files.
            
        B : int, optional
            The number of top words to display in the word histogram and frequency plot.
            Default is 10.
            
        default_font_size : int, optional
            The default font size for the plots. Default is 12.
            
        SEED : int, optional
            The seed value for random number generation. Default is 42.
        """
        all_data: dict = {}
        all_word_counts: dict = {}
        top_B_words: dict = {}
        fig: plt.Figure = plt.figure(figsize=(12, 8))
        gs: plt.GridSpec = fig.add_gridspec(2, 2)
        ax1: plt.Axes = fig.add_subplot(gs[0, 0])  
        ax2: plt.Axes = fig.add_subplot(gs[0, 1])  
        ax3: plt.Axes = fig.add_subplot(gs[1, :])  
        
        Animations.set_seed(SEED)  # For reproducibility
        
        for file in listdir(folder_path):
            if file.endswith('.csv'):
                file_path = path.join(folder_path, file)
                all_data[file] = process_file(file_path)

        for file, file_data in all_data.items():
            all_word_counts[file] = {}
            for gen, agents in file_data.items():
                all_word_counts[file][gen] = Counter(agent['word'] for agent in agents)

        for file, word_counts in all_word_counts.items():
            all_words = Counter()
            for gen_counts in word_counts.values():
                all_words.update(gen_counts)
            top_B_words[file] = [word for word, _ in all_words.most_common(B)]

        plt.rcParams.update({'font.size': default_font_size})

        def create_animation(file: str, verbose: bool =False) -> None:
            """
            Create an animation for a specific file.
            
            Parameters
            ----------
            file : str
                The name of the file to create an animation for.
                
            verbose : bool, optional
                If True, print additional information during the animation creation process.
                Default is False.
            """
            if verbose: print(f"Creating animation for {file}...")
            
            generations = sorted(list(all_data[file].keys()))
            
            # Top B words frequency data
            top_words = top_B_words[file]
            word_freq_data = {
                word: [
                    all_word_counts[file][gen][word] for gen in generations
                ] for word in top_words
            }
        
            def update(frame: int, B: int =10) -> None:
                """
                Update the animation for the current frame.

                Parameters
                ----------
                frame : int
                    The current frame index in the animation sequence.
                    
                B : int, optional
                    The number of top words to display in the word histogram and frequency plot.
                """
                ax1.clear()
                ax2.clear()
                ax3.clear()
                
                gen = generations[frame]
                agents = all_data[file][gen]
                
                ax1.set_xlim(0, 15)
                ax1.set_ylim(0, 15)
                ax1.set_aspect('equal', adjustable='box')
                
                for i, agent in enumerate(agents):
                    ax1.add_artist(plt.Circle((agent['x'], agent['y']), 0.3, fill=True, color='lightblue', alpha=0.7))
                    ax1.text(agent['x'], agent['y'], agent['word'], fontsize=default_font_size*0.5, ha='center', va='center')
                    
                
                ax1.set_title(f"Positions and Words at Generation {gen}", fontsize=default_font_size*1)
                ax1.set_xlabel("X", fontsize=default_font_size)
                ax1.set_ylabel("Y", fontsize=default_font_size)
                ax1.tick_params(axis='both', which='major', labelsize=default_font_size*0.8)
                
                word_counts = all_word_counts[file][gen]
                top_words_current = [word for word, _ in word_counts.most_common(B)]
                counts = [word_counts[word] for word in top_words_current]
                
                ax2.barh(top_words_current, counts)
                ax2.set_title(f"Top {B} Words at Generation {gen}", fontsize=default_font_size*1)
                ax2.set_xlabel("Count", fontsize=default_font_size)
                ax2.tick_params(axis='both', which='major', labelsize=default_font_size*0.8)
                ax2.invert_yaxis()
                
                for word in top_words:
                    ax3.plot(generations[:frame+1], word_freq_data[word][:frame+1], label=word)
                
                ax3.set_title(f"Top {B} Words Frequency Over Generations", fontsize=default_font_size*1)
                ax3.set_xlabel("Generation", fontsize=default_font_size)
                ax3.set_ylabel("Frequency", fontsize=default_font_size)
                ax3.tick_params(axis='both', which='major', labelsize=default_font_size*0.8)
                ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=default_font_size*0.8)
                
                fig.tight_layout()
            
            anim = FuncAnimation(fig, update, frames=len(generations), interval=200, repeat=False)
            
            mp4_filename = path.join(folder_path, file.replace('.csv', '.gif'))
            anim.save(mp4_filename, writer='ffmpeg', fps=10)
            print(f"Animation saved as {mp4_filename}")
            
            plt.close(fig)

        for file in all_data.keys():
            create_animation(file, default_font_size=15)  

        print("All animations have been created.")
    
    def set_seed(SEED: int):
        """
        Set the random seed for reproducibility.
        
        Parameters
        ----------
        SEED : int
            The seed value to set for random number generation.
        """
        rnd.seed(SEED)
        np.random.seed(SEED)