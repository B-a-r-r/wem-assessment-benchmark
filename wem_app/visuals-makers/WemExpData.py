
from dataclasses import dataclass
import numpy as np
from collections import Counter
import random as rnd
import torch
from os import environ, path

#
# author: Clément BARRIERE
#

@dataclass
class WemExpData:
    """
    A class to store data shared by every experiment.
    
    Attributes
    ----------
    folder_path : str
        Path to the folder containing the data files.
    lang : str
        Language of the data files.
    seed : int
        Random seed for reproducibility.
    label : str
        Label for the experiment. Added to the name of the output files.
    top_B : int
        Number of top B words to consider.
    all_data : dict[int, dict[int, list]]
        Dictionary containing all data, where keys are trial numbers and values are dictionaries of generation data.
    trials_gens_unique_words : dict[int, dict[int, set]]
        Dictionary containing unique words for each trial and generation.
    trials_gens_count_words : dict[int, dict[int, Counter]]
        Dictionary containing word counts for each trial and generation.
    trial_colors : dict[int, np.ndarray]
        Dictionary containing colors for each trial, used for visualization.
    """
    
    def __init__(self, folder_path: str, label: str, seed: int, lang: str, top_B: int):
        """
        Parameters
        ----------
        folder_path : str
            Path to the folder containing the data files.
        label : str
            Label for the experiment. Added to the name of the output files.
        seed : int
            Random seed for reproducibility.
        lang : str
            Language of the data files.
        topB : int
            Number of top B words to consider.
        """
        self.folder_path: str = folder_path
        self.lang: str = lang
        self.seed: int = seed
        self.label: str = label
        self.top_B: int = top_B
        
        self.all_data: dict[int, dict[int, list]] = {}
        self.trials_gens_unique_words: dict[int, dict[int, set]] = {}
        self.trials_gens_count_words: dict[int, dict[int, Counter]] = {}
        self.trial_colors: dict[int, np.ndarray] = {}
        
        self.set_seed(seed)
    
    def is_blank(self) -> bool:
        """
        Check if the current instance is blank (i.e., has no exp data).
        """
        return self.all_data == {}
    
    def is_empty(self) -> bool:
        """
        Check is the current instance is empty of exp computed data.
        """
        return self.trials_gens_unique_words == {} and \
                self.trials_gens_unique_words == {} and \
                self.trial_colors == {}
    
    def get_trial_topB_words(self, trial: int) -> list[str]:
        """
        Get the topB words for a given trial.
        
        Parameters
        ----------
        trial : int
            The trial number.
        """
        res = Counter()
        for count in self.trials_gens_count_words[trial].values():
            res.update(count)
        return [word for word, _ in res.most_common(self.top_B)]
    
    def get_trial_unique_words(self, trial: int) -> set:
        """
        Get the unique words for a given trial.

        Parameters
        ----------
        trial : int
            The trial number.
        """
        unique_words = set()
        for words in self.trials_gens_unique_words[trial].values():
            unique_words.update(words)
        
        return unique_words
    
    def get_all_unique_words(self) -> set:
        """
        Get all unique words across all trials.

        Returns
        -------
        set
            A set of all unique words across all trials.
        """
        unique_words = set()
        for trial in self.trials_gens_unique_words.values():
            for words in trial.values():
                unique_words.update(words)
        
        return unique_words
        
    @staticmethod
    def set_seed(SEED):
        """
        Set the random seed for the random, NumPy, and PyTorch random number generators.
        This helps ensure reproducibility of results.

        Parameters
        ----------
        SEED : int
            The random seed to be used.
        """
        rnd.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        environ['PYTHONHASHSEED'] = str(SEED)
    
    