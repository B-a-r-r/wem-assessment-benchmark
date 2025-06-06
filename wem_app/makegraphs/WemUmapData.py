from os import path
import pickle
from typing import Any
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import umap
from dataclasses import dataclass
from .WemExpData import WemExpData

#
# author: Clément BARRIERE
# github: B-a-r-r
#

@dataclass
class WemUmapData(WemExpData):
    """
    A dataclass to store UMAP data for word embeddings.
    
    Object Attributes
    ---------------
    model : SentenceTransformer
        The sentence transformer model used for encoding words.
    word_vectors : dict[str, Any]
        Dictionary mapping words to their vector representations.
    word_to_umap : dict[Any, Any]
        Dictionary mapping words to their UMAP reduced vectors.
    avg_coords : dict[int, dict[int, Any | np.ndarray]]
        Dictionary mapping trial to their generations to average coordinates x and y of words. 
    """
    
    def __init__(self, 
        model: str,
        folder_path: str = "data",
        label: str = "default",
        seed: int = 42,
        lang: str = "en",
        top_B: int = 10
    ) -> None:
        """
        Parameters
        ----------
        model : str
            The name of the sentence transformer model used for encoding words.
        folder_path : str, optional
            Path to the folder containing the data files, by default "data"
        label : str, optional
            Label for the experiment. Added to the name of the output files, by default "default"
        seed : int, optional
            Random seed for reproducibility, by default 42
        lang : str, optional
            Language of the data files, by default "en"
        top_B : int, optional
            Number of top B words to consider, by default 10
        """
        super().__init__(folder_path=folder_path, label=label, seed=seed, lang=lang, top_B=top_B)
        
        self.model: SentenceTransformer = SentenceTransformer(model)
        self.word_vectors: dict[str, Any] = {}
        self.word_to_umap: dict[Any, Any] = {}   
        self.avg_coords: dict[int, dict[int, Any | np.ndarray]] = {}

    def is_umap_blank(self) -> bool:
        """Check if this instance is blank."""
        return self.word_vectors == {}
    
    def vectorize_words(self, output_prefix: str ="word_vectors.pkl") -> None:
        """
        Vectorize all unique words using the sentence transformer model.
        This method stores the word vectors in word_vectors and saves them to a pickle file.
        
        Parameters
        ----------
        output_prefix : str, optional
            The pickle file name to save the word vectors, by default "word_vectors.pkl"
        """ 
        self.word_vectors = {word: self.model.encode(word) for word in tqdm(self.get_all_unique_words())}
        self.save_umap_data(output_prefix=output_prefix)
            
    def umap_reduce(self) -> None:
        """
        Reduce the dimensionality of the word vectors using UMAP.
        The reduced vectors are stored in word_to_umap.
        """
        umap_reducer = umap.UMAP(random_state=self.seed)
        umap_embeddings = umap_reducer.fit_transform(np.array(list(self.word_vectors.values())))
        self.word_to_umap = {word: umap_embeddings[i] for i, word in enumerate(self.word_vectors.keys())}
        
    def save_umap_data(self, output_prefix: str ="word_vectors.pkl"):
        """
        Save the UMAP data to a pickle file.

        Parameters
        ----------
        output_prefix : str, optional
            The path to save the UMAP data, by default "word_vectors.pkl"
        """
        with open(path.join(self.folder_path, output_prefix), 'wb') as f:
            pickle.dump(self.word_to_umap, f)
        f.close()
    
    def get_trial_avg_coords(self, trial: int) -> np.ndarray:
        """
        Get the average coordinates of words for a specific trial.
        
        Parameters
        ----------
        trial : int
            The trial number for which to get the average coordinates.
        
        Returns
        -------
        np.ndarray
            A 2D numpy array containing the average coordinates of the words in the specified trial.
        """
        trial_avg_coords = np.ndarray(shape=(0, 2))
        
        for coords in self.avg_coords[trial].values():
            tmp = coords
            if not isinstance(tmp, np.ndarray):
                tmp = np.array(coords)
                
            trial_avg_coords = np.vstack((trial_avg_coords, tmp))
        
        return trial_avg_coords
    
    def get_trials_avg_coords(self, trials: list[int] =None) -> np.ndarray:
        """
        Get the average coordinates of all trials or specified trials.
        
        Parameters
        ----------
        trials : list[int], optional
            List of trial numbers to get the average coordinates for. If None, all trials are included.
        
        Returns
        -------
        np.ndarray
            A 2D numpy array containing the average coordinates of the specified trials.
        """
        trials = None if trials is [] else trials
        trials_avg_coords = np.ndarray(shape=(0, 2))
        
        for trial in self.avg_coords.keys():
            if trials is not None and trial not in trials:
                continue
            
            trials_avg_coords = np.vstack((trials_avg_coords, self.get_trial_avg_coords(trial)))
            
        return trials_avg_coords
    
    def compute_umap_data(self) -> bool:
        """
        Launch the UMAP data computation process.
        
        Returns
        -------
        bool
            True if the computation was successful, False if the parent exp data instance is empty.
        """
        if self.is_empty():
            return False
        
        self.vectorize_words()
        self.umap_reduce()
        return True