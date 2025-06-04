from os import path
import pickle
from typing import Any
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import umap
from dataclasses import dataclass
from WemExpData import WemExpData


@dataclass
class WemUmapData(WemExpData):
    """
    A dataclass to store UMAP data for word embeddings.
    
    Object Attributes
    ---------------
    model : SentenceTransformer
        The sentence transformer model used for encoding words.
    word_vectors: dict[str, Any]
        Dictionary containing the vectorized words.
    word_to_umap: dict[Any, Any]
        Dictionary mapping words to their UMAP-reduced coordinates.
    avg_coords: dict[int, dict[int, Any | np.ndarray]]
        Dictionary containing the average coordinates of words for each generation in each file.
    """
    
    def __init__(self, 
        model: SentenceTransformer,
        folder_path: str = "data",
        label: str = "default",
        seed: int = 42,
        lang: str = "en",
        top_B: int = 10
    ) -> None:
        """
        Parameters
        ----------
        model : SentenceTransformer
            The sentence transformer model used for encoding words.
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
        
        self.model: SentenceTransformer = model
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
        self.save_umap_data(output_path=self.folder_path, output_prefix=output_prefix)
            
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
        
    def import_exp_data(self, exp_data: WemExpData) -> None:
        """
        Import experiment data from a WemExpData instance.
        Reprocesses the umap representation based on the imported data.
        
        Parameters
        ----------
        exp_data : WemExpData
            An instance of WemExpData containing the experiment data to be imported.
        """
        self.folder_path = exp_data.folder_path
        self.lang = exp_data.lang
        self.seed = exp_data.seed
        self.label = exp_data.label
        self.top_B = exp_data.top_B
        
        self.all_data = exp_data.all_data
        self.trials_gens_unique_words = exp_data.trials_gens_unique_words
        self.trials_gens_count_words = exp_data.trials_gens_count_words
        self.trial_colors = exp_data.trial_colors
        self.set_seed(self.seed)
        
        self.vectorize_words()
        self.umap_reduce()