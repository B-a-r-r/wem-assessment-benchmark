from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
import random as rnd
import torch
from os import environ
import numpy as np

class WemVisualsMaker(ABC):
    """
    Abstrac class used as an interface for object reading experiment data 
    and creating viuals.
    """
    
    def __init__(self, subclass_instance: object) -> None:
        """
        Verify that the subclass instance has certain properties.
        
        Parameters
        ----------
        subclass_instance : object
            The instance of the subclass to verify.
        """
        valid = False
        for attr in subclass_instance.__dict__:
            if attr.startswith('create_') and callable(getattr(subclass_instance, attr)):
                valid = True
                break
        assert valid, "Subclass must have at least one method starting with 'create_'. Method to create visuals have to start with 'create_'."
    
    @abstractmethod
    def _read_data(self) -> None:
        """
        Read the experiment data.
        """
        pass
    
    @abstractmethod
    def _process_data(self) -> None:
        """
        Process the experiment data by extracting relevant information
        and adjusting related attributes.
        """
        pass
    
    @staticmethod
    def _plt_font_for_lang(self, lang: str) -> None:
        """
        Adjust the matplotlib font settings based on the language of the experiment data.
        This method sets the font family to a language-specific font to ensure proper rendering of characters.
        """
        match lang:
            case 'ch':
                plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
            case 'jp':
                plt.rcParams['font.sans-serif'] = ['Noto Sans JP']
            case _:
                plt.rcParams['font.sans-serif'] = ['Arial']