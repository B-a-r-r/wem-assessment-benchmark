from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from os import path, makedirs, remove
from .WemExpData import WemExpData
from moviepy import VideoFileClip
from warnings import warn

#
# author: Clément BARRIERE
# github: B-a-r-r
#

class WemVisualsMaker(ABC):
    """
    Abstrac class for classes that create visuals from experiment data.
    """
    
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
    
    @abstractmethod
    def _save_visual(self, fig: plt.Figure | FuncAnimation, filename: str, output_extension: str) -> None:
        """
        Save the created visual to a file.
        """
        match type(fig):
            case plt.Figure:
                fig.savefig(filename, format=output_extension, dpi=300, bbox_inches='tight')
                
            case FuncAnimation:
                fig.save(filename, writer='ffmpeg', fps=10)
        
        print(f"Fig saved as {filename}")
    
    @abstractmethod
    def _create_dedicated_dir(self, folder_path: str) -> str:
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
        if not path.exists(path.abspath(folder_path)):
            makedirs(path.abspath(folder_path))
            
        return path.abspath(folder_path)
    
    @staticmethod
    def _trials_specifier_for_title(only_trials: list[int] =None) -> str:
        """
        Create a string that specifies which trials are included in the visual.
        If no trials are specified, it defaults to "all trials".
        
        Parameters
        ----------
        only_trials : list[int], optional
            A list of trial numbers to include in the specification. If None, all trials are included.
        
        Returns
        -------
        str
            A string that specifies the trials included in the visual.
            If no trials are specified, it returns "all trials".
        """
        spec_trials = "all trials"
        if only_trials is not None:
            spec_trials = "trials "
            for t in only_trials:
                spec_trials += f"{t}{", " if t != only_trials[-1] else ""}" 
                
        return spec_trials
    
    @abstractmethod
    def _plt_font_for_lang(self, lang: str) -> None:
        """
        Adjust the matplotlib font settings based on the language of the experiment data.
        This method sets the font family to a language-specific font to ensure proper rendering of characters.
        
        Parameters
        ----------
        lang : str
            The language code for the experiment data (e.g., 'ch' for Chinese, 'jp' for Japanese).
        """
        match lang:
            case 'ch':
                plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
            case 'jp':
                plt.rcParams['font.sans-serif'] = ['Noto Sans JP']
            case _:
                plt.rcParams['font.sans-serif'] = ['Arial']
                
    @abstractmethod
    def import_exp_data(self, exp_data: WemExpData) -> None:
        """
        Import the experiment data from the specified WemExpData object.
        
        Parameters
        ----------
        exp_data : WemExpData
            The WemExpData object containing the experiment data to be imported.
        """
        pass
    
    @abstractmethod
    def export_exp_data(self) -> WemExpData:
        """
        Export the WemExpData object containing the processed experiment data.
        
        Returns
        -------
        WemExpData
            The WemExpData object containing the processed experiment data.
        """
        pass
    
    def convert_to_video(self, input_path: str, output_path: str, remove_input: bool =False) -> None:
        """
        Convert an input file to a video format and save it to the specified output path.
        
        Parameters
        ----------
        input_path : str
            The path to the input file to be converted.
            The extension is deduced from the file name (e.g., .gif, .mp4, .avi).
        output_path : str
            The path where the converted video will be saved.
            The extension should be one of the supported video formats (e.g., .mp4, .avi).
        remove_input : bool, optional
            If True, the input file will be removed after conversion. Default is False.
        """
        if not path.exists(input_path):
            warn(f"Could not convert to video: {input_path} does not exist.")
            return
        
        clip = VideoFileClip(input_path)
        clip.write_videofile(output_path, codec='libx264')
        
        if remove_input:
            clip.close()
            remove(input_path)
    
    