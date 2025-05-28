from pandas import read_csv
from moviepy import VideoFileClip
from os import path, remove

#
# This file contains utility functions.
#
# author:   Clément BARRIERE
# github:   @B-a-r-r
#

def convert_gif_to_video(input_path: str, output_path: str, remove_input: bool =False) -> None:
    """
    Convert a GIF file to a video file using moviepy.
    
    Parameters
    ----------
    input_path : str
        The path to the input GIF file.
    output_path : str
        The path to the output video file.
    remove_input : bool
        If True, remove the input GIF file after conversion.
    """
    if not path.exists(input_path):
        raise FileNotFoundError(f"Input file {input_path} does not exist.")
    
    if not input_path.endswith('.gif'):
        raise ValueError("Input file must be a GIF.")
    
    clip = VideoFileClip(input_path)
    clip.write_videofile(output_path, codec='libx264')
    
    if remove_input:
        clip.close()
        remove(input_path)
            
def get_gpu_local_config() -> dict | None:
    """
    Get the local GPU configuration.

    Returns
    -------
    dict
        A dictionary containing the GPU IDs as keys and their total memory in GB as values.
        If no GPU is available, returns a dictonary with a single key "0" and value 0.
    """
    from torch import cuda

    gpu_config = {}
    if cuda.is_available():
        for i in range(cuda.device_count()):
            total_memory = cuda.get_device_properties(i).total_memory / (1024 ** 3)
            gpu_config[i] = int(total_memory)
        return gpu_config
    else:
        return {0: 0}

def get_cpu_local_config() -> dict:
    """
    Get the local CPU configuration.

    Returns
    -------
    dict
        A dictionary containing two keys: "cores" and "memory".
        The number of CPU cores and the total memory in GB.
    """
    from psutil import cpu_count, virtual_memory
    
    cpu_config = {"cores": 0, "memory": 0}
    cpu_config["cores"] = cpu_count(logical=False)
    cpu_config["memory"] = int(virtual_memory().total / (1024 ** 3))
    return cpu_config

def count_non_negative_one(lst: list) -> int:
    """ 
    Count the number of non-negative one elements in a nested list.
    
    Parameters
    ----------
    lst : list
        The input list which may contain nested lists.
    """
    count = 0
    
    for item in lst:
        if isinstance(item, list):
            count += count_non_negative_one(item)
            
        elif item != -1:
            count += 1
            
    return count

def encode_dict(d: dict) -> dict:
    """
    Encodes a dictionary with tuple keys to a dictionary with string keys.
    
    Parameters
    ----------
    d : dict
        The dictionary to encode.
    
    Returns
    -------
    dict
        The encoded dictionary with string keys.
    """
    return {str(k): v for k, v in d.items()}

def decode_dict(d: dict) -> dict:
    """
    Decodes a dictionary with string keys that represent tuples back to a dictionary with tuple keys.
    
    Parameters
    ----------
    d : dict
        The dictionary to decode.
    
    Returns
    -------
    dict
        The decoded dictionary with tuple keys.
    """
    return {tuple(eval(k)): v for k, v in d.items()}

def process_file(file_path):
    """
    Process a file containing the output of a simulation into a dictionary indexed by generation.
    
    Parameters
    ----------
    file_path : str
        The path to the CSV file to process.
    
    Returns
    -------
    dict
        A dictionary whose keys are the generation numbers and values are lists of dictionaries,
        describing the agents in the simulation, with keys 'id', 'x', 'y' and 'word'.
    """
    df = read_csv(file_path)
    return df.groupby('gen').apply(lambda x: x[['id', 'x', 'y', 'word']].to_dict('records')).to_dict()