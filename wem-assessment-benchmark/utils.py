#
# This file contains utility functions.
#
# author:   Clément BARRIERE
# github:   @B-a-r-r
#

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