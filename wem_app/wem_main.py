from sys import argv
from Simulation import Simulation
from os import path

#allow to input the config path and the logs path from the command line
if len(argv) > 1:
    config_path = argv[1]
    
    if len(argv) > 2:
        enable_logs = bool(argv[2])
    
    else:
        enable_logs = True

else:
    config_path = path.join(path.dirname(__file__), "config.json")
    enable_logs = True

#init and run the simulation
simulation = Simulation(
    config_path=config_path,
    enable_logs=enable_logs,
)

