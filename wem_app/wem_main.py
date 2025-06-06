from sys import argv
from simulation import Simulation
from os import path
from utils import load_config, verify_config

#Usage: wem_main.py <config_path> [enable_logs] [enable_real_time_views]

if len(argv) > 1:
    config_path = argv[1]
    
    enable_logs = True
    enable_real_time_views = True
    
    if len(argv) > 2:
        enable_logs = bool(argv[2])

        if len(argv) > 3:
            enable_real_time_views = bool(argv[3])

else:
    config_path = path.join(path.dirname(__file__), "config.json")
    enable_logs = True
    enable_real_time_views = True

config_path = path.abspath(config_path)
print(f"Config path: {config_path}")

config = load_config(config_path)
verify_config(config=config)

#init and run the simulation
simulation = Simulation(
    config=config,
    enable_logs=enable_logs,
    enable_real_time_views=enable_real_time_views
)
