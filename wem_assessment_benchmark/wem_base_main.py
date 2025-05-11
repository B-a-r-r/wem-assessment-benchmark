from sys import argv
from Simulation import Simulation

#allow to input the config path and the logs path from the command line
if len(argv) >= 2:
    config_path = argv[0]
    logs_path = argv[1]

else:
    config_path = "config.json"
    logs_path = "logs.txt"

#init and run the simulation
Simulation(
    config_path=config_path,
    logs_path=logs_path,
    log_trial_res=True,
)