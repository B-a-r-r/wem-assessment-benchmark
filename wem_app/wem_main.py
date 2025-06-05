from sys import argv
from simulation import Simulation
from makegraphs import *
from os import path
from utils import load_config, verify_config

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

config_path = path.abspath(config_path)
print(f"Config path: {config_path}")

config = load_config(config_path)
verify_config(config=config)

#init and run the simulation
simulation = Simulation(
    config=config,
    enable_logs=enable_logs,
)

trajectory_analyzer = ExpTrajectoryAnalyzer(
    folder_path=path.abspath(config["workspace"]["exp_dir"]),
    label=config["workspace"]["label"],
    lang=config["workspace"]["lang"],
    seed=config["simulation"]["SEED"],
    top_B=config["workspace"]["top_B"],
    sentence_transformer_model=config["workspace"]["sentence_transformer_model"],
)
umap_data = trajectory_analyzer.export_exp_data()

spatial_analyzer = ExpSpatialAnalyzer(exp_data=umap_data)

topB_analyzer = ExpTopBAnalyzer(
    folder_path=path.abspath(config["workspace"]["exp_dir"]),
    label=config["workspace"]["label"],
    lang=config["workspace"]["lang"],
    seed=config["simulation"]["SEED"],
    top_B=config["workspace"]["top_B"]
)

emergence_analyzer = ExpEmergenceAnalyzer(
    folder_path=path.abspath(config["workspace"]["exp_dir"]),
    label=config["workspace"]["label"],
    lang=config["workspace"]["lang"],
    seed=config["simulation"]["SEED"],
    top_B=config["workspace"]["top_B"]
)

plot1 = trajectory_analyzer.create_trajectory_graph()
anims_list1 = trajectory_analyzer.create_trajectory_animation(only_trials=[0])

plot2 = spatial_analyzer.create_areas_plot()
plot3 = spatial_analyzer.create_density_plot()
plot4 = spatial_analyzer.create_exploration_coverage_plot()
plot5 = spatial_analyzer.create_metrics_textblock()

anims_list2 = topB_analyzer.create_agents_pos_anim(only_trials=[0])
anims_list3 = topB_analyzer.create_top_freq_anim(only_trials=[0])
anims_list4 = topB_analyzer.create_top_hist_anim(only_trials=[0])

anims_list5 = emergence_analyzer.create_score_evo_anim(only_trials=[0])
