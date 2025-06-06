from makegraphs import *
from os import path
from sys import argv
from utils import load_config, verify_config

#Usage: wem_makegraph.py <config_path> [--ui]

if len(argv) > 1:
    config_path = argv[1]
else:
    config_path = path.join(path.dirname(__file__), "config.json")

config = load_config(path.abspath(config_path))
verify_config(config=config)

exp_dir = config['workspace']["exp_dir"]
lang = config['workspace']["lang"]
label = config['workspace']["label"]
top_B = int(config['workspace']["top_B"])
seed = int(config['simulation']["SEED"])
sentence_transformer_model = config['workspace']["sentence_transformer_model"]

if "--ui" in argv:
    #TODO: Implement and link the VisualsMakerUI class for cli visuals selection.
    pass

trajectory_analyzer = ExpTrajectoryAnalyzer(
    folder_path=exp_dir,
    label=label,
    lang=lang,
    seed=seed,
    top_B=top_B,
    sentence_transformer_model=sentence_transformer_model
)

spatial_analyzer = ExpSpatialAnalyzer(
    folder_path=exp_dir,
    label=label,
    lang=lang,
    seed=seed,
    top_B=top_B,
    sentence_transformer_model=sentence_transformer_model
)

topB_analyzer = ExpTopBAnalyzer(
    folder_path=exp_dir,
    label=label,
    lang=lang,
    seed=seed,
    top_B=top_B
)

emergence_analyzer = ExpEmergenceAnalyzer(
    folder_path=exp_dir,
    label=label,
    lang=lang,
    seed=seed,
    top_B=top_B
)

trajectory_analyzer.create_trajectory_graph()
trajectory_analyzer.create_trajectory_animation(only_trials=[0])

spatial_analyzer.create_areas_plot()
spatial_analyzer.create_density_plot()
spatial_analyzer.create_exploration_coverage_plot()
spatial_analyzer.create_metrics_textblock()

topB_analyzer.create_agents_pos_anim(only_trials=[0])
topB_analyzer.create_top_freq_anim(only_trials=[0])
topB_analyzer.create_top_hist_anim(only_trials=[0])

emergence_analyzer.create_score_evo_anim(only_trials=[0])