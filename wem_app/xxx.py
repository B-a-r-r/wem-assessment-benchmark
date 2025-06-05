import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from utils import load_config, verify_config
from os import path

if __name__ == "__main__":
    from ExpTrajectoryAnalyzer import ExpTrajectoryAnalyzer
    from ExpSpatialAnalyzer import ExpSpatialAnalyzer
    from ExpTopBAnalyzer import ExpTopBAnalyzer
    from ExpEmergenceAnalyzer import ExpEmergenceAnalyzer
    
    config_path = path.abspath("./config.json")
    config = load_config(config_path)
    verify_config(config=config)
    
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
