from ExpTrajectoryAnalyzer import ExpTrajectoryAnalyzer
from ExpSpatialAnalyzer import ExpSpatialAnalyzer
from ExpTopBAnalyzer import ExpTopBAnalyzer
from ExpEmergenceAnalyzer import ExpEmergenceAnalyzer
from os import path
from sys import argv

if len(argv) > 1:
    exp_dir = path.abspath(argv[1])
    label = argv[2] if len(argv) > 2 else "default_label"
    lang = argv[3] if len(argv) > 3 else "en"
    seed = int(argv[4]) if len(argv) > 4 else 42
    top_B = int(argv[5]) if len(argv) > 5 else 10
    sentence_transformer_model = argv[6] if len(argv) > 6 else "all-MiniLM-L6-v2"

else:
    exp_dir = path.abspath(input("Enter the path to the experiment directory: "))
    label = input("Enter the label for the experiment (default: 'default_label'): ") or "default_label"
    lang =  input("Enter the language (default: 'en'): ") or "en"
    seed = int(input("Enter the seed (default: 42): ") or 42)
    top_B = int(input("Enter the value for top_B (default: 10): ") or 10)
    sentence_transformer_model = input(
        "Enter the sentence transformer model (default: 'all-MiniLM-L6-v2'): "
    ) or "all-MiniLM-L6-v2"
    
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

# emergence_analyzer = ExpEmergenceAnalyzer(
#     folder_path=exp_dir,
#     label=label,
#     lang=lang,
#     seed=seed,
#     top_B=top_B
# )

plot1 = trajectory_analyzer.create_trajectory_graph()
anims_list1 = trajectory_analyzer.create_trajectory_animation(only_trials=[0])

plot2 = spatial_analyzer.create_areas_plot()
plot3 = spatial_analyzer.create_density_plot()
plot4 = spatial_analyzer.create_exploration_coverage_plot()
plot5 = spatial_analyzer.create_metrics_textblock()

anims_list2 = topB_analyzer.create_agents_pos_anim(only_trials=[0])
anims_list3 = topB_analyzer.create_top_freq_anim(only_trials=[0])
anims_list4 = topB_analyzer.create_top_hist_anim(only_trials=[0])

# anims_list5 = emergence_analyzer.create_score_evo_anim(only_trials=[0])