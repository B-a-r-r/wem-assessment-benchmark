from abc import ABC
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
from warnings import warn

class WemVisualsBoarding(ABC):
    """
    An abstract class that provides attributes and methods to board visuals.
    
    Attributes
    ----------
    board_figure : plt.Figure
        The figure that contains the visuals board.
    board_current_shape : tuple[int, int]
        The current shape of the board in terms of rows and columns.
    board_max_shape : tuple[int, int]
        The maximum shape of the board in terms of rows and columns.
    subplots : dict[str, plt.Axes]
        A dictionary that maps labels to subplots (axes) in the board figure.
    _pre_registered_label : dict[str, str]
        A dictionary that maps pre-registered visual labels to their corresponding creation methods.
    board_size : int
        The size of the board in inches.
    current_grid_spec : plt.GridSpec
        The current grid specification for the board figure.
    """
    
    AVAILABLE_AXES_PROPS = ["adjustable", "alpha", "anchor", "animated", "aspect", "autoscale_on", "autoscalex_on", "autoscaley_on", "axes_locator", "axisbelow", "box_aspect", "clip_box", "clip_on", "clip_path", "facecolor", "frame_on", "in_layout", "label", "navigate", "path_effects", "picker", "position", "prop_cycle", "rasterization_zorder", "rasterized", "sketch_params", "snap", "subplotspec", "title", "transform", "url", "visible", "xbound", "xlabel", "xlim", "xmargin", "xscale", "xticklabels", "xticks", "ybound", "ylabel", "ylim", "ymargin", "yscale", "yticklabels", "yticks", "zorder"]
    
    def __init__(self, visuals_maker_instance: object):
        self.board_figure: plt.Figure = None
        self.board_current_shape: list[int, int] = [1, 1]
        self.board_max_shape: tuple[int, int]
        self.subplots: dict[str, plt.Axes] = {}
        self._pre_registered_label: dict[str, str] = []
        self.board_size: int = (10,10)
        self.current_grid_spec: plt.GridSpec = plt.GridSpec(1, 1)
        
        for attr_name, attr in visuals_maker_instance.__dict__.items():
            if attr_name.startswith('create_') and callable(attr):
                self._pre_registered_label[attr_name.removeprefix('create_')] = attr
        
    def make_board(self, 
        board_size: list[int, int] =[10,10], 
        board_max_shape: tuple[int, int] =(100, 100),
        visuals: list[plt.Axes | FuncAnimation | str] =[]
    ):
        """
        Reset and create a board for the visuals.
        
        Parameters
        ----------
        board_size : list[int, int], optional
            The size of the board in inches, by default (10, 10).
        board_max_shape : tuple[int, int], optional
            The maximum shape of the board in terms of rows and columns, by default (100, 100).
        visuals : list[plt.Figure | plt.Axes | FuncAnimation | "all"], optional
            A list of visuals to add to the board. Can contain figures, axes or animations.
            If "all" is present, all pre-registered visuals will be added to the board, by default [].
        """
        self.board_size = board_size
        self.board_max_shape = board_max_shape
        
        #if visuals contains "all", add every pre registered visuals,
        #from the subclass, to the list of visuals to plot.
        if visuals.__contains__("all"):
            visuals.remove("all")
            visuals.extend([
                self._pre_registered_label[visual](save=False)
                for visual in self._pre_registered_label
            ])
        
        #labels must be unique, replace duplicates with their index.   
        labels = [visual.get_label() if hasattr(visual, 'get_label') else str(visual) for visual in visuals]
        for idx, label in enumerate(labels):
            if labels.count(label) > 1:
                label = str(idx)
                
        self._reset_board_figure()

        for idx, visual in enumerate(visuals):
            self.add_visual_to_board(visual=visual, label=labels[idx])
        
    def add_visual_to_board(self, visual: plt.Axes | FuncAnimation, label: str):
        """
        Add a visual to the board.
        If the label is already registered, append the visual to the existing subplots.
        
        Parameters
        ----------
        visual : plt.Figure | plt.Axes | FuncAnimation
            The visual to add to the board. Can be a figure, axes or animation.
        label : str
            The label of the visual. 
        """
        if not self._increment_shape(1):
            warn("Couldn't add another visual to the board: not enough space remaining." \
            + "Try either to remake a board with a larger max shape or don't specify any.")
            return
        
        self._actualize_grid_spec()
        
        if isinstance(visual, plt.Axes):
            self._add_subplot_to_board(visual, label)
        
        elif isinstance(visual, FuncAnimation):
            self._add_animation_to_board(visual, label)
            
        else:
            warn(f"Visual of type {type(visual)} is not supported. Only plt.Axes and FuncAnimation are supported.")
            return
        
        self._assign_subplots()
        self.board_figure.tight_layout()
        
        print(self.subplots)
        print(self.current_grid_spec)
        print(self.board_figure.get_axes())
        
    def _add_subplot_to_board(self, ax: plt.Axes, label: str):
        """
        Add a subplot (axes) to the board figure.
        If the label is already registered, append the subplot to the existing subplots.
        
        Parameters
        ----------
        ax : plt.Axes
            The axes to add to the board.
        label : str
            The label of the subplot.
        """
        ax_new = ax_new = self.board_figure.add_subplot(
            self.current_grid_spec[
                len(self.subplots) -1 if len(self.subplots) > 0 else 0,
                0
            ],
            label=label
        )
        self.subplots[label] = WemVisualsBoarding.copy_artist_props(ax, ax_new)

    def _add_animation_to_board(self, anim: FuncAnimation, label: str):
        """
        Add an animation to the board figure.
        If the label is already registered, append the animation to the existing subplots.
        
        Parameters
        ----------
        anim : FuncAnimation
            The animation to add to the board.
        label : str
            The label of the animation.
        """
        fig = self.subfigs[label]
        ax = fig.add_subplot(111)
        
        line_data = anim.frame_seq[0].get_data()
        anim_line = ax.plot(line_data[0], line_data[1])
        ax.set_title(anim.frame_seq[0].axes.get_title())
        ax.set_xlabel(anim.frame_seq[0].axes.get_xlabel())
        ax.set_ylabel(anim.frame_seq[0].axes.get_ylabel())
        ax.legend(anim.frame_seq[0].axes.get_legend_handles_labels()[0], 
                  anim.frame_seq[0].axes.get_legend_handles_labels()[1])
        ax.set_xlim(anim.frame_seq[0].axes.get_xlim())
        ax.set_ylim(anim.frame_seq[0].axes.get_ylim())

        def update(frame):
            # Mettre à jour les données de l'anim
            line_data = anim.frame_seq[frame].get_data()
            anim_line.set_data(line_data[0], line_data[1])
            return anim_line,

        FuncAnimation(fig, update, frames=anim.frames, interval=anim._interval, blit=True)
            
    def _increment_shape(self, incr: int) -> bool:
        """
        Increment the current shape of the board by a given increment, and
        actualize the board figure gridspec.
        If the increment exceeds the maximum shape, return False. Else, return True.
        
        Parameters
        ----------
        incr : int
            The increment to add to the current shape.
        """
        if not self.board_current_shape[1] + incr <= self.board_max_shape[1]:
            
            if self.board_current_shape[0] + incr <= self.board_max_shape[0]:
                if len(self.subplots) > 0:
                    self.board_current_shape[0] += incr
            else: 
                return False
        else:
            if len(self.subplots) > 0:
                self.board_current_shape[1] += incr
                    
        return True
    
    def _rearange_positions(self):
        self._assign_subplots()
        # tmp = deepcopy(self.subplots)
        # self._reset_board_figure()
        
        # for idx, (label, subplot) in enumerate(tmp.items()):
        #     self._assign_subplots(label, index=idx)
        #     self.subplots[label] = WemVisualsBoarding.copy_artist_props(subplot, self.subplots[label])
            
        # del tmp
                
    def _assign_subplots(self):
        for idx, subplot in enumerate(self.subplots.values()):
            i = idx -1 if idx > 0 else 0
            subplot.set_position(self.current_grid_spec[i, 0].get_position(self.board_figure))
            subplot.set_subplotspec(self.current_grid_spec[i, 0])
    
    def _reset_board_figure(self):
        """
        Reset the board figure to a new figure with the current board size.
        """
        if self.board_figure is None:
            self.board_figure = plt.figure(figsize=self.board_size)
        
        self.board_figure.clear()
        self._actualize_grid_spec()
    
    def _actualize_grid_spec(self):
        num_subplots = len(self.subplots) + 1
        self.current_grid_spec = GridSpec(
            num_subplots,
            1, 
            wspace=0.5, 
            hspace=0.5,
            figure=self.board_figure
        )
    
    def return_visuals_board(self, show: bool =False):
        """
        Return the visuals board figure, optionally showing it.
        """
        if show:
            if self.board_figure is None:
                warn("Couldn't show the visuals board: board figure is None.")
            else:
                self.board_figure.show()
                plt.show()
                
        return self.board_figure
                    
    @staticmethod
    def copy_artist_props(artist_source: plt.Axes, artist_dest: plt.Axes) -> plt.Axes:
        """
        Copy properties from one artist to another (axes).
        
        Parameters
        ----------
        artist_source : plt.Axes
            The source artist from which to copy properties.
        artist_dest : plt.Axes
            The destination artist to which properties will be copied.
        
        Returns
        -------
        plt.Axes
            The destination artist with copied properties.
        """
        assert type(artist_source) == type(artist_dest), \
            "Source and destination artists must be of the same type."
            
        for prop, value in artist_source.properties().items():
            if prop in WemVisualsBoarding.AVAILABLE_AXES_PROPS:
                artist_dest.set(**{prop: value})
                
        for line in artist_source.get_lines():
            artist_dest.plot(line.get_xdata(), line.get_ydata(),
                        color=line.get_color(),
                        linestyle=line.get_linestyle(),
                        linewidth=line.get_linewidth(),
                        label=line.get_label(),
                        marker=line.get_marker())

        if artist_source.get_legend():
            artist_dest.legend()
                
        return artist_dest
                    