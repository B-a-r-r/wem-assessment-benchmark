from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
from warnings import warn

#
# author: Clément BARRIERE
# github: B-a-r-r
#

class WemVisualsBoarding:
    """
    A class that provides attributes and methods to board visuals.
    
    Objects Attributes
    ----------
    """
    
    AVAILABLE_AXES_PROPS = ["adjustable", "alpha", "anchor", "animated", "aspect", "autoscale_on", "autoscalex_on", "autoscaley_on", "axes_locator", "axisbelow", "box_aspect", "clip_box", "clip_on", "clip_path", "facecolor", "frame_on", "in_layout", "label", "navigate", "path_effects", "picker", "position", "prop_cycle", "rasterization_zorder", "rasterized", "sketch_params", "snap", "subplotspec", "title", "transform", "url", "visible", "xbound", "xlabel", "xlim", "xmargin", "xscale", "xticklabels", "xticks", "ybound", "ylabel", "ylim", "ymargin", "yscale", "yticklabels", "yticks", "zorder"]
    
    def __init__(self, **kwargs):
        self.board_size: tuple[int, int] = kwargs.get('board_size', (10, 10))
        self.board_figure: plt.Figure = kwargs.get('board_figure', plt.figure(figsize=self.board_size))
        self.subplots: list[plt.Axes] = []
        self.ordered_visuals: list[plt.Axes | FuncAnimation]
        
        board_shape = kwargs.get('board_shape')
        assert board_shape[0] > 0 and board_shape[1] > 0, "Board shape must be greater than 0 in both dimensions."
        self.grid_spec: GridSpec = GridSpec(board_shape[0], board_shape[1], figure=self.board_figure)
        
        self._make_board()
    
    def _make_board(self):
        tmp = self.grid_spec.nrows * self.grid_spec.ncols
        
        for i in range(0, tmp, 1):
            self.subplots.append(self.board_figure.add_subplot(
                self.grid_spec[
                    i // self.grid_spec.ncols, 
                    i % self.grid_spec.ncols
                ],
            ))
    
    def board_visuals(self, ordered_visuals: list[plt.Axes | FuncAnimation]):
        """
        """
        assert len(ordered_visuals) <= self.grid_spec.ncols * self.grid_spec.nrows, "The shape is too small to fit all visuals. Increase the shape or reduce the number of visuals."
        
        self.ordered_visuals = ordered_visuals
        
        anims = []
        for idx, visual in enumerate(ordered_visuals):
            if isinstance(visual, plt.Axes):
                self.subplots[idx] = WemVisualsBoarding.copy_artist_props(visual, self.subplots[idx])
            
            elif isinstance(visual, FuncAnimation):
                anims.append(visual)
                
            else:
                warn(f"Visual {visual or idx} is not a valid type (plt.Axes or FuncAnimation). Skipping it.")
                
        self._convert_board_to_anim(anims, subplots=[self.subplots[len(ordered_visuals):]])
            
    def get_board_subplots(self) -> list[plt.Axes]:
        return self.subplots
            
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
    
    def _convert_board_to_anim(self, anims: list[FuncAnimation], subplots: list[plt.Axes]):
        def update(frame):
            for subplot in subplots:
                subplot.clear()
                for anim in anims:
                    anim._draw_frame(frame)
            for idx, visual in enumerate(self.ordered_visuals):
                if isinstance(visual, plt.Axes):
                    self.subplots[idx] = WemVisualsBoarding.copy_artist_props(visual, self.subplots[idx])
        
        return FuncAnimation(self.board_figure, update, frames=len(anims[0]), interval=50)
            
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
        
        artist_dest.set_label(artist_source.get_label())
        
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
                    