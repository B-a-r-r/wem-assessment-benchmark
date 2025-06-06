import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from warnings import warn
from typing import Any
from .WemVisualsMaker import WemVisualsMaker
from .WemVisualsBoarding import WemVisualsBoarding
from sklearn.cluster import DBSCAN
from .WemUmapData import WemUmapData
from os import path, listdir
import pandas as pd
from collections import Counter
from scipy.spatial import ConvexHull
from typing import Literal 

#
# author: Reiji SUZUKI et al.
# refactor: Clément BARRIERE
#

class ExpSpatialAnalyzer(WemVisualsMaker):
    """
    Quantitative analysis class for 2D spatial data using kernel density estimation
    
    Object Attributes
    -----------------
    exp_data : WemUmapData
        An instance of the experiment data containing UMAP-reduced word embeddings and general experiement data.
    bandwidth : float or str
        Bandwidth for the kernel density estimation, can be 'scott', 'silverman', or a float value.
    kernel : str
        Kernel type for the KDE, default is 'gaussian'.
    density_grid_x : np.ndarray
        X-coordinates of the density grid.
    density_grid_y : np.ndarray
        Y-coordinates of the density grid.
    density_meshgrid_x : np.ndarray[np.floating[Any], Any]
        Meshgrid X-coordinates for the density grid.
    density_meshgrid_y : np.ndarray[np.floating[Any], Any]
        Meshgrid Y-coordinates for the density grid.
    grid_density_map : np.ndarray[tuple[int, ...], np.dtype[Any]]
        2D array representing the density values on the grid.
    density_grid_resolution : int
        Resolution of the density grid, default is 100.
    density_margin_factor : float
        Margin factor for the density grid, default is 0.1.
    current_density_grid : list[int]
        List of trial indices currently being analyzed for density.
    """
    
    def __init__(self, **kwargs):
        """
        Parameters
        -----------
        folder_path : str, optional
            Path to the experiment folder containing data files, needed if no exp_data is provided.
        label : str, optional
            Label for the experiment, by default ''
        seed : int, optional
            Random seed for reproducibility, by default 42
        lang : str, optional
            Language of the data, by default 'en'
        top_B : int, optional
            Number of top B words to consider, by default 10
        sentence_transformer_model : str, optional
            Name of the sentence transformer model to use, by default 'all-MiniLM-L6-v2'
        bandwidth : float or str, optional
            Bandwidth for the kernel density estimation, by default 'scott'.
        kernel : str, optional
            Kernel type for the KDE, by default 'gaussian'.
        density_grid_resolution : int, optional
            Resolution of the density grid, by default 100.
        density_margin_factor : float, optional
            Margin factor for the density grid, by default 0.1.
        exp_data : WemUmapData, optional
            An instance of existing experiment data, by default None.
            If provided, other kwargs are ignored.
        """
        assert 'folder_path' in kwargs or 'exp_data' in kwargs, "Either provide an experiment folder path or an instance of existing experiment data."
        
        exp_data = kwargs.get('exp_data', None)
        if exp_data is None:
            exp_data = WemUmapData(
                folder_path=kwargs.get('folder_path'),
                label=kwargs.get('label', ''),
                seed=kwargs.get('seed', 42),
                lang=kwargs.get('lang', 'en'),
                top_B=kwargs.get('top_B', 10),
                model=kwargs.get('sentence_transformer_model', 'all-MiniLM-L6-v2')
            )
        
        self.exp_data: WemUmapData = exp_data
        self.bandwidth = kwargs.get('bandwidth', 'scott')
        self.kernel = kwargs.get('kernel', 'gaussian')
        
        self.density_grid_x: np.ndarray = []
        self.density_grid_y: np.ndarray = []
        self.density_meshgrid_x: np.ndarray[np.floating[Any], Any] = []
        self.density_meshgrid_y: np.ndarray[np.floating[Any], Any] = []
        self.current_density_grid: list[int] = []
        self.density_grid_resolution: int = kwargs.get('density_grid_resolution', 100)
        self.density_margin_factor: float = kwargs.get('density_margin_factor', 0.1)
        self.grid_density_map: np.ndarray[tuple[int, ...], np.dtype[Any]] = []
        
        self._plt_font_for_lang()
        self._read_data()
        self._process_data()
        self.exp_data.folder_path = self._create_dedicated_dir(path.join(self.exp_data.folder_path, "spacial-analysis"))

    def _plt_font_for_lang(self) -> None:
        """
        Set the matplotlib font for the specified language.
        """
        super()._plt_font_for_lang(lang=self.exp_data.lang)
        
    def _create_dedicated_dir(self, folder_path) -> str:
        """
        Create a dedicated directory for saving visuals if it does not exist.
        If the directory already exists, it does nothing.
        
        Parameters
        ----------
        folder_path : str
            The path to the folder where the visuals will be saved.
            
        Returns
        -------
        str:
            The absolute path to the created or existing directory.
        """
        return super()._create_dedicated_dir(folder_path)
    
    def _read_data(self):
        """
        Read the experiment data from the specified folder path.
        If experiement data is already provided, it checks the type of the first element.
        Here it expects the first element to be a string representing words.
        If the first element is not a string, it attempts to reread data files from the folder path.
        """
        first_element = None
        if not self.exp_data.is_blank():
            first_element = self.exp_data.all_data[0][0][0]
        
        tmp = type(first_element)
        if not self.exp_data.is_blank() and not tmp == str:
            warn("The units in the provided experiment data must be string " \
                + f"representing words to create trajectory visuals, found {tmp} instead.\n" \
                + "NB: dict type is used to create topB animation visuals.\n" \
                + "---> Rereading experiment data...\n"
            )
            
        if self.exp_data.is_blank() or not tmp == str:
            csv_idx = 0
            for file in listdir(path.abspath(self.exp_data.folder_path)):
                if file.endswith('csv'):
                    file_path = path.join(self.exp_data.folder_path, file)
                    self.exp_data.all_data[csv_idx] = (\
                        pd.read_csv(file_path, usecols=['gen', 'word'], encoding='utf-8')\
                        .groupby('gen')['word'].apply(list).to_dict()
                    )
                    csv_idx += 1
            
            if csv_idx == 0:
                warn("No CSV files found in the provided folder path.")
    
    def _process_data(self):
        """
        Process the experiment data by extracting relevant information
        and adjusting related attributes.
        """
        if self.exp_data.is_empty():
            for file, data in self.exp_data.all_data.items():
                
                self.exp_data.trials_gens_unique_words[file] = {}
                self.exp_data.trials_gens_count_words[file] = {}
                word_counts = None
                
                for gen, words in data.items():
                    self.exp_data.trials_gens_unique_words[file][gen] = set()
                    self.exp_data.trials_gens_unique_words[file][gen].update(words)
                    
                    word_counts = Counter(words)
                    
                    self.exp_data.trials_gens_count_words[file] = {}
                    self.exp_data.trials_gens_count_words[file][gen] = word_counts
            
            if self.exp_data.is_umap_blank():
                self.exp_data.vectorize_words()
                self.exp_data.umap_reduce()
                
            for file, data in self.exp_data.all_data.items():
                self.exp_data.avg_coords[file] = {}
                prev_gen = None
                for gen, words in data.items():
                    coords = np.array([self.exp_data.word_to_umap[word] for word in words if word in self.exp_data.word_to_umap])
                    self.exp_data.avg_coords[file][gen] = coords.mean(axis=0) if len(coords) > 0 \
                        else self.exp_data.avg_coords[file][prev_gen] if prev_gen is not None else np.array([np.nan, np.nan])
                    prev_gen = gen
    
    def _auto_bandwidth(self, shape: tuple[Any, Any]) -> float:
        """
        Automatically determine the bandwidth for KDE based on the shape of the data.
        
        Parameters
        -----------
        shape : tuple[int, int]
            Shape of the data array (n_samples, n_features).
            
        Returns
        --------
        float :
            Calculated bandwidth for the KDE.
        """
        match self.bandwidth:
            case 'scott':
                # Scott's rule: n^(-1/(d+4))
                n, d = shape
                return n ** (-1.0 / (d + 4))
                
            case 'silverman':
                # Silverman's rule
                n, d = shape
                return (n * (d + 2) / 4.) ** (-1. / (d + 4))
    
    def _fit_kde(self, only_trials: list[int] =None) -> KernelDensity:
        """
        Fit Kernel Density Estimation (KDE) models for the trials.
        
        Parameters
        -----------
        only_trials : list[int], optional
            List of trial indices to fit KDE for. If None, fits for all trials.
            If empty list, fits for all trials.
        """
        all_coords = self._return_right_coords(only_trials)
            
        bandwidth = self._auto_bandwidth(all_coords.shape)
        kde = KernelDensity(kernel=self.kernel, bandwidth=bandwidth)
        kde.fit(all_coords)
        return kde
        
    def _create_density_grid(self, only_trials: list[int] =None):
        """
        Create a grid for density estimation based on the trial data
        
        Parameters
        -----------
        only_trials : list[int], optional
            List of trial indices to include in the density estimation
            If None, includes all trials
            If empty list, includes all trials
        """ 
        data = self._return_right_coords(only_trials)
        kde = self._fit_kde(only_trials)
            
        # Get data range
        x_min, x_max = data[:, 0].min(), data[:, 0].max()
        y_min, y_max = data[:, 1].min(), data[:, 1].max()
        
        # Add margins
        x_range, y_range = x_max - x_min, y_max - y_min
        x_margin, y_margin = x_range * self.density_margin_factor, y_range * self.density_margin_factor
        
        x_min -= x_margin
        x_max += x_margin
        y_min -= y_margin
        y_max += y_margin
    
        # Create grid
        self.density_grid_x = np.linspace(x_min, x_max, self.density_grid_resolution)
        self.density_grid_y = np.linspace(y_min, y_max, self.density_grid_resolution)
        self.density_meshgrid_x, self.density_meshgrid_y = np.meshgrid(self.density_grid_x, self.density_grid_y)
        
        # Calculate density
        grid_points = np.vstack([self.density_meshgrid_x.ravel(), self.density_meshgrid_y.ravel()]).T
        log_density = kde.score_samples(grid_points)
        self.grid_density_map = np.exp(log_density).reshape(self.density_meshgrid_x.shape)
        
        self.current_density_grid = only_trials

    def _calculate_levels_from_percentiles(self, percentiles: list[int]) -> list[np.floating[Any]]:
        """
        Calculate density levels for specified percentiles
        
        Parameters
        -----------
        percentiles : list[int]
            List of percentiles to calculate levels for (e.g., [50, 95])

        Returns
        --------
        list[np.floating[Any]] : 
            Density levels corresponding to the specified percentiles
        """
        return [np.percentile(self.grid_density_map[self.grid_density_map > 0], p) for p in percentiles]
    
    def _calculate_contour_areas(self,
        only_trials: list[int] =None,
        levels: list[np.floating[Any]] =None, 
        percentiles: list[int] =[50, 60, 80, 97]
    ) -> dict:
        """
        Calculate areas of regions above specified density levels
        
        Parameters
        -----------
        only_trials : list[int], optional
            List of trial indices to include in the area calculation
            If None, includes all trials
            If empty list, includes all trials
        levels : list[np.floating[Any]], optional
            List of density levels to calculate areas for
            If None, uses percentiles to calculate levels
        percentiles : list[int], optional
            List of percentiles to calculate levels from
            If provided, overrides levels parameter

        Returns
        --------
        dict :
            Dictionary mapping level names to calculated areas
            e.g., {'50th percentile': area, '60th percentile': area, ...}
        """
        only_trials = None if only_trials is [] else only_trials
        if not self.current_density_grid == only_trials:
            self._create_density_grid(only_trials=only_trials)
        
        if percentiles is not None:
            # Calculate density levels from percentiles
            levels = self._calculate_levels_from_percentiles(percentiles)
            level_names = [f'{p}th percentile' for p in percentiles]
        else:
            level_names = [f'Level {l:.4f}' for l in levels]
            
        areas = {}
        
        # Grid cell area
        dx = self.density_grid_x[1] - self.density_grid_x[0]
        dy = self.density_grid_y[1] - self.density_grid_y[0]
        cell_area = dx * dy
        
        for level, name in zip(levels, level_names):
            # Identify regions above specified level
            mask = self.grid_density_map >= level
            area = np.sum(mask) * cell_area
            areas[name] = area
            
        return areas
    
    def _calculate_density_statistics(self, only_trials: list[int] =None) -> dict:
        """
        Calculate statistics for the density distribution of a specific trial
        
        Parameters
        -----------
        only_trials : list[int], optional
            List of trial indices to include in the density statistics
            If None, includes all trials
            If empty list, includes all trials
        
        Returns
        --------
        dict :
            Dictionary containing density statistics:
            - max_density: Maximum density value
            - mean_density: Mean density value
            - std_density: Standard deviation of density values
            - density_range: Range of density values (max - min)
            - entropy: Shannon entropy of the density distribution
        """
        only_trials = None if only_trials is [] else only_trials
        if not self.current_density_grid == only_trials:
            self._create_density_grid(only_trials=only_trials)
        
        return {
            'max_density': np.max(self.grid_density_map),
            'mean_density': np.mean(self.grid_density_map),
            'std_density': np.std(self.grid_density_map),
            'density_range': np.max(self.grid_density_map) - np.min(self.grid_density_map),
            'entropy': -np.sum(
                self.grid_density_map * np.log(self.grid_density_map + 1e-10)) 
                * (self.density_grid_x[1] - self.density_grid_x[0]) 
                * (self.density_grid_y[1] - self.density_grid_y[0])
        }
    
    def _return_right_coords(self, only_trials: list[int] =None) -> np.ndarray:
        """
        Return the coordinates for the specified trial(s).
        
        Parameters
        -----------
        only_trials : list[int], optional
            List of trial indices to include in the coordinates
            If None, includes all trials
            If empty list, includes all trials
        
        Returns
        --------
        np.ndarray :
            Array of coordinates for the specified trial
            Shape: (n_points, 2)
        """
        only_trials = None if only_trials is [] else only_trials
        
        if only_trials is None:
            return self.exp_data.get_trials_avg_coords()
        
        elif len(only_trials) == 1:
            return self.exp_data.get_trial_avg_coords(only_trials[0])
        
        else:
            return self.exp_data.get_trials_avg_coords(only_trials)
    
    def _calculate_unique_points(self, only_trials: list[int] =None, tolerance: float =None):
        """
        Calculate unique points in the trial data based on a distance threshold
        
        Parameters
        -----------
        only_trials : list[int], optional
            List of trial indices to include in the unique point calculation
            If None, includes all trials
            If empty list, includes all trials
        tolerance : float, optional
            Distance threshold for considering points as unique
            If None, uses adaptive tolerance based on data range
        
        Returns
        --------
        dict :
            Dictionary containing:
            - n_unique_points: Number of unique points
            - n_total_points: Total number of points
            - unique_points: List of unique point coordinates and their counts
            - tolerance: Used distance threshold
            - redundancy_ratio: Fraction of redundant points
        """
        only_trials = None if only_trials is [] else only_trials
        if not self.current_density_grid == only_trials:
            self._create_density_grid(only_trials=only_trials)
            
        coords = self._return_right_coords(only_trials)
         
        if tolerance is None:
            # Adaptive tolerance: 1% of average data range
            x_range = coords[:, 0].max() - coords[:, 0].min()
            y_range = coords[:, 1].max() - coords[:, 1].min()
            tolerance = 0.01 * np.mean([x_range, y_range])
        
        # Find unique points using distance-based clustering
        clustering = DBSCAN(eps=tolerance, min_samples=1)
        cluster_labels = clustering.fit_predict(coords)
        
        unique_points = []
        for cluster_id in np.unique(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            cluster_points = coords[cluster_mask]
            # Use centroid as representative point
            centroid = np.mean(cluster_points, axis=0)
            unique_points.append({
                'centroid': centroid,
                'point_count': np.sum(cluster_mask),
                'original_indices': np.where(cluster_mask)[0]
            })
        
        return {
            'n_unique_points': len(unique_points),
            'n_total_points': len(coords),
            'unique_points': unique_points,
            'tolerance': tolerance,
            'redundancy_ratio': 1 - len(unique_points) / len(coords)
        }
    
    def _calculate_exploration_metrics(self, only_trials: list[int] =None, coverage_percentile: int =95, tolerance: float =None):
        """
        Calculate exploration metrics for a specific trial
        
        Parameters
        -----------
        only_trials : list[int], optional
            List of trial indices to include in the exploration metrics
            If None, includes all trials
            If empty list, includes all trials
        coverage_percentile : int, default=95
            Percentile level for defining exploration area
        tolerance : float, optional
            Distance threshold for unique point calculation
            If None, uses adaptive tolerance based on data range
        
        Returns
        --------
        dict :
            Dictionary containing exploration metrics:
            - n_unique_points: Number of unique points
            - total_area: Total exploration area at the specified percentile
            - exploration_density: Density of unique points per unit area
            - area_per_point: Area covered per unique point
            - convex_hull_area: Area of the convex hull around the points
            - coverage_efficiency: Fraction of convex hull area covered by exploration area
            - exploration_efficiency: Composite efficiency index combining density and coverage
            - redundancy_ratio: Fraction of redundant points
            - coverage_percentile: Percentile used for area calculation
            - tolerance: Distance threshold used for unique point calculation
        """
        only_trials = None if only_trials is [] else only_trials
        if not self.current_density_grid == only_trials:
            self._create_density_grid(only_trials=only_trials)
            
        # Calculate unique points
        unique_info = self._calculate_unique_points(only_trials=only_trials, tolerance=tolerance)
        n_unique = unique_info['n_unique_points']
        
        # Calculate total exploration area
        areas = self._calculate_contour_areas(only_trials=only_trials, percentiles=[coverage_percentile])
        total_area = areas[f'{coverage_percentile}th percentile']
        
        # Calculate exploration metrics
        exploration_density = n_unique / total_area if total_area > 0 else 0
        area_per_point = total_area / n_unique if n_unique > 0 else float('inf')
        
        data = self._return_right_coords(only_trials)
        # Calculate convex hull area for comparison
        if len(data) >= 3:
            try:
                hull = ConvexHull(data)
                convex_hull_area = hull.volume  # 'volume' is area in 2D
            except:
                convex_hull_area = 0
        else:
            convex_hull_area = 0
        
        # Coverage efficiency (how much area is covered relative to convex hull)
        coverage_efficiency = total_area / convex_hull_area if convex_hull_area > 0 else 0
        
        # Exploration efficiency index (combining density and coverage)
        # Higher values indicate more efficient exploration
        exploration_efficiency = np.sqrt(n_unique * total_area) / len(data) if len(data) > 0 else 0
        
        return {
            'n_unique_points': n_unique,
            'total_area': total_area,
            'exploration_density': exploration_density,  # points per unit area
            'area_per_point': area_per_point,  # area per unique point
            'convex_hull_area': convex_hull_area,
            'coverage_efficiency': coverage_efficiency,  # fraction of convex hull covered
            'exploration_efficiency': exploration_efficiency,  # composite efficiency index
            'redundancy_ratio': unique_info['redundancy_ratio'],  # fraction of redundant points
            'coverage_percentile': coverage_percentile,
            'tolerance': unique_info['tolerance']
        }
    
    def create_density_plot(self, 
        only_trials: list[int] =None,
        output_prefixe: str ="density", 
        output_extension: Literal['pdf', 'png'] ='png',
        fig_size: tuple = (10, 8),
        save: bool =True
    ) -> plt.Axes:
        """
        Visualize the density estimation results as a contour plot
        
        Parameters
        -----------
        only_trials : list[int], optional
            List of trial indices to include in the plot
            If None, includes all trials
        output_prefixe : str, default='density'
            Prefix for the output file name
        output_extension : Literal['pdf', 'png'], default='png'
            File format for saving the plot
        fig_size : tuple, default=(10, 8)
            Size of the figure for the plot
        save : bool, default=True
            Whether to save the plot to a file
        
        Returns
        --------
        plt.Axes :
            The Axes object containing the density plot
        """
        only_trials = None if only_trials is [] else only_trials
        if not self.current_density_grid == only_trials:
            self._create_density_grid(only_trials=only_trials)
            
        fig, ax = plt.subplots(figsize=fig_size)
        all_coords = self.exp_data.get_trials_avg_coords(only_trials)
        
        contour = ax.contourf(self.density_meshgrid_x, self.density_meshgrid_y, self.grid_density_map, levels=20, cmap='viridis', alpha=0.7)
        ax.scatter(all_coords[:,0], all_coords[:,1], c='white', s=10, alpha=0.4)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('KDE Contour Plot')
        spec_trials = self._trials_specifier_for_title(only_trials)
        ax.set_title(f"KDE Contour Plot\nExperiment '{self.exp_data.label}' with {spec_trials}", fontsize=12)
        
        plt.colorbar(contour, ax=ax)
        plt.tight_layout()
        
        if save:
            self._save_visual(fig, output_prefixe, output_extension)
        
        return ax
    
    def _save_visual(self, fig: plt.Figure, output_prefixe: str, output_extension: str):
        """
        Save the created visual to a file.
        
        Parameters
        -----------
        fig : plt.Figure
            The matplotlib figure to save
        output_prefixe : str
            Prefix for the output file name
        output_extension : Literal['pdf', 'png']
            File format for saving the plot
        """
        filename = path.join(self.exp_data.folder_path, f"{output_prefixe}{"_" if self.exp_data.label is not None else ""}{self.exp_data.label}.{output_extension}")
        super()._save_visual(fig, filename, output_extension)
    
    def create_areas_plot(self, 
        only_trials: list[int] =None,
        output_prefixe: str ="areas", 
        output_extension: Literal['pdf', 'png'] ='png',
        fig_size: tuple = (10, 8),
        save: bool =True,
        percentiles: list[int] =[50, 60, 80, 97]
    ) -> plt.Axes:
        """
        Visualize density areas as contour plots for specified percentiles
        
        Parameters
        -----------
        only_trials : list[int], optional
            List of trial indices to include in the plot
            If None, includes all trials
        output_prefixe : str, default='areas'
            Prefix for the output file name
        output_extension : Literal['pdf', 'png'], default='png'
            File format for saving the plot
        fig_size : tuple, default=(10, 8)
            Size of the figure for the plot
        save : bool, default=True
            Whether to save the plot to a file
        percentiles : list[int], default=[50, 60, 80, 97]
            List of percentiles to calculate density levels for
            
        Returns
        --------
        plt.Axes :
            The Axes object containing the contour plot of density areas
        """
        only_trials = None if only_trials is [] else only_trials
        if not self.current_density_grid == only_trials:
            self._create_density_grid(only_trials=only_trials)
        
        fig, ax = plt.subplots(figsize=fig_size)
        all_coords = self.exp_data.get_trials_avg_coords(only_trials)
        levels = self._calculate_levels_from_percentiles(percentiles)
        
        cs = ax.contour(self.density_meshgrid_x, self.density_meshgrid_y, self.grid_density_map, levels=levels, colors=['green', 'yellow', 'orange', 'red'])
        ax.clabel(cs, inline=True, fontsize=8)
        ax.scatter(all_coords[:,0], all_coords[:,1], c='red', s=10, alpha=0.4)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        spec_trials = self._trials_specifier_for_title(only_trials)
        ax.set_title(f"Density areas contour {str(percentiles)}\nExperiment '{self.exp_data.label}' with {spec_trials}", fontsize=12)
        
        plt.tight_layout()
        
        if save:
            self._save_visual(fig, output_prefixe, output_extension)
        
        return ax
    
    def create_exploration_coverage_plot(self, 
        only_trials: list[int] =None,
        coverage_percentile: int =95, 
        tolerance: float =None, 
        fig_size=(20, 5),
        save: bool =True,
        output_prefixe: str ="exploration_coverage",
        output_extension: Literal['pdf', 'png'] ='png'
    ):
        """
        Visualize the exploration coverage of trials as a scatter plot with convex hull
        
        Parameters
        -----------
        only_trials : list[int], optional
            List of trial indices to include in the plot
            If None, includes all trials
        coverage_percentile : int, default=95
            Percentile level for defining exploration area
        tolerance : float, optional
            Distance threshold for unique point calculation
            If None, uses adaptive tolerance based on data range
        fig_size : tuple, default=(20, 5)
            Size of the figure for the plot
        save : bool, default=True
            Whether to save the plot to a file
        output_prefixe : str, default='exploration_coverage'
        output_extension : Literal['pdf', 'png'], default='png'
        
        Returns
        --------
        plt.Axes :
            The Axes object containing the exploration coverage plot
        """
        only_trials = None if only_trials is [] else only_trials
        if not self.current_density_grid == only_trials:
            self._create_density_grid(only_trials=only_trials)
            
        fig, ax = plt.subplots(figsize=fig_size)
        all_coords = all_coords = self.exp_data.get_trials_avg_coords(only_trials)
        total_area = self._calculate_exploration_metrics(only_trials, coverage_percentile, tolerance)["total_area"]
        
        ax.scatter(all_coords[:, 0], all_coords[:, 1], c='red', s=10, alpha=0.4)
        
        # Add convex hull for comparison
        if len(all_coords) >= 3:
            try:
                hull = ConvexHull(all_coords)
                hull_points = all_coords[hull.vertices]
                hull_points = np.vstack([hull_points, hull_points[0]])  # Close the polygon
                ax.plot(hull_points[:, 0], hull_points[:, 1], 'b--', linewidth=2, label='Convex Hull')
            except:
                warn("Convex hull calculation failed, not enough points or collinear points.")
                hull_points = None
        
        spec_trials = self._trials_specifier_for_title(only_trials)
        ax.set_title(f"Exploration Coverage (area: {total_area:.2f})\nExperiement'{self.exp_data.label}' with {spec_trials}", fontsize=12)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()
        plt.tight_layout()
        
        if save:
            self._save_visual(fig, output_prefixe, output_extension)
            
        return ax
    
    def create_metrics_textblock(self,
        only_trials: list[int] =None,
        coverage_percentile: int =95,
        tolerance: float =None,
        fig_size: tuple = (10, 5),
        save: bool =True,
        output_prefixe: str ="metrics_textblock",
        output_extension: Literal['pdf', 'png'] ='png'
    ) -> plt.Axes:
        """
        Create a text block summarizing exploration metrics for the specified trials
        
        Parameters
        -----------
        only_trials : list[int], optional
            List of trial indices to include in the summary
        coverage_percentile : int, default=95
            Percentile level for defining exploration area
        tolerance : float, optional
            Distance threshold for unique point calculation
            If None, uses adaptive tolerance based on data range
        fig_size : tuple, default=(10, 5)
            Size of the figure for the text block
        save : bool, default=True
            Whether to save the text block to a file
        output_prefixe : str, default='metrics_textblock'
        output_extension : Literal['pdf', 'png'], default='png'
        
        Returns
        --------
        plt.Axes :
            The Axes object containing the text block with exploration metrics
        """
        only_trials = None if only_trials is [] else only_trials
        if not self.current_density_grid == only_trials:
            self._create_density_grid(only_trials=only_trials)
        
        fig, ax = plt.subplots(figsize=fig_size)
        total_points = len(self.exp_data.get_trials_avg_coords(only_trials))
        metrics = self._calculate_exploration_metrics(only_trials, coverage_percentile, tolerance)
        
        ax.axis('off')
        summary_text = f"""
    Experiment '{self.exp_data.label}' with {self._trials_specifier_for_title(only_trials)}
    EXPLORATION SUMMARY
    ==================
    
    Data Points:
    • Total: {total_points}
    • Unique: {metrics['n_unique_points']}
    • Redundancy: {metrics['redundancy_ratio']:.1%}
    
    Spatial Coverage:
    • {coverage_percentile}% Area: {metrics['total_area']:.2f}
    • Convex Hull: {metrics['convex_hull_area']:.2f}
    • Coverage Efficiency: {metrics['coverage_efficiency']:.3f}
    
    Exploration Metrics:
    • Density: {metrics['exploration_density']:.3f} pts/area
    • Area per Point: {metrics['area_per_point']:.3f}
    • Efficiency Index: {metrics['exploration_efficiency']:.3f}
    
    Tolerance: {metrics['tolerance']:.4f}
        """
        
        ax.text(
            0.05, 
            0.95, 
            summary_text, 
            transform=ax.transAxes, 
            fontsize=10, 
            verticalalignment='top', 
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8)
        )
        
        plt.tight_layout()
        
        if save:
            self._save_visual(fig, output_prefixe, output_extension)
        
        return ax
    
    def import_exp_data(self, exp_data: WemUmapData):
        """
        Import existing experiment data into the analyzer.
        
        Parameters
        ----------
        exp_data : WemUmapData
            An instance of WemUmapData containing the experiment data to import.
        """
        assert isinstance(exp_data, WemUmapData), "Provided data to import must be an instance of WemUmapData."
        self.exp_data = exp_data

    def export_exp_data(self) -> WemUmapData:
        """
        Export the processed experiment data as a WemUmapData object.
        
        Returns
        -------
        WemUmapData :
            The processed experiment data as a WemUmapData object.
        """
        return self.exp_data

    def convert_to_video(self, targeted_extensions: list[str], remove_input: bool =True) -> None:
        """
        Convert all files with specified extensions in the experiment folder to video format (.mp4).
        
        Parameters
        ----------
        targeted_extensions : list[str]
            A list of file extensions to target for conversion (e.g., ['.gif', '.avi']).
        remove_input : bool, optional
            If True, the input files will be removed after conversion. Default is True.
        """
        for file in listdir(path.abspath(self.exp_data.folder_path)):
            for ext in targeted_extensions:
                if file.endswith(ext):
                    input_path = path.join(self.exp_data.folder_path, file)
                    output_path = path.join(self.exp_data.folder_path, file.replace(ext, '.mp4'))
                    super().convert_to_video(input_path, output_path, remove_input)
    
if __name__ == "__main__":
    
    analyzer = ExpSpatialAnalyzer(
        folder_path="makegraph-en-llama-2",
        lang='en',
        seed=42,
        label='Llama-2',
        top_B=10,
        density_grid_resolution=50
    )
    
    analyzer.create_areas_plot(
        only_trials=[0],
        output_prefixe='areas_plot',
        output_extension='png',
        fig_size=(12, 8),
        save=True,
        percentiles=[50, 60, 80, 97]
    )
    
    analyzer.create_density_plot(
        only_trials=[0],
        output_prefixe='density_plot',
        output_extension='png',
        fig_size=(12, 8),
        save=True
    )
    
    analyzer.create_exploration_coverage_plot(
        only_trials=[0],
        coverage_percentile=95,
        tolerance=None,
        fig_size=(12, 8),
        save=True,
        output_prefixe='exploration_coverage_plot',
        output_extension='png'
    )
    
    analyzer.create_metrics_textblock(
        only_trials=[0],
        coverage_percentile=95,
        tolerance=None,
        fig_size=(12, 8),
        save=True,
        output_prefixe='metrics_textblock',
        output_extension='png'
    )