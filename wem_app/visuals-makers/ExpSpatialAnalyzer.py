import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import warnings
from typing import Any
from short.WemVisualsMaker import ExpVisualsMaker
from WemExpData import WemExpData
from sklearn.cluster import DBSCAN
from WemUmapData import WemUmapData
from os import path, makedirs, listdir
import pandas as pd
warnings.filterwarnings('ignore')

class ExpSpatialAnalyzer(ExpVisualsMaker):
    """
    Quantitative analysis class for 2D spatial data using kernel density estimation
    """
    
    def __init__(self, **kwargs):
        """
        Kwargs Parameters:
        -----------
        bandwidth : float or str, default='scott'
            Bandwidth parameter. 'scott', 'silverman', or numeric value
        kernel : str, default='gaussian'
            Kernel function ('gaussian', 'tophat', 'epanechnikov', etc.)
        board_size : tuple, default=(10, 10)
            Size of the analysis board figure, by default (10, 10).
        folder_path : str
            The path to the folder containing the experiment data. Mendatory if exp_data is not provided.
        umap_data : WemUmapData
            An existing instance of WemUmapData, by default None.
        """
        assert 'folder_path' in kwargs or 'exp_data' in kwargs, "Either provide an experiment folder path or an instance of existing experiment data."
        
        self.umap_data: WemUmapData = kwargs.get('umap_data') or WemUmapData(model=kwargs.get('sentence_transformer_model', 'all-MiniLM-L6-v2'))
        
        self.bandwidth = kwargs.get('bandwidth', 'scott')
        self.kernel = kwargs.get('kernel', 'gaussian')
        self.kde: KernelDensity = None
        self.x_grid: np.ndarray = None
        self.y_grid: np.ndarray = None
        self.X: np.ndarray[np.floating[Any], Any] = None
        self.Y: np.ndarray[np.floating[Any], Any] = None
        self.density: tuple[np.ndarray[Any, Any]] = None
        self.board_figure: tuple[plt.Figure, list[plt.Axes]] = plt.subplots(2, 3, figsize=kwargs.get('board_size', (10, 10)))
        
        self._read_data()
        self._process_data()
        
        self._fit_kde()
        self._create_density_grid()
    
    def _read_data(self):
        if not self.exp_data.is_blank():
            current_data_type = type(self.exp_data.all_data[0][0][0])
            assert current_data_type == str, "The units in the provided experiment data must be string " \
                + f"representing words to create spatial analysis visuals, found {current_data_type} instead.\n" \
                + "NB: dict type is used to create topB animation visuals."
        else:
            csv_idx = 0
            for file in listdir(path.abspath(self.exp_data.folder_path)):
                if file.endswith('csv'):
                    file_path = path.abspath(self.exp_data.folder_path, file)
                    self.exp_data.all_data[csv_idx] = (\
                        pd.read_csv(file_path, usecols=['gen', 'word'], encoding='utf-8')\
                        .groupby('gen')['word'].apply(list).to_dict()
                    )
                    csv_idx += 1
    
    def _process_data(self):
        pass
    
    def _fit_kde(self):
        """Fit KDE model"""
        # Automatic bandwidth selection
        match self.bandwidth:
            case 'scott':
                # Scott's rule: n^(-1/(d+4))
                n, d = self.data.shape
                self.bandwidth = n ** (-1.0 / (d + 4))
                
            case 'silverman':
                # Silverman's rule
                n, d = self.data.shape
                self.bandwidth = (n * (d + 2) / 4.) ** (-1. / (d + 4))
            
        self.kde = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel)
        self.kde.fit(self.data)
        
    def _create_density_grid(self, resolution=100, margin_factor=0.1):
        """
        Create grid for density calculation
        
        Parameters:
        -----------
        resolution : int, default=100
            Number of points along each axis for the grid
        margin_factor : float, default=0.1
            Fraction of data range to add as margin around the grid
        """ 
        # Get data range
        x_min, x_max = self.data[:, 0].min(), self.data[:, 0].max()
        y_min, y_max = self.data[:, 1].min(), self.data[:, 1].max()
        
        # Add margins
        x_range, y_range = x_max - x_min, y_max - y_min
        x_margin, y_margin = x_range * margin_factor, y_range * margin_factor
        
        x_min -= x_margin
        x_max += x_margin
        y_min -= y_margin
        y_max += y_margin
        
        # Create grid
        self.x_grid = np.linspace(x_min, x_max, resolution)
        self.y_grid = np.linspace(y_min, y_max, resolution)
        self.X, self.Y = np.meshgrid(self.x_grid, self.y_grid)
        
        # Calculate density
        grid_points = np.vstack([self.X.ravel(), self.Y.ravel()]).T
        log_density = self.kde.score_samples(grid_points)
        self.density = np.exp(log_density).reshape(self.X.shape)
    
    def calculate_contour_areas(self, levels=None, percentiles=None):
        """
        Calculate areas enclosed by contour lines
        
        Parameters:
        -----------
        levels : array-like, optional
            Absolute density level values
        percentiles : array-like, optional
            Percentiles (e.g., [50, 95] for 50%, 95% confidence intervals)
        
        Returns:
        --------
        dict : 
            Areas for each level
        """
        if levels is None and percentiles is None:
            percentiles = [50, 68, 95, 99]  # Default
            
        if percentiles is not None:
            # Calculate density levels from percentiles
            levels = [np.percentile(self.density[self.density > 0], p) for p in percentiles]
            level_names = [f'{p}th percentile' for p in percentiles]
        else:
            level_names = [f'Level {l:.4f}' for l in levels]
            
        areas = {}
        
        # Grid cell area
        dx = self.x_grid[1] - self.x_grid[0]
        dy = self.y_grid[1] - self.y_grid[0]
        cell_area = dx * dy
        
        for level, name in zip(levels, level_names):
            # Identify regions above specified level
            mask = self.density >= level
            area = np.sum(mask) * cell_area
            areas[name] = area
            
        return areas
    
    def calculate_density_statistics(self) -> dict:
        """
        Calculate statistics of density distribution
        
        Returns
        --------
        dict : Statistics including max, mean, std, range, and entropy
        """
            
        stats_dict = {
            'max_density': np.max(self.density),
            'mean_density': np.mean(self.density),
            'std_density': np.std(self.density),
            'density_range': np.max(self.density) - np.min(self.density),
            'entropy': -np.sum(self.density * np.log(self.density + 1e-10)) * (self.x_grid[1] - self.x_grid[0]) * (self.y_grid[1] - self.y_grid[0])
        }
        
        return stats_dict
    
    def calculate_unique_points(self, tolerance=None):
        """
        Calculate number of unique data points
        
        Parameters:
        -----------
        tolerance : float, optional
            Distance threshold for considering points as duplicates
            If None, uses adaptive tolerance based on data scale
            
        Returns:
        --------
        dict : Information about unique points
        """
        if tolerance is None:
            # Adaptive tolerance: 1% of average data range
            x_range = self.data[:, 0].max() - self.data[:, 0].min()
            y_range = self.data[:, 1].max() - self.data[:, 1].min()
            tolerance = 0.01 * np.mean([x_range, y_range])
        
        # Find unique points using distance-based clustering
        clustering = DBSCAN(eps=tolerance, min_samples=1)
        cluster_labels = clustering.fit_predict(self.data)
        
        unique_points = []
        for cluster_id in np.unique(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            cluster_points = self.data[cluster_mask]
            # Use centroid as representative point
            centroid = np.mean(cluster_points, axis=0)
            unique_points.append({
                'centroid': centroid,
                'point_count': np.sum(cluster_mask),
                'original_indices': np.where(cluster_mask)[0]
            })
        
        return {
            'n_unique_points': len(unique_points),
            'n_total_points': len(self.data),
            'unique_points': unique_points,
            'tolerance': tolerance,
            'redundancy_ratio': 1 - len(unique_points) / len(self.data)
        }
    
    def calculate_exploration_metrics(self, coverage_percentile=95, tolerance=None):
        """
        Calculate comprehensive exploration metrics combining unique points and total area
        
        Parameters:
        -----------
        coverage_percentile : float, default=95
            Percentile level for defining total exploration area
        tolerance : float, optional
            Distance threshold for unique point calculation
            
        Returns:
        --------
        dict : Comprehensive exploration metrics
        """
        # Calculate unique points
        unique_info = self.calculate_unique_points(tolerance=tolerance)
        n_unique = unique_info['n_unique_points']
        
        # Calculate total exploration area
        areas = self.calculate_contour_areas(percentiles=[coverage_percentile])
        total_area = areas[f'{coverage_percentile}th percentile']
        
        # Calculate exploration metrics
        exploration_density = n_unique / total_area if total_area > 0 else 0
        area_per_point = total_area / n_unique if n_unique > 0 else float('inf')
        
        # Calculate convex hull area for comparison
        from scipy.spatial import ConvexHull
        if len(self.data) >= 3:
            try:
                hull = ConvexHull(self.data)
                convex_hull_area = hull.volume  # 'volume' is area in 2D
            except:
                convex_hull_area = 0
        else:
            convex_hull_area = 0
        
        # Coverage efficiency (how much area is covered relative to convex hull)
        coverage_efficiency = total_area / convex_hull_area if convex_hull_area > 0 else 0
        
        # Exploration efficiency index (combining density and coverage)
        # Higher values indicate more efficient exploration
        exploration_efficiency = np.sqrt(n_unique * total_area) / len(self.data) if len(self.data) > 0 else 0
        
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
    
    def create_analysis_board(self):
        """
        Create a board of everal visuals for a comprehensive spatial analysis
        
        """
        pass
    
    def create_contour_plot(self):
        """
        Create a contour plot of the density distribution
        """
        im1 = self.board_figure[1][0].contourf(self.X, self.Y, self.density, levels=20, cmap='viridis', alpha=0.7)
        self.board_figure[1][0].scatter(self.data[:, 0], self.data[:, 1], c='red', s=10, alpha=0.6)
        self.board_figure[1][0].set_title('Kernel Density Estimation Contour Plot')
        self.board_figure[1][0].set_xlabel('x')
        self.board_figure[1][0].set_ylabel('y')
        
        plt.colorbar(im1, ax=self.board_figure[1][0])
        
    def plot_analysis(self, figsize=(15, 5)):
        """
        Visualize analysis results
        
        Parameters:
        -----------
        figsize : tuple, default=(15, 5)
            Figure size for the plot
        """
        # 1. Density distribution and data points
        im1 = self.board_figure[1][0].contourf(self.X, self.Y, self.density, levels=20, cmap='viridis', alpha=0.7)
        self.board_figure[1][0].scatter(self.data[:, 0], self.data[:, 1], c='red', s=10, alpha=0.6)
        self.board_figure[1][0].set_title('Kernel Density Estimation')
        self.board_figure[1][0].set_xlabel('X')
        self.board_figure[1][0].set_ylabel('Y')
        plt.colorbar(im1, ax=self.board_figure[1][0])
        
        # 2. Contour lines and areas
        levels = [np.percentile(self.density[self.density > 0], p) for p in [50, 68, 95]]
        cs = self.board_figure[1][1].contour(self.X, self.Y, self.density, levels=levels, colors=['blue', 'green', 'red'])
        self.board_figure[1][1].clabel(cs, inline=True, fontsize=8)
        self.board_figure[1][1].scatter(self.data[:, 0], self.data[:, 1], c='black', s=10, alpha=0.6)
        self.board_figure[1][1].set_title('Density Contours (50%, 68%, 95%)')
        self.board_figure[1][1].set_xlabel('X')
        self.board_figure[1][1].set_ylabel('Y')
        
        plt.tight_layout()
        return fig
    
    def plot_exploration_analysis(self, coverage_percentile=95, tolerance=None, figsize=(20, 5)):
        """
        Visualize exploration metrics analysis
        
        Parameters:
        -----------
        coverage_percentile : float, default=95
            Percentile level for defining exploration area
        tolerance : float, optional
            Distance threshold for unique point calculation
        figsize : tuple, default=(20, 5)
            Figure size
        """
        # Calculate exploration metrics
        metrics = self.calculate_exploration_metrics(coverage_percentile, tolerance)
        unique_info = self.calculate_unique_points(tolerance)
        
        if not hasattr(self, 'density'):
            self._create_density_grid()
        
        fig, axes = plt.subplots(1, 4, figsize=figsize)
        
        # 1. Original data points vs unique points
        axes[0].scatter(self.data[:, 0], self.data[:, 1], c='lightblue', s=15, alpha=0.6, label='All points')
        
        # Plot unique points
        for i, unique_point in enumerate(unique_info['unique_points']):
            centroid = unique_point['centroid']
            count = unique_point['point_count']
            size = 20 + count * 5  # Size proportional to point count
            axes[0].scatter(centroid[0], centroid[1], c='red', s=size, alpha=0.8, edgecolors='black', linewidth=1)
            if count > 1:
                axes[0].text(centroid[0], centroid[1], str(count), fontsize=8, ha='center', va='center', color='white', weight='bold')
        
        axes[0].set_title(f'Unique Points Analysis\n({metrics["n_unique_points"]} unique / {len(self.data)} total)')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        axes[0].legend()
        
        # 2. Exploration area coverage
        # Plot density contour for the specified percentile
        threshold = np.percentile(self.density[self.density > 0], coverage_percentile)
        mask = self.density >= threshold
        axes[1].contourf(self.X, self.Y, mask.astype(int), levels=[0.5, 1.5], colors=['lightgreen'], alpha=0.7, label=f'{coverage_percentile}% area')
        axes[1].scatter(self.data[:, 0], self.data[:, 1], c='red', s=10, alpha=0.6)
        
        # Add convex hull for comparison
        from scipy.spatial import ConvexHull
        if len(self.data) >= 3:
            try:
                hull = ConvexHull(self.data)
                hull_points = self.data[hull.vertices]
                hull_points = np.vstack([hull_points, hull_points[0]])  # Close the polygon
                axes[1].plot(hull_points[:, 0], hull_points[:, 1], 'b--', linewidth=2, label='Convex Hull')
            except:
                pass
        
        axes[1].set_title(f'Exploration Coverage\nArea: {metrics["total_area"]:.2f}')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
        axes[1].legend()
        
        # 4. Summary metrics text
        axes[3].axis('off')
        summary_text = f"""
        EXPLORATION SUMMARY
        ==================
        
        Data Points:
        • Total: {len(self.data)}
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
        
        axes[3].text(0.05, 0.95, summary_text, transform=axes[3].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        return fig, metrics

# 使用例
def example_analysis():
    """使用例の実行"""
    # サンプルデータの生成（2つのクラスター）
    np.random.seed(42)
    cluster1 = np.random.multivariate_normal([2, 2], [[0.5, 0.1], [0.1, 0.3]], 150)
    cluster2 = np.random.multivariate_normal([6, 5], [[0.8, -0.2], [-0.2, 0.6]], 100)
    noise = np.random.uniform([0, 0], [8, 7], (50, 2))
    data = np.vstack([cluster1, cluster2, noise])
    
    # 分析実行
    analyzer = ExpSpatialAnalyzer(data, bandwidth='scott')
    
    print("=== カーネル密度推定による空間分析 ===")
    print(f"データ点数: {len(data)}")
    if isinstance(analyzer.bandwidth, (int, float)):
        print(f"使用バンド幅: {analyzer.bandwidth:.4f}")
    else:
        print(f"使用バンド幅: {analyzer.bandwidth}")
    
    # 面積計算
    areas = analyzer.calculate_contour_areas(percentiles=[50, 68, 95])
    print("\n--- 等高線面積 ---")
    for level, area in areas.items():
        print(f"{level}: {area:.2f}")
    
    # 密度統計
    stats = analyzer.calculate_density_statistics()
    print("\n--- 密度統計 ---")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")
    
    # 可視化
    fig = analyzer.plot_analysis()
    plt.show()
    
    return analyzer

if __name__ == "__main__":
    analyzer = example_analysis()
    analyzer.plot_exploration_analysis()