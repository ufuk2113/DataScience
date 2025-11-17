import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from typing import List, Dict, Optional
import warnings

class HierarchicalClustering:
    """
    Eine Klasse für hierarchisches Clustering mit Dendrogramm-Visualisierung.
    
    Attributes:
        data (np.ndarray): Die zu clusternden Daten
        linkage_matrix (np.ndarray): Die Linkage-Matrix nach dem Fit
        labels (List[str]): Beschriftungen der Datenpunkte
        method (str): Linkage-Methode
        metric (str): Distanzmetrik
    """
    
    def __init__(self, x: List[float], y: List[float], 
                 labels: Optional[List[str]] = None,
                 method: str = 'ward', 
                 metric: str = 'euclidean'):
        """
        Initialisiert den HierarchicalClustering.
        
        Args:
            x: x-Koordinaten der Datenpunkte
            y: y-Koordinaten der Datenpunkte
            labels: Beschriftungen für die Datenpunkte
            method: Linkage-Methode ('ward', 'complete', 'average', 'single')
            metric: Distanzmetrik ('euclidean', 'cityblock', 'cosine', etc.)
        """
        self.data = np.array(list(zip(x, y)))
        self.method = method
        self.metric = metric
        self.linkage_matrix = None
        
        # Labels generieren falls nicht provided
        if labels is None:
            self.labels = [f"P{i+1}({x[i]},{y[i]})" for i in range(len(x))]
        else:
            if len(labels) != len(x):
                raise ValueError("Anzahl der Labels muss der Anzahl der Datenpunkte entsprechen")
            self.labels = labels
    
    def fit(self) -> None:
        """
        Führt die hierarchische Clusterbildung durch.
        """
        try:
            self.linkage_matrix = linkage(self.data, 
                                        method=self.method, 
                                        metric=self.metric)
        except Exception as e:
            raise ValueError(f"Fehler bei der Clusterbildung: {e}")
    
    def plot_dendrogram(self, title: Optional[str] = None, 
                       figsize: tuple = (10, 6),
                       color_threshold: float = 0,
                       show_grid: bool = True) -> plt.Figure:
        """
        Erstellt und zeigt das Dendrogramm.
        
        Args:
            title: Titel des Plots
            figsize: Größe der Figure
            color_threshold: Schwellenwert für Cluster-Farben
            show_grid: Zeigt Gitterlinien an
            
        Returns:
            plt.Figure: Das Figure-Objekt
        """
        if self.linkage_matrix is None:
            self.fit()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Dendrogramm erstellen
        dendrogram(self.linkage_matrix,
                  labels=self.labels,
                  leaf_rotation=45,
                  leaf_font_size=10,
                  color_threshold=color_threshold,
                  ax=ax)
        
        # Plot anpassen
        if title is None:
            title = f'Hierarchisches Clustering ({self.method} Linkage)'
        ax.set_title(title)
        ax.set_xlabel('Datenpunkte')
        ax.set_ylabel('Distanz')
        
        if show_grid:
            ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def get_clusters(self, n_clusters: int) -> Dict[int, List[int]]:
        """
        Gibt die Cluster-Zuordnungen für eine gegebene Anzahl von Clustern zurück.
        
        Args:
            n_clusters: Anzahl der gewünschten Cluster
            
        Returns:
            Dict mit Cluster-ID als Key und Indizes der Punkte als Value
        """
        if self.linkage_matrix is None:
            self.fit()
        
        cluster_labels = fcluster(self.linkage_matrix, n_clusters, criterion='maxclust')
        
        clusters = {}
        for i, cluster_id in enumerate(cluster_labels):
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(i)
        
        return clusters
    
    def set_method(self, method: str) -> None:
        """
        Setzt die Linkage-Methode.
        
        Args:
            method: Linkage-Methode
        """
        valid_methods = ['ward', 'complete', 'average', 'single', 'weighted', 'centroid', 'median']
        if method not in valid_methods:
            raise ValueError(f"Ungültige Methode. Erlaubt: {valid_methods}")
        self.method = method
        self.linkage_matrix = None  # Reset für neues Fitting
    
    def set_metric(self, metric: str) -> None:
        """
        Setzt die Distanzmetrik.
        
        Args:
            metric: Distanzmetrik
        """
        valid_metrics = ['euclidean', 'cityblock', 'cosine', 'correlation']
        if metric not in valid_metrics:
            warnings.warn(f"Metrik {metric} könnte Probleme verursachen. Empfohlen: {valid_metrics}")
        self.metric = metric
        self.linkage_matrix = None  # Reset für neues Fitting
    
    def get_linkage_matrix(self) -> np.ndarray:
        """
        Gibt die Linkage-Matrix zurück.
        
        Returns:
            np.ndarray: Die Linkage-Matrix
        """
        if self.linkage_matrix is None:
            self.fit()
        return self.linkage_matrix.copy()
    
    def __str__(self) -> str:
        """String-Repräsentation der Klasse."""
        return (f"HierarchicalClustering(n_points={len(self.data)}, "
                f"method='{self.method}', metric='{self.metric}')")