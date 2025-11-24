import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from typing import List, Tuple, Optional
import logging

# ✅ NEU: Für automatische Erkennung des "Elbow"-Punkts
from kneed import KneeLocator

# Logger konfigurieren
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KMeansAnalyzer:
    """
    Eine Klasse zur Analyse des K-Means-Algorithmus für verschiedene Cluster-Anzahlen.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.inertia_values = []
        self.kmeans_models = {}
        self.cluster_range = None
        logger.info(f"KMeansAnalyzer mit random_state={random_state} initialisiert")
    
    def generate_sample_data(self, n_samples: int = 300, n_features: int = 2, 
                           centers: int = 4, cluster_std: float = 1.0) -> np.ndarray:
        X, y_true = make_blobs(
            n_samples=n_samples, 
            n_features=n_features, 
            centers=centers, 
            cluster_std=cluster_std,
            random_state=self.random_state
        )
        self.X = X
        logger.info(f"Daten erfolgreich generiert: Shape {X.shape}")
        return X
    
    def run_kmeans_analysis(self, X: Optional[np.ndarray] = None, 
                          max_clusters: int = 10) -> List[float]:
        if X is None:
            X = self.generate_sample_data()
        else:
            self.X = X
        
        self.cluster_range = range(1, max_clusters + 1)
        self.inertia_values = []
        self.kmeans_models = {}

        for k in self.cluster_range:
            kmeans = KMeans(
                n_clusters=k,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=self.random_state
            )
            kmeans.fit(X)
            inertia = kmeans.inertia_
            self.inertia_values.append(inertia)
            self.kmeans_models[k] = kmeans
            logger.info(f"K-Means k={k} abgeschlossen - Inertia: {inertia:.2f}")

        return self.inertia_values
    
    def plot_elbow_curve(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Erstellt einen Elbow-Plot der Inertia-Werte gegen die Anzahl der Cluster
        und markiert automatisch den gefundenen Elbow-Punkt (roter Strich).
        """
        if not self.inertia_values:
            raise ValueError("Keine Inertia-Werte verfügbar. Bitte zuerst run_kmeans_analysis() ausführen.")

        # ✅ KneeLocator zur Bestimmung des Knickpunkts
        knee_locator = KneeLocator(
            x=list(self.cluster_range),
            y=self.inertia_values,
            curve='convex',
            direction='decreasing'
        )
        optimal_k = knee_locator.knee

        # 🔹 Plot vorbereiten
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.cluster_range, self.inertia_values, 'bo-', linewidth=2, markersize=8, label="Inertia")

        # 🔴 Vertikale Linie beim „Elbow“-Punkt einzeichnen
        if optimal_k is not None:
            ax.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, label=f'Elbow bei k={optimal_k}')
            ax.scatter(optimal_k, self.inertia_values[optimal_k - 1], color='red', s=100, zorder=5)  # Punkt markieren

        # 🔹 Achsenbeschriftung & Titel
        ax.set_xlabel('Anzahl der Cluster', fontsize=12)
        ax.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
        ax.set_title('Elbow-Methode für optimale Cluster-Anzahl', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(list(self.cluster_range))
        ax.legend()

        plt.tight_layout()

        # ✅ Plot speichern (optional)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot gespeichert unter: {save_path}")

        plt.show()
        logger.info(f"Elbow-Plot erstellt (Elbow bei k={optimal_k})")

        return fig

    
    # ==============================================================
    # 🔴 ALTER CODE (manuelle Elbow-Methode) – deaktiviert, aber belassen
    # ==============================================================
    # def get_optimal_clusters(self, method: str = 'elbow') -> int:
    #     if not self.inertia_values:
    #         raise ValueError("Keine Inertia-Werte verfügbar")
    #     if method == 'elbow':
    #         differences = []
    #         for i in range(1, len(self.inertia_values)):
    #             improvement = self.inertia_values[i-1] - self.inertia_values[i]
    #             differences.append(improvement)
    #         optimal_k_index = np.argmax(differences) + 1
    #         optimal_k = optimal_k_index + 1
    #     elif method == 'second_derivative':
    #         first_deriv = np.diff(self.inertia_values)
    #         second_deriv = np.diff(first_deriv)
    #         optimal_k = np.argmax(second_deriv) + 2
    #     else:
    #         raise ValueError(f"Unbekannte Methode: {method}")
    #     logger.info(f"Optimale Cluster-Anzahl geschätzt (manuell): k={optimal_k}")
    #     return optimal_k

    # ==============================================================
    # 🟢 NEUER CODE mit KneeLocator – Automatische Elbow-Erkennung
    # ==============================================================
    def get_optimal_clusters(self) -> int:
        """
        Bestimmt die optimale Anzahl der Cluster automatisch mit dem KneeLocator.
        """
        if not self.inertia_values:
            raise ValueError("Keine Inertia-Werte verfügbar – bitte zuerst run_kmeans_analysis() ausführen.")

        # KneeLocator sucht den Knickpunkt („Knee“) in der Inertia-Kurve
        # curve='convex' und direction='decreasing' sind typisch für Inertia-Werte:
        # Je mehr Cluster → desto kleiner die Inertia → abnehmende Kurve
        knee_locator = KneeLocator(
            x=list(self.cluster_range),
            y=self.inertia_values,
            curve='convex',
            direction='decreasing'
        )

        # knee_locator.knee gibt die geschätzte optimale Cluster-Anzahl zurück
        optimal_k = knee_locator.knee

        logger.info(f"Automatisch erkannter Elbow-Punkt (KneeLocator): k={optimal_k}")
        return optimal_k
    
    def get_results(self) -> dict:
        if not self.inertia_values:
            raise ValueError("Keine Ergebnisse verfügbar")

        optimal_k = self.get_optimal_clusters()
        results = {
            'cluster_range': list(self.cluster_range),
            'inertia_values': self.inertia_values,
            'optimal_k': optimal_k
        }
        logger.info("Ergebnisse erfolgreich zurückgegeben")
        return results

def main():
    """
    Hauptfunktion zur Demonstration der KMeansAnalyzer-Klasse.
    """
    logger.info("Starte K-Means Analyse Demo")
    
    # 1. KMeansAnalyzer Instanz erstellen
    analyzer = KMeansAnalyzer(random_state=42)
    
    # 2. Beispieldaten generieren
    X = analyzer.generate_sample_data(n_samples=300, centers=4, cluster_std=0.8)
    
    # 3. K-Means Analyse für 1-10 Cluster durchführen
    inertia_values = analyzer.run_kmeans_analysis(X=X, max_clusters=10)
    
    # 4. Ergebnisse ausgeben
    results = analyzer.get_results()
    print("\n" + "="*50)
    print("K-MEANS ANALYSE ERGEBNISSE")
    print("="*50)
    for k, inertia in zip(results['cluster_range'], results['inertia_values']):
        print(f"k={k}: Inertia = {inertia:.2f}")
    
    print(f"\nGeschätzte optimale Cluster-Anzahl: k={results['optimal_k']}")
    
    # 5. Elbow-Plot erstellen und anzeigen
    analyzer.plot_elbow_curve(save_path='elbow_plot.png')
    
    logger.info("K-Means Analyse Demo abgeschlossen")


# Code nur ausführen wenn die Datei direkt ausgeführt wird
if __name__ == "__main__":
    main()