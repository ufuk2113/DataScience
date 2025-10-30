import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from typing import List, Tuple, Optional
import logging

# Logger konfigurieren
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KMeansAnalyzer:
    """
    Eine Klasse zur Analyse des K-Means-Algorithmus für verschiedene Cluster-Anzahlen.
    
    Diese Klasse führt K-Means für verschiedene Cluster-Anzahlen durch,
    sammelt die Inertia-Werte und visualisiert das Ergebnis im Elbow-Plot.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialisiert den KMeansAnalyzer.
        
        Args:
            random_state (int): Seed für Reproduzierbarkeit der Ergebnisse
        """
        self.random_state = random_state
        self.inertia_values = []  # Liste zur Speicherung der Inertia-Werte
        self.kmeans_models = {}   # Dictionary zur Speicherung der trainierten Modelle
        self.cluster_range = None # Bereich der getesteten Cluster-Anzahlen
        logger.info(f"KMeansAnalyzer mit random_state={random_state} initialisiert")
    
    def generate_sample_data(self, n_samples: int = 300, n_features: int = 2, 
                           centers: int = 4, cluster_std: float = 1.0) -> np.ndarray:
        """
        Generiert synthetische Daten für das Clustering.
        
        Args:
            n_samples (int): Anzahl der Datenpunkte
            n_features (int): Anzahl der Features pro Datenpunkt
            centers (int): Anzahl der echten Cluster in den Daten
            cluster_std (float): Standardabweichung der Cluster
            
        Returns:
            np.ndarray: Generierte Daten mit Shape (n_samples, n_features)
        """
        logger.info(f"Generiere Beispieldaten: {n_samples} Samples, {n_features} Features, {centers} Cluster")
        
        # Synthetische Daten mit klar getrennten Clustern erzeugen
        X, y_true = make_blobs(
            n_samples=n_samples, 
            n_features=n_features, 
            centers=centers, 
            cluster_std=cluster_std,
            random_state=self.random_state
        )
        
        self.X = X  # Daten als Instanzvariable speichern
        logger.info(f"Daten erfolgreich generiert: Shape {X.shape}")
        return X
    
    def run_kmeans_analysis(self, X: Optional[np.ndarray] = None, 
                          max_clusters: int = 10) -> List[float]:
        """
        Führt K-Means für Cluster-Anzahlen von 1 bis max_clusters aus.
        
        Args:
            X (np.ndarray, optional): Die zu clusternden Daten. Wenn None, werden Beispieldaten generiert.
            max_clusters (int): Maximale Anzahl der zu testenden Cluster
            
        Returns:
            List[float]: Liste der Inertia-Werte für jede Cluster-Anzahl
        """
        # Daten vorbereiten
        if X is None:
            logger.info("Keine Daten provided, generiere Beispieldaten")
            X = self.generate_sample_data()
        else:
            self.X = X
            logger.info(f"Verwende provided Daten: Shape {X.shape}")
        
        # Bereich der Cluster-Anzahlen definieren
        self.cluster_range = range(1, max_clusters + 1)
        logger.info(f"Führe K-Means Analyse durch für Cluster-Anzahlen: 1 bis {max_clusters}")
        
        # Liste zurücksetzen
        self.inertia_values = []
        self.kmeans_models = {}
        
        # For-Schleife über alle Cluster-Anzahlen
        for k in self.cluster_range:
            logger.info(f"Berechne K-Means mit k={k} Clustern")
            
            # KMeans-Modell initialisieren
            # n_init=10: Anzahl der Initialisierungen mit unterschiedlichen Zentroiden
            # random_state: Für reproduzierbare Ergebnisse
            kmeans = KMeans(
                n_clusters=k,           # Anzahl der Cluster
                init='k-means++',       # Intelligente Initialisierung der Zentroide
                n_init=10,              # Anzahl der Initialisierungsversuche
                max_iter=300,           # Maximale Iterationen pro Initialisierung
                random_state=self.random_state  # Reproduzierbarkeit
            )
            
            # K-Means auf den Daten trainieren
            kmeans.fit(X)
            
            # Inertia-Wert aus dem Modell extrahieren
            # Inertia = Summe der quadrierten Distanzen der Punkte zu ihrem nächsten Cluster-Zentrum
            inertia = kmeans.inertia_
            
            # Inertia-Wert in der Liste speichern
            self.inertia_values.append(inertia)
            
            # Modell für spätere Verwendung speichern
            self.kmeans_models[k] = kmeans
            
            logger.info(f"K-Means k={k} abgeschlossen - Inertia: {inertia:.2f}")
        
        logger.info(f"K-Means Analyse abgeschlossen. Gesammelte Inertia-Werte: {self.inertia_values}")
        return self.inertia_values
    
    def plot_elbow_curve(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Erstellt einen Elbow-Plot der Inertia-Werte gegen die Anzahl der Cluster.
        
        Args:
            save_path (str, optional): Pfad zum Speichern des Plots
            
        Returns:
            plt.Figure: Das Figure-Objekt des Plots
        """
        if not self.inertia_values:
            logger.error("Keine Inertia-Werte verfügbar. Führen Sie zuerst run_kmeans_analysis() aus.")
            raise ValueError("Keine Inertia-Werte verfügbar. Führen Sie zuerst run_kmeans_analysis() aus.")
        
        logger.info("Erstelle Elbow-Plot")
        
        # Plot erstellen
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Inertia-Werte gegen Cluster-Anzahl plotten
        ax.plot(self.cluster_range, self.inertia_values, 'bo-', linewidth=2, markersize=8)
        
        # Plot beschriften
        ax.set_xlabel('Anzahl der Cluster', fontsize=12)
        ax.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
        ax.set_title('Elbow-Methode für optimale Cluster-Anzahl', fontsize=14, fontweight='bold')
        
        # Grid für bessere Lesbarkeit
        ax.grid(True, alpha=0.3)
        
        # X-Achse auf ganze Zahlen setzen
        ax.set_xticks(list(self.cluster_range))
        
        # Plot optimieren
        plt.tight_layout()
        
        # Plot speichern falls gewünscht
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot gespeichert unter: {save_path}")
        
        # Plot anzeigen
        plt.show()
        logger.info("Elbow-Plot erfolgreich erstellt und angezeigt")
        
        return fig
    
    def get_optimal_clusters(self, method: str = 'elbow') -> int:
        """
        Schätzt die optimale Anzahl der Cluster basierend auf dem Elbow-Point.
        
        Args:
            method (str): Methode zur Bestimmung des optimalen k ('elbow' oder 'second_derivative')
            
        Returns:
            int: Geschätzte optimale Anzahl der Cluster
        """
        if not self.inertia_values:
            logger.error("Keine Inertia-Werte verfügbar")
            raise ValueError("Keine Inertia-Werte verfügbar")
        
        if method == 'elbow':
            # Einfache Elbow-Methode: Punkt mit größtem Abknick
            differences = []
            for i in range(1, len(self.inertia_values)):
                # Relative Verbesserung berechnen
                improvement = self.inertia_values[i-1] - self.inertia_values[i]
                differences.append(improvement)
            
            # Den Punkt mit der größten Verbesserung finden
            optimal_k_index = np.argmax(differences) + 1  # +1 weil wir bei k=2 starten
            optimal_k = optimal_k_index + 1  # +1 weil Index bei 0 beginnt
            
        elif method == 'second_derivative':
            # Methode basierend auf der zweiten Ableitung
            first_deriv = np.diff(self.inertia_values)
            second_deriv = np.diff(first_deriv)
            optimal_k = np.argmax(second_deriv) + 2  # +2 wegen zwei Ableitungen
            
        else:
            logger.error(f"Unbekannte Methode: {method}")
            raise ValueError(f"Unbekannte Methode: {method}")
        
        logger.info(f"Optimale Cluster-Anzahl geschätzt: k={optimal_k}")
        return optimal_k
    
    def get_results(self) -> dict:
        """
        Gibt die Ergebnisse der Analyse zurück.
        
        Returns:
            dict: Dictionary mit Cluster-Anzahlen und zugehörigen Inertia-Werten
        """
        if not self.inertia_values:
            logger.error("Keine Ergebnisse verfügbar")
            raise ValueError("Keine Ergebnisse verfügbar")
        
        results = {
            'cluster_range': list(self.cluster_range),
            'inertia_values': self.inertia_values,
            'optimal_k': self.get_optimal_clusters()
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