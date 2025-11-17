from hierarchical_clustering import HierarchicalClustering
import matplotlib.pyplot as plt

def main():
    """
    Hauptfunktion zur Demonstration der HierarchicalClustering Klasse.
    """
    print("=== Hierarchical Clustering Demo ===\n")
    
    # Daten definieren
    x = [4, 6, 9, 4, 3, 11, 12, 6, 10, 12]
    y = [22, 18, 25, 16, 16, 24, 24, 22, 21, 21]
    
    print(f"Datenpunkte: {len(x)} Punkte")
    for i, (xi, yi) in enumerate(zip(x, y)):
        print(f"P{i+1}: ({xi}, {yi})")
    
    print("\n" + "="*50 + "\n")
    
    # 1. Grundlegendes Clustering mit Ward-Methode
    print("1. Grundlegendes Clustering mit Ward-Methode")
    hc = HierarchicalClustering(x, y, method='ward')
    
    # Dendrogramm plotten
    hc.plot_dendrogram(title="Hierarchisches Clustering - Ward Methode", figsize=(12, 7))
    plt.show()
    
    # Cluster-Zuordnungen für 3 Cluster
    clusters = hc.get_clusters(3)
    print("Cluster-Zuordnungen (3 Cluster):")
    for cluster_id, points in clusters.items():
        point_names = [hc.labels[i] for i in points]
        print(f"Cluster {cluster_id}: {points} -> {point_names}")
    
    print("\n" + "="*50 + "\n")
    
    # 2. Mit benutzerdefinierten Labels
    print("2. Clustering mit benutzerdefinierten Labels")
    custom_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    hc_custom = HierarchicalClustering(x, y, labels=custom_labels, method='complete')
    
    hc_custom.plot_dendrogram(title="Mit benutzerdefinierten Labels (Complete Linkage)")
    plt.show()
    
    clusters_custom = hc_custom.get_clusters(4)
    print("Cluster-Zuordnungen (4 Cluster mit custom Labels):")
    for cluster_id, points in clusters_custom.items():
        point_names = [hc_custom.labels[i] for i in points]
        print(f"Cluster {cluster_id}: {point_names}")
    
    print("\n" + "="*50 + "\n")
    
    # 3. Vergleich verschiedener Methoden
    print("3. Vergleich verschiedener Linkage-Methoden")
    methods = ['ward', 'complete', 'average']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for ax, method in zip(axes, methods):
        hc_temp = HierarchicalClustering(x, y, method=method)
        hc_temp.fit()
        
        from scipy.cluster.hierarchy import dendrogram
        dendrogram(hc_temp.get_linkage_matrix(), 
                  labels=hc_temp.labels,
                  leaf_rotation=45,
                  ax=ax)
        ax.set_title(f'{method.capitalize()} Linkage')
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    print("Vergleich abgeschlossen. Dendrogramme für verschiedene Methoden wurden angezeigt.")
    
    print("\n" + "="*50 + "\n")
    
    # 4. Erweiterte Analyse mit verschiedenen Cluster-Anzahlen
    print("4. Cluster-Analyse mit verschiedenen Anzahlen")
    
    hc_analysis = HierarchicalClustering(x, y)
    
    for n_clusters in [2, 3, 4]:
        clusters = hc_analysis.get_clusters(n_clusters)
        print(f"\n--- {n_clusters} Cluster ---")
        for cluster_id, points in clusters.items():
            point_coords = [(x[i], y[i]) for i in points]
            print(f"Cluster {cluster_id}: {points} -> Koordinaten: {point_coords}")

if __name__ == "__main__":
    main()