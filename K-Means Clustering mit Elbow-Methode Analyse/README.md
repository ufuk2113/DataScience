Diese UML-Diagramme dokumentieren vollständig:

    Struktur - Klassenaufbau mit Attributen und Methoden

    Verhalten - Abläufe und Zustandsänderungen

    Interaktionen - Wie Komponenten zusammenarbeiten

    Architektur - Gesamtsystem-Design und Abhängigkeiten

    Datenfluss - Wie Daten durch das System fließen

## UML-Klassendiagramm

```plaintext


+----------------------------------------------------------------------------------------+
|                                  KMeansAnalyzer                                        |
+----------------------------------------------------------------------------------------+
| - random_state: int                                                                    |
| - inertia_values: List[float]                                                          |
| - kmeans_models: Dict[int, KMeans]                                                     |
| - cluster_range: range                                                                 |
| - X: np.ndarray                                                                        |
+----------------------------------------------------------------------------------------+
| + __init__(random_state: int = 42)                                                     |
| + generate_sample_data(n_samples: int = 300, n_features: int = 2,                      |
|                       centers: int = 4, cluster_std: float = 1.0) -> np.ndarray        |
| + run_kmeans_analysis(X: Optional[np.ndarray] = None,                                  |
|                      max_clusters: int = 10) -> List[float]                            |
| + plot_elbow_curve(save_path: Optional[str] = None) -> plt.Figure                      |
| + get_optimal_clusters(method: str = 'elbow') -> int                                   |
| + get_results() -> dict                                                                |
+----------------------------------------------------------------------------------------+

Detaillierte UML-Beschreibung
Klasse: KMeansAnalyzer

Attribute (Private):

    random_state: int - Seed für Reproduzierbarkeit

    inertia_values: List[float] - Gespeicherte Inertia-Werte für jedes k

    kmeans_models: Dict[int, KMeans] - Dictionary der trainierten KMeans-Modelle

    cluster_range: range - Bereich der getesteten Cluster-Anzahlen

    X: np.ndarray - Die analysierten Daten

Methoden:

Konstruktor:

    __init__(random_state: int = 42)

        Initialisiert alle Attribute

        Setzt den Random State für Reproduzierbarkeit

Öffentliche Methoden:

    generate_sample_data(n_samples=300, n_features=2, centers=4, cluster_std=1.0) -> np.ndarray

        Generiert synthetische Testdaten mit make_blobs

        Rückgabe: numpy Array mit Shape (n_samples, n_features)

    run_kmeans_analysis(X=None, max_clusters=10) -> List[float]

        Kernmethode: Führt die For-Schleife für k=1 bis max_clusters aus

        Extrahiert und speichert Inertia-Werte

        Rückgabe: Liste der Inertia-Werte

    plot_elbow_curve(save_path=None) -> plt.Figure

        Erstellt den Elbow-Plot (Inertia vs. Cluster-Anzahl)

        Optional: Speichert den Plot als Bild

        Rückgabe: matplotlib Figure-Objekt

    get_optimal_clusters(method='elbow') -> int

        Berechnet die optimale Cluster-Anzahl basierend auf Inertia-Werten

        Unterstützte Methoden: 'elbow', 'second_derivative'

    get_results() -> dict

        Gibt alle Ergebnisse strukturiert zurück

        Enthält: cluster_range, inertia_values, optimal_k
```

## UML-Sequenzdiagramm

```plaintext

 Client───────┐
              │
KMeansAnalyzer│
              │
    │         │
    │ __init__() │
    │────────>│
    │         │
    │ generate_sample_data() │
    │────────>│
    │         │
    │ run_kmeans_analysis() │
    │────────>│
    │         │
    │   │ for k in 1..10   │
    │   │────>│
    │   │      │
    │   │ KMeans(k) │
    │   │<────│
    │   │      │
    │   │ kmeans.fit(X)   │
    │   │<────│
    │   │      │
    │   │ inertia = kmeans.inertia_ │
    │   │<────│
    │   │      │
    │   │ store results   │
    │   │────>│
    │         │
    │ plot_elbow_curve() │
    │────────>│
    │         │
    │◀───────│
```

## UML-Paketdiagramm

```plaintext
+-------------------+
|   KMeansAnalyzer  |
|      Package      |
+-------------------+
|                   |
| Dependencies:     |
| - numpy           |
| - sklearn.cluster |
| - sklearn.datasets|
| - matplotlib      |
| - logging         |
| - typing          |
|                   |
+-------------------+
         ^
         |
         v
+-------------------+
|   Client Code     |
|   (main.py)       |
+-------------------+
```

## UML-Zustandsdiagramm

```plaintext
+----------------+     +----------------+     +-----------------+     +----------------+
|   Initialized  |     |   Data Ready   |     |   Analysis      |     |   Results      |
|                |     |                |     |   Complete      |     |   Ready        |
| - params set   |---->| - data loaded  |---->| - models trained|---->| - plots ready  |
|                |     |                |     | - inertia saved |     | - optimal k    |
+----------------+     +----------------+     +-----------------+     +----------------+
         ^                      |                      |                      |
         |                      |                      |                      |
         +----------------------+----------------------+----------------------+
                         reset() or new analysis
```

## UML-Use Case Diagramm

```plaintext
+----------------+      +----------------------+      +-------------------+
|   Data         |      |   KMeansAnalyzer     |      |   Visualization   |
|   Scientist    |----->|   System             |----->|   Tools           |
+----------------+      +----------------------+      +-------------------+
        |                       |                             |
        | 1. Load Data          | 2. Analyze Clusters         | 3. Show Elbow Plot
        |                       |                             |
        | 4. Get Optimal k      |                             |
        └───────────────────────┘                             |

```

## Datenfluss-Diagramm

```plaintext

+-------------+     +-----------------+     +----------------+     +---------------+
|   Input     |     |   K-Means       |     |   Inertia      |     |   Elbow       |
|   Data      |---->|   Algorithmus   |---->|   Extraction   |---->|   Plot        |
|   (X)       |     |   (for k=1..10) |     |   & Storage    |     |   Generation  |
+-------------+     +-----------------+     +----------------+     +---------------+
                          ^                         |                       |
                          |                         v                       v
                    +------------+             +-------------+         +-----------+
                    |   K        |             |   Inertia   |         |   Optimal |
                    |   Parameter|             |   Values    |         |   k       |
                    +------------+             +-------------+         +-----------+

```

## Komponentendiagramm

```plaintext

+--------------------------------------------------------------------+
|                      KMeansAnalyzer System                         |
+--------------------------------------------------------------------+
|  +----------------+    +------------------+   +------------------+ |
|  | Data Generator |    | K-Means Executor |   | Result Visualizer| |
|  |                |    |                  |   |                  | |
|  | - make_blobs() |    | - fit()          |   | - plot()         | |
|  | - load_data()  |    | - inertia_       |   | - savefig()      | |
|  +----------------+    +------------------+   +------------------+ |
|                                                                    |
|  +-------------------+   +-------------------+                     |
|  | Optimal k Finder  |   | Results Exporter  |                     |
|  |                   |   |                   |                     |
|  | - elbow_method()  |   | - get_results()   |                     |
|  | - derivative()    |   | - to_dict()       |                     |
|  +-------------------+   +-------------------+                     |
+--------------------------------------------------------------------+

```

## Aktivitätsdiagramm

```plaintext
Start
  │
  ▼
[Initialisiere KMeansAnalyzer]
  │
  ▼
[Daten generieren/laden]
  │
  ▼
┌─────────────────┐
│ for k = 1 to 10 │
└─────────────────┘
  │
  ▼
[KMeans mit k Clustern initialisieren]
  │
  ▼
[Modell auf Daten trainieren]
  │
  ▼
[Inertia-Wert extrahieren und speichern]
  │
  ▼
[Speichere trainiertes Modell]
  │
  ▼
[Alle k durchlaufen?] ──nein──┐
  │ ja                        │
  ▼                           │
[Erstelle Elbow-Plot]         │
  │                           │
  ▼                           │
[Berechne optimale k]         │
  │                           │
  ▼                           │
[Gib Ergebnisse zurück]       │
  │                           │
  ▼                           │
Ende                          │
                              │
                              │
┌─────────────────────────────┘
│
└───> [Nächstes k]
```

## Interface-Realization-Diagramm

```plaintext
+-------------------+       +-------------------------+
|   <<interface>>   |       |    KMeansAnalyzer       |
|   Analyzer        |       |                         |
+-------------------+       +-------------------------+
| + analyze()       |       | + run_kmeans_analysis() |
| + visualize()     |<------| + plot_elbow_curve()    |
| + get_results()   |       | + get_results()         |
+-------------------+       +-------------------------+
```

## Erweiterte Klassendiagramm-Beziehungen

```plaintext
+----------------+       +------------------+       +--------------------+
|   KMeans       |       | KMeansAnalyzer   |       |    matplotlib      |
|   (sklearn)    |<------|                  |------>|     pyplot         |
+----------------+       +------------------+       +--------------------+
| + inertia_     |       | - models         |       | + plot()           |
| + fit()        |       | - results        |       | + show()           |
| + predict()    |       +------------------+       | + savefig()        |
+----------------+                                  +--------------------+
          ^
          |
+-------------------+
|   make_blobs      |
|   (sklearn)       |
+-------------------+
| + __call__()      |
+-------------------+
```
