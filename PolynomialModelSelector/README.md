# Diese UML-Diagramme zeigen:

```plaintext

    Klassendiagramm - Struktur der Klasse mit Attributen und Methoden

    Sequenzdiagramm - Ablauf der Kreuzvalidierung

    Paketdiagramm - Abhängigkeiten zu externen Bibliotheken

    Zustandsdiagramm - Lebenszyklus der Klasse

    Use Case Diagramm - Interaktion mit dem Benutzer

Die Klasse folgt den Prinzipien der Objektorientierung:

    Encapsulation: Private Attribute, öffentliche Methoden

    Abstraction: Komplexe Kreuzvalidierung hinter einfacher Schnittstelle

    Modularity: Klare Trennung der Verantwortlichkeiten

```

###

# UML-Klassendiagramm

```plaintext
+-----------------------------------------------------------------------+
| PolynomialCV                                                          |
+-----------------------------------------------------------------------+
| - k_folds: int                                                        |
| - degrees: List[int]                                                  |
| - random_state: int                                                   |
| - results: Dict                                                       |
| - best_degree: int                                                    |
| - best_score: float                                                   |
+-----------------------------------------------------------------------+
| + **init**(k_folds: int = 5,                                          |
| degrees: List[int] = None,                                            |
| random_state: int = 42)                                               |
| + cross_validate(X: np.ndarray, y: np.ndarray) -> Dict                |
| - \_train_model(X_train: np.ndarray, y_train: np.ndarray,             |
| X_val: np.ndarray, degree: int)                                       |
| -> Tuple[np.ndarray, Pipeline]                                        |
| - \_evaluate_model(y_true: np.ndarray, y_pred: np.ndarray)            |
| -> Tuple[float, float]                                                |
| + get_best_degree() -> Tuple[int, float]                              |
| + plot_results(save_path: str = None) -> None                         |
| + get_results_dataframe() -> pd.DataFrame                             |
+-----------------------------------------------------------------------+
```

# Detaillierte UML-Beschreibung

Klasse: PolynomialModelSelector

```plaintext
Attribute (Private):

    k_folds: int - Anzahl der Folds für Kreuzvalidierung

    degrees: List[int] - Liste der zu testenden Polynomgrade

    random_state: int - Seed für Reproduzierbarkeit

    results: Dict - Speichert alle Validierungsergebnisse

    best_degree: int - Bester gefundener Polynomgrad

    best_score: float - Bester Score (niedrigster MSE)

Methoden:

Öffentliche Methoden:

    __init__(k_folds: int = 5, degrees: List[int] = None, random_state: int = 42)

        Konstruktor zur Initialisierung der Parameter

    cross_validate(X: np.ndarray, y: np.ndarray) -> Dict

        Führt die k-fache Kreuzvalidierung durch

        Rückgabe: Dictionary mit Ergebnissen

    get_best_degree() -> Tuple[int, float]

        Gibt den besten Polynomgrad und Score zurück

    plot_results(save_path: str = None) -> None

        Visualisiert die Validierungsergebnisse

    get_results_dataframe() -> pd.DataFrame

        Gibt Ergebnisse als DataFrame zurück

Private Methoden:

    _train_model(X_train, y_train, X_val, degree) -> Tuple[np.ndarray, Pipeline]

        Trainiert das Polynommodell

    _evaluate_model(y_true, y_pred) -> Tuple[float, float]

        Berechnet MSE und R² Metriken
```

# UML-Sequenzdiagramm

```plaintext
Client ────────┐
               │
PolynomialCV   │
               │
    │          │
    │ __init__() │
    │─────────>│
    │          │
    │ cross_validate(X, y) │
    │─────────>│
    │          │
    │   │ _create_folds()  │
    │   │<─────│
    │   │      │
    │   │ for each degree │
    │   │─────>│
    │   │      │
    │   │ for each fold   │
    │   │─────>│
    │   │      │
    │   │ _train_model()  │
    │   │<─────│
    │   │      │
    │   │ _evaluate_model() │
    │   │<─────│
    │   │      │
    │   │ store results   │
    │   │─────>│
    │          │
    │◀─────────│
    │          │

```

# UML-Paketdiagramm

```plaintext

+------------------------------+
|   PolynomialModelSelector    |
|   Package                    |
+------------------------------+
|                              |
| Dependencies:                |
| - numpy                      |
| - sklearn                    |
| - matplotlib                 |
| - pandas                     |
|                              |
+------------------------------+

```

# Datenstruktur des Results-Dictionary

```plaintext

results: Dict
├── degrees: List[int]
├── mean_mse: List[float]
├── std_mse: List[float]
├── mean_r2: List[float]
├── std_r2: List[float]
└── fold_scores: Dict
    ├── degree_1: Dict
    │   ├── mse: List[float]
    │   └── r2: List[float]
    ├── degree_2: Dict
    │   ├── mse: List[float]
    │   └── r2: List[float]
    └── ...
```

# UML-Zustandsdiagramm

```plaintext

+-------------------+     +------------------+     +-------------------+
|   Initialized     |     |   Validating     |     |   Completed       |
|                   |     |                  |     |                   |
| - parameters set  |---->| - cross_validate |---->| - results ready   |
|                   |     |   running        |     | - best_degree set |
+-------------------+     +------------------+     +-------------------+
         ^                                              |
         |                                              |
         +----------------------------------------------+
                  reset() or new cross_validate()

```

# UML-Use Case Diagramm

```plaintext
+----------------+      +-----------------------------+      +-------------------+
|   Data         |      |   PolynomialModelSelector   |      |   Results         |
|   Scientist    |----->|   System                    |----->|   Visualization   |
+----------------+      +-----------------------------+      +-------------------+
        |                       |                             |
        | 1. Configure          | 2. Perform CV               | 3. Display
        |    parameters         |    for multiple             |    results
        |                       |    degrees                  |
        | 4. Get best model     |                             |
        └───────────────────────┘                             |
```
