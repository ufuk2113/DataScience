# Polynomial Regression mit TensorFlow/Keras

## 📋 Überblick

Dieses Skript implementiert eine **polynomiale Regression** mit TensorFlow/Keras für nichtlineare Daten. Es demonstriert einen vollständigen Machine-Learning-Workflow von der Datengenerierung bis zur Visualisierung.

## 🏗️ UML-Sequenzdiagramm - Gesamter Workflow

```mermaid
sequenceDiagram
    participant User
    participant Script as Main Script
    participant NP as NumPy
    participant TF as TensorFlow/Keras
    participant PLT as Matplotlib

    User->>Script: Starte Ausführung
    
    Script->>Script: Setze Parameter (SEED, N_SAMPLES, etc.)
    Script->>NP: set_seed(SEED)
    Script->>TF: set_seed(SEED)
    
    Script->>Script: 1. Datengenerierung
    Script->>NP: random.uniform(-3, 3)
    Script->>Script: true_function(x)
    Script->>NP: random.normal(scale=6.0)
    
    Script->>Script: 2. Train/Test Split
    Script->>NP: arange(N_SAMPLES)
    Script->>NP: random.shuffle()
    
    Script->>Script: 3. Polynom-Features
    Script->>Script: polynomial_features(x, POLY_DEGREE)
    loop Für jeden Grad 1..POLY_DEGREE
        Script->>NP: base ** d
    end
    
    Script->>Script: 4. Standardisierung
    Script->>NP: mean(X_train_poly)
    Script->>NP: std(X_train_poly)
    Script->>NP: X_train_poly - mean / std
    
    Script->>Script: 5. Modell-Definition
    Script->>TF: keras.Sequential()
    Script->>TF: Dense(1, activation=None)
    
    Script->>Script: 6. Kompilierung
    Script->>TF: compile(optimizer='adam', loss='mse')
    Script->>Script: model.summary()
    
    Script->>Script: 7. Training
    Script->>TF: model.fit()
    loop Für jede Epoch 1..EPOCHS
        Script->>TF: Forward Pass
        Script->>TF: Loss Berechnung
        Script->>TF: Backward Pass
        Script->>TF: Gewichts-Update
    end
    
    Script->>Script: 8. Evaluation
    Script-->>User: Print Train/Test Loss
    Script-->>User: Print Verlustverlauf
    
    Script->>Script: 9. Visualisierung
    Script->>PLT: figure(figsize=(10,4))
    Script->>PLT: subplot(1,2,1)
    Script->>PLT: plot(loss_history)
    Script->>NP: linspace(-3.2, 3.2, 400)
    Script->>Script: model.predict()
    Script->>PLT: subplot(1,2,2)
    Script->>PLT: scatter(), plot()
    Script->>PLT: show()
    
    Script->>Script: 10. Koeffizienten-Analyse
    Script->>TF: model.layers[0].get_weights()
    Script-->>User: Print Polynom-Koeffizienten
```

## 📊 UML-Aktivitätsdiagramm - Hauptprozess

```mermaid
graph TD
    Start[Start Skript] --> Param[Parameter setzen]
    Param --> Seed[Seeds setzen]
    Seed --> GenData[Daten generieren]
    
    GenData --> Split[Trenne Train/Test]
    Split --> PolyFeat[Erstelle Polynom-Features]
    PolyFeat --> Scale[Standardisiere Features]
    
    Scale --> BuildMod[Baue Modell]
    BuildMod --> Compile[Kompiliere Modell]
    Compile --> Train[Trainiere Modell]
    
    Train --> Eval[Evaluiere Modell]
    Eval --> Vis1[Plot Verlustkurven]
    Vis1 --> Vis2[Plot Vorhersagen]
    
    Vis2 --> Coeff[Zeige Koeffizienten]
    Coeff --> End[Ende]
    
    subgraph "Datenvorbereitung"
        GenData
        Split
        PolyFeat
        Scale
    end
    
    subgraph "Modellierung"
        BuildMod
        Compile
        Train
        Eval
    end
    
    subgraph "Visualisierung"
        Vis1
        Vis2
        Coeff
    end
    
    style Start fill:#4CAF50
    style End fill:#F44336
    style GenData fill:#2196F3
    style Train fill:#FF9800
```

## 🔧 UML-Komponentendiagramm - Systemarchitektur

```mermaid
graph TB
    subgraph "Datenpipeline"
        A[Datengenerierung]
        B[Train/Test Split]
        C[Feature-Engineering]
        D[Skalierung]
    end
    
    subgraph "ML-Pipeline"
        E[Modell-Architektur]
        F[Kompilierung]
        G[Training]
        H[Evaluation]
    end
    
    subgraph "Visualisierung"
        I[Verlustkurven]
        J[Vorhersage-Plots]
        K[Koeffizienten]
    end
    
    subgraph "Bibliotheken"
        L[NumPy]
        M[TensorFlow/Keras]
        N[Matplotlib]
    end
    
    A --> B --> C --> D --> E
    E --> F --> G --> H --> I
    H --> J --> K
    
    A --> L
    B --> L
    C --> L
    D --> L
    
    E --> M
    F --> M
    G --> M
    H --> M
    
    I --> N
    J --> N
    K --> N
    
    style A fill:#e1f5fe
    style E fill:#f3e5f5
    style I fill:#e8f5e8
    style L fill:#fff3e0
    style M fill:#fce4ec
    style N fill:#f1f8e9
```

## 🎯 UML-Use Case Diagramm

```mermaid
graph TD
    subgraph "Akteure"
        A[Data Scientist]
        B[Student]
        C[ML-Entwickler]
    end
    
    subgraph "Funktionalitäten"
        UC1[Datengenerierung mit Rauschen]
        UC2[Polynom-Feature-Engineering]
        UC3[Automatische Skalierung]
        UC4[Neuronales Netz Training]
        UC5[Verlust-Visualisierung]
        UC6[Vorhersage-Visualisierung]
        UC7[Koeffizienten-Analyse]
    end
    
    A --> UC1
    A --> UC4
    A --> UC7
    
    B --> UC2
    B --> UC3
    B --> UC5
    B --> UC6
    
    C --> UC1
    C --> UC2
    C --> UC4
    C --> UC7
    
    style A fill:#2196F3,color:white
    style B fill:#4CAF50,color:white
    style C fill:#FF9800,color:white
```

## 📊 UML-Zustandsdiagramm - Datenzustände

```mermaid
stateDiagram-v2
    [*] --> RawData
    RawData --> SplitData : Train/Test Split
    
    state SplitData {
        [*] --> TrainData
        TrainData --> TestData
    }
    
    SplitData --> PolynomialFeatures
    
    state PolynomialFeatures {
        [*] --> Degree1
        Degree1 --> Degree2
        Degree2 --> Degree3
        Degree3 --> DegreeN
    }
    
    PolynomialFeatures --> ScaledFeatures
    ScaledFeatures --> ModelReady
    
    state ModelReady {
        [*] --> Training
        Training --> Trained
        Trained --> Evaluated
    }
    
    ModelReady --> Visualization
    
    state Visualization {
        [*] --> LossPlot
        LossPlot --> PredictionPlot
        PredictionPlot --> Analysis
    }
    
    Visualization --> [*]
    
    note right of RawData
        x: Rohdaten [-3, 3]
        y: f(x) + Rauschen
    end note
    
    note right of PolynomialFeatures
        Features: x, x², x³, ...
        bis Grad POLY_DEGREE
    end note
    
    note right of ModelReady
        Modell lernt
        Gewichte anpassen
        MSE optimieren
    end note
```

## 🏗️ UML-Klassendiagramm - Code-Struktur

```mermaid
classDiagram
    class PolynomialRegressionScript {
        -SEED: int
        -N_SAMPLES: int
        -TEST_RATIO: float
        -POLY_DEGREE: int
        -EPOCHS: int
        -BATCH_SIZE: int
        -LEARNING_RATE: float
        
        -x: ndarray
        -y: ndarray
        -x_train: ndarray
        -y_train: ndarray
        -x_test: ndarray
        -y_test: ndarray
        
        -feature_mean: ndarray
        -feature_std: ndarray
        
        -model: tf.keras.Sequential
        -history: History
        
        +true_function(x): ndarray
        +polynomial_features(x, degree): ndarray
        +main_workflow(): void
    }
    
    class TensorFlowComponents {
        +Sequential()
        +Dense()
        +Adam()
        +compile()
        +fit()
        +predict()
        +get_weights()
    }
    
    class NumPyComponents {
        +random.seed()
        +random.uniform()
        +random.normal()
        +arange()
        +hstack()
        +mean()
        +std()
        +linspace()
    }
    
    class MatplotlibComponents {
        +figure()
        +subplot()
        +plot()
        +scatter()
        +xlabel()
        +ylabel()
        +title()
        +legend()
        +tight_layout()
        +show()
    }
    
    PolynomialRegressionScript --> TensorFlowComponents : verwendet
    PolynomialRegressionScript --> NumPyComponents : verwendet
    PolynomialRegressionScript --> MatplotlibComponents : verwendet
```

## 🛠️ Funktionale Übersicht

### 1. **Datengenerierung**
```python
# Nichtlineare Funktion
def true_function(x):
    return 0.5 * x**3 - 1.0 * x**2 + 2.0 * x

# Mit Rauschen
y = true_function(x) + np.random.normal(scale=6.0)
```

### 2. **Feature-Engineering**
```python
def polynomial_features(x, degree):
    X_poly = []
    for d in range(1, degree+1):
        X_poly.append(x ** d)
    return np.hstack(X_poly)
```

### 3. **Modellarchitektur**
- **Input**: Polynom-Features (Grad 1..POLY_DEGREE)
- **Layer**: Dense(1) ohne Aktivierung (lineare Regression)
- **Loss**: Mean Squared Error (MSE)
- **Optimizer**: Adam

### 4. **Training & Evaluation**
```python
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=0
)
```

### 5. **Visualisierung**
- Verlustkurven (Training vs. Validation)
- Vorhersage vs. echte Funktion
- Train/Test Datenpunkte

## 📈 Performance-Metriken

### Ausgabe-Beispiel:
```
Endgültiger Trainingsverlust (MSE): 25.1234
Endgültiger Testverlust (MSE):      28.5678

Beispiel: Verlustverlauf (erste 5 Epochen):
 Epoche   1 - train_loss: 120.4567 - val_loss: 115.7890
 Epoche   2 - train_loss: 89.1234 - val_loss: 92.3456
...
```

### Gelernte Koeffizienten:
```
x^1 * 1.2345
x^2 * -0.9876
x^3 * 0.4567
...
Bias (Intercept): 0.1234
```

## 🔧 Konfigurationsparameter

| Parameter | Standardwert | Beschreibung |
|-----------|--------------|--------------|
| SEED | 42 | Reproduzierbarkeit |
| N_SAMPLES | 300 | Anzahl Datenpunkte |
| TEST_RATIO | 0.2 | Test-Anteil |
| POLY_DEGREE | 5 | Polynom-Grad |
| EPOCHS | 300 | Trainingsepochen |
| BATCH_SIZE | 32 | Batch-Größe |
| LEARNING_RATE | 0.01 | Lernrate |

## 🎯 Anwendungsfälle

1. **Lernwerkzeug**: Verständnis polynomieller Regression
2. **Experimentieren**: Hyperparameter-Tuning
3. **Visualisierung**: Bias-Variance Trade-off
4. **Benchmarking**: Vergleich mit anderen Algorithmen

## 📚 Theoretische Grundlagen

### Polynomiale Regression
```
y = β₀ + β₁x + β₂x² + ... + βₙxⁿ + ε
```

### Normalisierung
```
x_scaled = (x - μ) / σ
```

### Optimierung
- **Loss Function**: Mean Squared Error
- **Optimizer**: Adam (Adaptive Moment Estimation)

## 🏆 Besonderheiten

### Komplett in NumPy/TF: Keine externen ML-Bibliotheken für Preprocessing
- **Reine NumPy-Implementierung** für alle Preprocessing-Schritte:
  - Datengenerierung (`np.random.uniform`, `np.random.normal`)
  - Train/Test Split (manuell mit `np.random.shuffle`)
  - Polynom-Feature-Engineering (eigene `polynomial_features` Funktion)
  - Standardisierung (manuelle Mittelwert- und Standardabweichungsberechnung)
- **Keine scikit-learn Abhängigkeit** für Preprocessing-Funktionen
- **Transparente Implementierung** aller Datenvorbereitungsschritte

### Reproduzierbar: Deterministische Ergebnisse durch Seeds
- **Seed-Setzung** für NumPy (`np.random.seed(42)`)
- **Seed-Setzung** für TensorFlow (`tf.random.set_seed(42)`)
- **Vorhersehbare Ergebnisse** bei wiederholter Ausführung
- **Wissenschaftliche Reproduzierbarkeit** gewährleistet

### Visuell ansprechend: Integrierte Visualisierungen
- **Komplette Plot-Suite** mit Matplotlib
- **Zweiteilige Visualisierung**:
  1. Verlustkurven über Trainingsepochen
  2. Vorhersage vs. echte Funktion
- **Direkte Ergebnisanalyse** ohne zusätzliche Tools

### Praktisch: Direkt ausführbar ohne Konfiguration
- **Einzelnes ausführbares Skript**
- **Keine externe Konfigurationsdatei** nötig
- **Parameter direkt im Code** anpassbar
- **Sofortige Visualisierung** nach Ausführung

## 🔍 Bibliotheken-Übersicht

| Bibliothek | Verwendungszweck | Alternative in anderen Projekten |
|------------|------------------|----------------------------------|
| **NumPy** | Datenmanipulation, Preprocessing | Oft scikit-learn für Preprocessing |
| **TensorFlow/Keras** | ML-Modell, Training, Evaluation | scikit-learn, PyTorch |
| **Matplotlib** | Visualisierung | Seaborn, Plotly |

**Der Kernunterschied** zu typischen ML-Projekten: Hier wird **alles Preprocessing manuell mit NumPy** implementiert, während in vielen Tutorials scikit-learn-Funktionen wie `PolynomialFeatures`, `train_test_split` und `StandardScaler` verwendet werden. Dies macht den Code **transparenter und lehrreicher**.

---

**Hinweis**: Dieses Skript demonstriert grundlegende ML-Konzepte mit TensorFlow/Keras und eignet sich ideal für Bildungszwecke und Experimente mit nichtlinearen Daten.