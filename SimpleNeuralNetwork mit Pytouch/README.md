# MNIST Activation Function Experiments

## 📋 Projektübersicht

Dieses Projekt implementiert ein neuronales Netzwerk zur Klassifikation der MNIST-Handschriftendatenbank mit verschiedenen Aktivierungsfunktionen. Es enthält zwei Implementierungen:
1. **SimpleNeuralNetwork** - Eine reine NumPy-Implementierung
2. **PyTorchNeuralNetwork** - Eine PyTorch-Implementierung mit erweiterten Funktionen

## 📁 Dateistruktur

```
├── main.py                    # Hauptskript für Experimente
├── neural_network.py          # NumPy-Netzwerk (SimpleNeuralNetwork)
├── PyTorchNeuralNetwork.py    # PyTorch-Netzwerk (PyTorchNeuralNetwork)
├── requirements.txt           # Python-Abhängigkeiten
└── README.md                  # Diese Datei
```

## 🧠 UML-Diagramme

### 1. Klassendiagramm

```mermaid
classDiagram
    class SimpleNeuralNetwork {
        -input_size: int
        -hidden_size: int
        -output_size: int
        -learning_rate: float
        -weights_input_hidden: ndarray
        -weights_hidden_output: ndarray
        +__init__(input_size, hidden_size, output_size, learning_rate, weights_input_hidden, weights_hidden_output)
        +sigmoid(x) ndarray
        +test(input_vector) ndarray
        +get_weights() tuple
        +set_weights(weights_input_hidden, weights_hidden_output)
        +get_network_info() str
    }

    class PyTorchNeuralNetwork {
        -fc1: Linear
        -fc2: Linear
        -activation: Module
        -loss_fn: CrossEntropyLoss
        -optimizer: Optimizer
        -device: Device
        +__init__(input_size, hidden_size, output_size, activation, activation_param, learning_rate, optimizer_name)
        +to_device(device)
        +forward(x) Tensor
        +train_model(train_loader, epochs, log_every)
        +evaluate(test_loader) float
    }

    SimpleNeuralNetwork <|-- PyTorchNeuralNetwork
    nn.Module <|-- PyTorchNeuralNetwork

    note for SimpleNeuralNetwork "Reine NumPy-Implementierung\n• Manuelle Gewichtsverwaltung\n• Nur Sigmoid-Aktivierung\n• Kein automatisches Training"
    
    note for PyTorchNeuralNetwork "PyTorch-Implementierung mit Features:\n• Mehrere Aktivierungsfunktionen\n• GPU-Unterstützung\n• Adam/SGD-Optimizer\n• Automatisches Training/Evaluation"
```

### 2. Sequenzdiagramm - Trainingsprozess

```mermaid
sequenceDiagram
    participant Main as main.py
    participant PTNN as PyTorchNeuralNetwork
    participant DataLoader as DataLoader
    participant Optimizer as Optimizer
    participant Device as Device (CPU/GPU/MPS)

    Main->>+PTNN: Erstellt Modell mit Parametern
    Main->>Device: get_device() - Hardware-Detektion
    Main->>+DataLoader: prepare_data() - MNIST laden
    
    Main->>PTNN: to_device(device)
    PTNN->>Device: self.to(device)
    PTNN->>Optimizer: Erstellt Optimizer (Adam/SGD)
    
    loop Für jede Epoche
        loop Für jeden Batch
            DataLoader->>PTNN: Batch (Images, Labels)
            PTNN->>PTNN: forward(images)
            PTNN->>PTNN: loss = loss_fn(outputs, labels)
            PTNN->>PTNN: loss.backward()
            PTNN->>Optimizer: optimizer.step()
            PTNN->>Optimizer: optimizer.zero_grad()
            
            alt log_every erreicht
                PTNN->>Main: Druckt Trainingsfortschritt
            end
        end
    end
    
    Main->>PTNN: evaluate(test_loader)
    PTNN->>Main: Gibt Genauigkeit zurück
    Main->>Main: Speichert Ergebnisse (CSV, JSON, PNG)
```

### 3. Aktivierungsfunktionen UML

```mermaid
classDiagram
    class ActivationFunctions {
        <<interface>>
        +forward(x) Tensor
    }
    
    class Sigmoid {
        +forward(x) Tensor
    }
    
    class LeakyReLU {
        -negative_slope: float
        +forward(x) Tensor
    }
    
    class PReLU {
        -weight: Parameter
        +forward(x) Tensor
    }
    
    class ELU {
        -alpha: float
        +forward(x) Tensor
    }
    
    ActivationFunctions <|.. Sigmoid
    ActivationFunctions <|.. LeakyReLU
    ActivationFunctions <|.. PReLU
    ActivationFunctions <|.. ELU
    
    PyTorchNeuralNetwork o-- ActivationFunctions : verwendet
```

### 4. Datenflussdiagramm

```mermaid
flowchart TD
    A[main.py CLI Aufruf] --> B[Argument Parsing]
    B --> C[Hardware-Detektion CPU/GPU/MPS]
    C --> D[Experiment-Liste definieren]
    
    D --> E{Für jedes Experiment}
    E --> F[DataLoader MNIST vorbereiten]
    F --> G[PyTorchNeuralNetwork erstellen]
    G --> H[Modell auf Device verschieben]
    H --> I[Training mit Trainingsdaten]
    I --> J[Evaluation mit Testdaten]
    J --> K[Ergebnis speichern]
    K --> E
    
    E --> L[Alle Experimente abgeschlossen]
    L --> M[CSV Export]
    L --> N[JSON Export]
    L --> O[PNG Diagramm]
    M --> P[Ergebnisse anzeigen]
    N --> P
    O --> P
```

## 🚀 Installation & Ausführung

### Voraussetzungen
```bash
# Python 3.8+ erforderlich
python --version

# Abhängigkeiten installieren
pip install -r requirements.txt
```

### Ausführung
```bash
# Standard-Experiment (10 Epochen, Adam Optimizer)
python main.py

# Mit angepassten Parametern
python main.py --epochs 20 --batch_size 128 --lr 0.001 --optimizer sgd

# CPU erzwingen (auch wenn GPU verfügbar)
python main.py --force_cpu

# Output-Präfix für Dateien
python main.py --out_prefix "exp1_"
```

## ⚙️ Standard-Experimente

Das Skript führt folgende Experimente automatisch durch:

| Aktivierungsfunktion | Parameter | Beschreibung |
|---------------------|-----------|--------------|
| Sigmoid             | -         | Klassische Sigmoid-Funktion |
| Leaky ReLU          | 0.01      | Kleine negative Steigung |
| Leaky ReLU          | 0.05      | Mittlere negative Steigung |
| Leaky ReLU          | 0.1       | Große negative Steigung |
| Leaky ReLU          | 0.5       | Sehr große negative Steigung |
| PReLU               | -         | Parametrisierte ReLU (lernbar) |
| ELU                 | 0.1       | Exponential Linear Unit (kleines Alpha) |
| ELU                 | 0.2       | ELU mit mittlerem Alpha |
| ELU                 | 0.3       | ELU mit großem Alpha |

## 📊 Ausgabe

Nach der Ausführung werden folgende Dateien erstellt:

1. **results.csv** - Tabellarische Ergebnisse
2. **results.json** - JSON-Formatierte Ergebnisse
3. **results.png** - Balkendiagramm der Genauigkeiten

### Beispielausgabe:
```
Activation       Param    Accuracy (%)
---------------  -------  ------------
sigmoid          -        97.23
leaky_relu       0.01     98.45
leaky_relu       0.05     98.51
leaky_relu       0.1      98.37
leaky_relu       0.5      97.89
prelu            -        98.63
elu              0.1      98.29
elu              0.2      98.34
elu              0.3      98.27
```

## 🏗️ Architekturdetails

### SimpleNeuralNetwork (neural_network.py)
- **Input Layer**: 784 Neuronen (28×28 MNIST Bilder)
- **Hidden Layer**: 100 Neuronen (konfigurierbar)
- **Output Layer**: 10 Neuronen (0-9 Ziffern)
- **Aktivierung**: Nur Sigmoid
- **Gewichtsinitialisierung**: Zufällig [-0.5, 0.5]

### PyTorchNeuralNetwork (PyTorchNeuralNetwork.py)
- **Vererbung**: Erbt von `SimpleNeuralNetwork` und `nn.Module`
- **Aktivierungsfunktionen**:
  - Sigmoid (Standard)
  - Leaky ReLU (mit parametrisierbarem Slope)
  - PReLU (parametrisiert, lernbar)
  - ELU (Exponential Linear Unit)
- **Optimizer**: Adam oder SGD mit Momentum
- **Loss Function**: CrossEntropyLoss
- **Hardware-Unterstützung**: CPU, CUDA, MPS (macOS)

## 🔧 Erweiterungsmöglichkeiten

1. **Neue Aktivierungsfunktionen hinzufügen**:
   - Swish, Mish, GELU in `PyTorchNeuralNetwork.__init__()`

2. **Hyperparameter-Tuning**:
   - Learning Rate Scheduler
   - Batch Normalization
   - Dropout Layer

3. **Weitere Datensätze**:
   - Fashion-MNIST
   - CIFAR-10 (mit Conv-Netzwerk)

4. **Visualisierung erweitern**:
   - Loss-Kurven pro Experiment
   - Confusion Matrices
   - Feature Visualisierung

## 📝 Lizenz

Dieses Projekt ist für Bildungszwecke erstellt. Die MNIST-Datenbank wird unter der Creative Commons Attribution-Share Alike 3.0 Lizenz verteilt.