# Neural Network Experiment Project

Dieses Projekt untersucht die Auswirkungen verschiedener Aktivierungsfunktionen auf die Performance eines einfachen MNIST-Neural-Networks.  
Es enthält sowohl eine numpy-basierte Implementierung (SimpleNeuralNetwork) als auch eine PyTorch-basierte Implementierung (PyTorchNeuralNetwork).


-   **SimpleNeuralNetwork (numpy-basiert)**\
-   **PyTorchNeuralNetwork (PyTorch-basiert, erweitert die
    SimpleNeuralNetwork-Klasse)**

Zusätzlich gibt es ein **Experiment-Skript (`main.py`)**, das
verschiedene Aktivierungsfunktionen ausprobiert, Trainingsläufe
durchführt und die Ergebnisse automatisch speichert und visualisiert.

------------------------------------------------------------------------
## 📁 Projektstruktur

```
├── neural_network.py        # numpy Neural Network (Baseline)
├── pytorch_nn.py            # PyTorch Neural Network (erweitert SimpleNeuralNetwork)
├── main.py                  # Führt alle Aktivierungs-Experimente durch
├── README.md                # Diese Datei
└── data/                    # MNIST Dataset (automatisch heruntergeladen)
```
---

# 📌 Dateien

  -----------------------------------------------------------------------
  Datei                   Beschreibung
  ----------------------- -----------------------------------------------
  `neural_network.py`      Enthält die Klasse `SimpleNeuralNetwork`
                           (numpy-Version).

  `PyTorchNeuralNetwork.py` Enthält die Klasse `PyTorchNeuralNetwork`
                           (PyTorch-Version, erweitert
                           SimpleNeuralNetwork).

  `main.py`                Führt alle Aktivierungs-Funktions-Experimente
                           durch.

  Automatisch erzeugte Ergebnisdateien.
  `results.csv`,
  `results.json`,         
  `results.png`           
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 🧠 Klassenbeschreibung

## \### **SimpleNeuralNetwork (in `neural_network.py`)**

Eine einfache numpy-Implementierung eines 3‑Layer Neural Networks (784 →
100 → 10).

- Input: 784 Neuronen
- Hidden: 100 Neuronen
- Output: 10 Neuronen
- Aktivierung: Sigmoid
- Methoden: sigmoid(), test(), get_weight, set_weights, get_network_info
- Wird als Basisklasse verwendet

### Eigenschaften:

-   Zufällige Gewichtsinitialisierung
-   Sigmoid als feste Aktivierungsfunktion
-   `test(input)` → führt einen Forward-Pass für einen einzelnen Input
    aus

### Wird benutzt für:

-   einfache Demonstration
-   Vergleich mit PyTorch-Version
-   keine Training-Funktion enthalten

------------------------------------------------------------------------

## \### **PyTorchNeuralNetwork (in `pytorch_nn.py`)**

Erbt von `SimpleNeuralNetwork` **und** `torch.nn.Module`.

### Eigenschaften:

-   Beliebige Aktivierungsfunktion:
    -   Sigmoid\
    -   LeakyReLU (mit Parametern)\
    -   PReLU (trainierbarer Parameter)\
    -   ELU (mit Parametern)
-   Trainingsfunktion (`train_model`)
-   Evaluationsfunktion (`evaluate`)
-   Unterstützt GPU / CUDA / MPS
-   Optimizer wählbar:
    -   Adam\
    -   SGD mit Momentum 0.9

### Aufbau:

    SimpleNeuralNetwork
           ↑
           | (Vererbung)
    PyTorchNeuralNetwork ───→ torch.nn.Module

------------------------------------------------------------------------

# 📘 UML Klassendiagramm

    +-------------------------------------------------------------+
    |                    SimpleNeuralNetwork                      |
    +-------------------------------------------------------------+
    | - input_size: int                                           |
    | - hidden_size: int                                          |
    | - output_size: int                                          |
    | - learning_rate: float                                      |
    | - weights_input_hidden: np.array                            |
    | - weights_hidden_output: np.array                           |
    +-------------------------------------------------------------+
    | + sigmoid(x)                                                |
    | + test(input_vector)                                        |
    | + get_weights() : tuple                                     |
    | + set_weights(wIH, wHO)                                     |
    | + get_network_info() : str                                  |
    +-------------------------------------------------------------+

                               ▲
                               |
                               |
    +-------------------------------------------------------------+
    |                   PyTorchNeuralNetwork                      |
    |        inherits SimpleNeuralNetwork, nn.Module              |
    +-------------------------------------------------------------+
    | - activation: nn.Module                                     |
    | - optimizer: torch.optim                                    |
    | - device: torch.device                                      |
    | - loss_fn: CrossEntropyLoss                                 |
    | - fc1: Linear                                               |
    | - fc2: Linear                                               |
    +-------------------------------------------------------------+
    | + to_device(device)                                         |
    | + forward(x)                                                |
    | + train_model(loader, epochs, log_every)                    |
    | + evaluate(loader)                                          |
    +-------------------------------------------------------------+

------------------------------------------------------------------------

# 🚀 Verwendung von `main.py`

Das Script lädt MNIST, trainiert Modelle mit verschiedenen
Aktivierungsfunktionen und speichert Ergebnisse automatisch als:

-   **CSV**
-   **JSON**
-   **PNG Grafik**

------------------------------------------------------------------------

# ▶️ Programm starten (Standard-Einstellungen)

    python main.py

Standard:

-   Epochs: 10\
-   Batchsize: 64\
-   Learning Rate: 0.01\
-   Optimizer: Adam\
-   Device: automatisch (GPU → MPS → CPU)

------------------------------------------------------------------------

# ⚙️ Parameteroptionen

## \### 1️⃣ Epochen ändern

    python main.py --epochs 15

    Anzahl der Trainingsdurchläufe.  
    Höher = bessere Genauigkeit, aber längere Trainingszeit.

------------------------------------------------------------------------

## \### 2️⃣ Batchgröße ändern

    python main.py --batch_size 128

    Wie viele Bilder gleichzeitig verarbeitet werden.  
    Kleinere Batches = stabiler, größere Batches = schneller.

------------------------------------------------------------------------

## \### 3️⃣ Learning Rate ändern

    python main.py --lr 0.005

    Steuert die Stärke der Gewichtsänderungen.  
    Zu hoch = instabil, zu niedrig = Training dauert lange.

------------------------------------------------------------------------

## \### 4️⃣ Optimizer wählen

Adam → schnell, stabil, empfohlen

    python main.py --optimizer adam

SGD mit Momentum → klassisch, manchmal präziser, oft langsam

    python main.py --optimizer sgd

------------------------------------------------------------------------

## \### 5️⃣ GPU deaktivieren (Force CPU)

    python main.py --force_cpu

------------------------------------------------------------------------

## \### 6️⃣ Ergebnisdateien mit Prefix speichern

    python main.py --out_prefix experiment1_

Erzeugt:

-   `experiment1_results.csv`
-   `experiment1_results.json`
-   `experiment1_results.png`

------------------------------------------------------------------------

## Ergebnisdateien

| Datei | Erklärung |
|-------|-----------|
| results.csv | Tabellarische Übersicht |
| results.json | Maschinenlesbare Ausgabe |
| results.png | Diagramm der ACC für alle Aktivierungen |

---


# 📊 Ausgabedateien

Nach dem Lauf erhältst du automatisch:

  Datei            Bedeutung
  ---------------- -----------------------------------
  `results.csv`    Tabellenformat für Excel / Sheets
  `results.json`   Maschinenlesbares Format
  `results.png`    Plot der Accuracy-Werte

------------------------------------------------------------------------

## Lizenz

Dieses Projekt darf frei für Lernzwecke genutzt, modifiziert und erweitert werden.
