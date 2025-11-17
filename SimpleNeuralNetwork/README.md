
# Simple Neural Network (NumPy)

Dieses Projekt implementiert ein einfaches dreischichtiges Neural Network (Feedforward Network) mit NumPy.  
Es besteht aus einer Klasse `SimpleNeuralNetwork` und einer `main.py`, die verschiedene Netzwerkvarianten lädt, testet und die Ergebnisse ausgibt.

---

## 📘 Inhalt

- [Überblick](#überblick)
- [Dateistruktur](#dateistruktur)
- [NeuralNetwork-Klasse](#neuralnetwork-klasse)
- [UML-Diagramm](#uml-diagramm)
- [Beispielausgabe](#beispielausgabe)
- [Ausführen](#ausführen)

---

## Überblick

Das Neural Network hat folgende Architektur:

- **Input Layer:** 784 Neuronen  
- **Hidden Layer:** 100 Neuronen  
- **Output Layer:** 10 Neuronen  
- **Aktivierungsfunktion:** Sigmoid  
- **Ziel:** Klassifizieren von handgeschriebenen Ziffern (MNIST)

---

## Dateistruktur

```
📁 Projekt
│
├── neural_network.py   # Implementiert die Klasse SimpleNeuralNetwork
├── main.py             # Führt Beispieltests durch
└── README.md           # Dieses Dokument
```

---

## NeuralNetwork-Klasse

Die Klasse ist komplett modular aufgebaut:

### **Hauptattribute**
- `input_size` – Anzahl Input-Neuronen (Standard: 784)  
- `hidden_size` – Anzahl Hidden-Neuronen (Standard: 100)  
- `output_size` – Anzahl Output-Neuronen (Standard: 10)  
- `learning_rate` – Lernrate  
- `weights_input_hidden` – Gewichtsmatrix Input → Hidden  
- `weights_hidden_output` – Gewichtsmatrix Hidden → Output  

### **Methoden**
| Methode | Beschreibung |
|--------|--------------|
| `sigmoid(x)` | Aktivierungsfunktion |
| `test(input_vector)` | Forward Pass und Ausgabe des Netzwerk-Outputs |
| `get_weights()` | Gibt die beiden Gewichtsmatrizen zurück |
| `set_weights(w1, w2)` | Setzt neue Gewichtsmatrizen |
| `get_network_info()` | Gibt eine formatierte Info zum Netzwerk zurück |

---

## UML-Diagramm

```
+--------------------------------------------------+
|               SimpleNeuralNetwork                |
+--------------------------------------------------+
| - input_size: int                                |
| - hidden_size: int                               |
| - output_size: int                               |
| - learning_rate: float                           |
| - weights_input_hidden: np.ndarray               |
| - weights_hidden_output: np.ndarray              |
+--------------------------------------------------+
| + __init__(input_size, hidden_size,              |
|             output_size, learning_rate,          |
|             weights_input_hidden,                |
|             weights_hidden_output)               |
| + sigmoid(x) : np.ndarray                        |
| + test(input_vector) : np.ndarray                |
| + get_weights() : tuple                          |
| + set_weights(wIH, wHO)                          |
| + get_network_info() : str                       |
+--------------------------------------------------+
```

---

## Beispielausgabe

Beim Starten von `main.py` werden erzeugt:

- ein Standard-Netzwerk  
- ein Netzwerk mit selbst definierten Gewichten  
- ein Simulierter MNIST-Input  
- Forward Pass durch beide Netzwerke  
- Ausgabe der Klassenvorhersage  

---

## Ausführen

```bash
python3 main.py
```

Stellt sicher, dass NumPy installiert ist:

```bash
pip install numpy
```

---

## Lizenz

Dieses Projekt darf frei für Lernzwecke genutzt, modifiziert und erweitert werden.
