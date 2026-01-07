

# Data Science Beispielprojekte

Diese Sammlung enthält verschiedene Data Science Projekte und Algorithmen, die für Lern- und Demonstrationszwecke erstellt wurden.

## 📊 Projekte im Überblick

### 1. **DataCleaner**
   - **Beschreibung**: Tool zur Datenbereinigung und -vorverarbeitung
   - **Funktionen**: Handhabung von fehlenden Werten, Ausreißererkennung, Datenormalisierung
   - **Anwendungsfall**: Vorbereitung von Datensätzen für maschinelles Lernen

### 2. **Clustering Algorithmen**
   - **HierarchicalClustering**: Hierarchisches Clustering mit dendrogramm-basierter Visualisierung
   - **K-Means Clustering mit Elbow-Methode**: Implementierung von K-Means mit optimaler Cluster-Anzahl Bestimmung
   - **Analyse**: Evaluation von Clustering-Ergebnissen und Performance-Metriken

### 3. **PolynomialModelSelector**
   - **Beschreibung**: Automatische Auswahl des optimalen Polynomgrades für Regressionsmodelle
   - **Features**: Kreuzvalidierung, Regularisierung, Modellvergleich
   - **Use Case**: Finden der besten Anpassung für nicht-lineare Beziehungen

### 4. **Neuronale Netze**
   - **SimpleNeuralNetwork**: Grundlegende Implementierung eines neuronalen Netzes
   - **SimpleNeuralNetwork mit PyTorch**: Modernere Implementierung mit PyTorch Framework
   - **Anwendung**: Klassifikation und Regressionsaufgaben mit Deep Learning

### 5. **Data Generation**
   - **UserGenerator**: Tool zur Erzeugung synthetischer Benutzerdaten
   - **Features**: Realistische Daten mit konsistenten Abhängigkeiten

### 6. **weatherAnalysisProject**
   - **Beschreibung**: Komplettes Projekt zur Wetterdatenanalyse
   - **Komponenten**: Datenaufbereitung, Visualisierung, Vorhersagemodelle
   - **Ziel**: Mustererkennung in meteorologischen Daten

## 🛠️ Technologien

- **Programmiersprachen**: Python
- **Bibliotheken**: 
  - Scikit-learn (Clustering, Modellauswahl)
  - PyTorch (Neuronale Netze)
  - NumPy & Pandas (Datenverarbeitung)
  - Matplotlib/Seaborn (Visualisierung)

## 📁 Projektstruktur


data-science-examples/
│
├── data_cleaning/
│   └── DataCleaner.py
│
├── clustering/
│   ├── hierarchical_clustering.py
│   ├── kmeans_elbow.py
│   └── analysis/
│
├── model_selection/
│   └── PolynomialModelSelector.py
│
├── neural_networks/
│   ├── SimpleNeuralNetwork.py
│   └── pytorch_nn.py
│
├── data_generation/
│   └── UserGenerator.py
│
├── projects/
│   └── weather_analysis/
│
└── README.md


## 🚀 Schnellstart

1. **Repository klonen**
   ```bash
   git clone [repository-url]
   cd data-science-examples
   ```

2. **Abhängigkeiten installieren**
   ```bash
   pip install -r requirements.txt
   ```

3. **Beispiel ausführen**
   ```bash
   python clustering/kmeans_elbow.py
   ```

## 📈 Anwendungsbeispiele

### K-Means mit Elbow-Methode
```python
from clustering.kmeans_elbow import KMeansElbow

# Daten laden
data = load_your_data()

# Optimales K finden
kmeans = KMeansElbow(data)
optimal_k = kmeans.find_optimal_clusters()

# Clustering durchführen
clusters = kmeans.cluster_data(optimal_k)
```

### Datenbereinigung
```python
from data_cleaning.DataCleaner import DataCleaner

cleaner = DataCleaner('raw_data.csv')
cleaned_data = cleaner.clean()
```

## 📚 Lernziele

- Verständnis grundlegender Data Science Konzepte
- Implementierung verschiedener ML-Algorithmen
- Datenvorverarbeitung und -bereinigung
- Modellauswahl und -evaluation
- Praktische Anwendung in realen Szenarien

## 🤝 Beitragen

Beiträge sind willkommen! Bitte erstellen Sie einen Pull Request oder öffnen Sie ein Issue für:
- Neue Algorithmen
- Verbesserungen der bestehenden Implementierungen
- Fehlerbehebungen
- Zusätzliche Dokumentation

---

*Diese Sammlung dient Bildungszwecken und kann in eigenen Projekten verwendet werden.*
```

Diese README.md Datei bietet:
1. Eine klare Übersicht aller Projekte
2. Strukturierte Beschreibungen
3. Installations- und Nutzungsanleitungen
4. Code-Beispiele
5. Beitragsmöglichkeiten

Du kannst die Datei je nach Bedarf anpassen, z.B. um spezifische Installationsanweisungen oder Lizenzinformationen hinzuzufügen.