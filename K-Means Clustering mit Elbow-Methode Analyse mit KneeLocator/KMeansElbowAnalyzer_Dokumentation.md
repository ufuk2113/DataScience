# KMeans-Elbow-Analyse (UML & Erklärung)

## 🧩 UML-Diagramm

![UML Diagramm](KMeansElbowAnalyzer_UML.png)

---

## 📘 Klassenbeschreibung

### **KMeansElbowAnalyzer**
Diese Klasse führt eine Elbow-Analyse für den K-Means-Algorithmus durch.

#### **Attribute**
| Attribut | Typ | Beschreibung |
|-----------|-----|---------------|
| `data` | array-like | Datensatz, auf dem K-Means angewendet wird |
| `inertia_values` | list | Liste zur Speicherung der berechneten Inertia-Werte |
| `k_values` | list | Liste der getesteten Clusteranzahlen (1 bis 10) |

#### **Methoden**
| Methode | Beschreibung |
|----------|---------------|
| `__init__(data)` | Initialisiert das Objekt mit den Eingabedaten |
| `run_kmeans()` | Führt K-Means für k=1...10 aus und speichert die Inertia-Werte |
| `plot_elbow()` | Erstellt den Elbow-Plot zur Bestimmung der optimalen Clusteranzahl |
| `analyze()` | Führt die komplette Analyse (K-Means + Plot) automatisch aus |

---

## 📊 Beispielhafte Verwendung

```python
from sklearn.datasets import make_blobs

# Beispielhafte Datenerzeugung
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)

# Analyse starten
analyzer = KMeansElbowAnalyzer(X)
analyzer.analyze()
```

---

## 🧠 Erklärung der Elbow-Methode

Die **Elbow-Methode** dient zur Bestimmung der optimalen Anzahl von Clustern.
Man führt K-Means für verschiedene k-Werte durch und beobachtet die Trägheit (`inertia_`).
Wenn die Reduktion der Inertia bei einem bestimmten k-Wert deutlich abflacht,
spricht man vom **"Knick" (Elbow)** – dieser k-Wert ist oft optimal.
