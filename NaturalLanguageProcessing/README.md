# NaturalLanguageProcessing Verbessert - README

## 📋 Projektübersicht

Dieses **vollständig überarbeitete NLP-System** implementiert eine moderne Natural Language Processing Pipeline mit erweiterten Funktionen. Es kombiniert **Word2Vec Skip-Gram Embeddings** mit einem **kontextuellen Transformer-Modell** für state-of-the-art semantische Analysen.

### 🎯 Hauptmerkmale
- ✅ **Variable Satzlängen**: Unterstützung für unterschiedliche Sequenzlängen mit intelligenter Padding-Logik
- ✅ **Batch-Verarbeitung**: Effiziente Parallelverarbeitung mehrerer Texte
- ✅ **Erweiterte Tokenisierung**: Unicode-Unterstützung für Deutsch (Umlaute, ß)
- ✅ **Stopwort-Filterung**: Optionale Entfernung häufiger Wörter
- ✅ **Negative Sampling**: Verbessertes Training für große Vokabulare
- ✅ **Visualisierungen**: PCA/t-SNE Plots und Heatmaps
- ✅ **Export-Funktionen**: TXT, CSV, JSON, Word2Vec-kompatible Formate
- ✅ **Robuste Fehlerbehandlung**: Behandlung unbekannter Wörter und leerer Eingaben

## 🏗️ Architektur-Übersicht

```mermaid
graph TB
    subgraph "Datenverarbeitung"
        A[Rohtext Korpus] --> B[Tokenisierung]
        B --> C[Wort-zu-Index Mapping]
        C --> D[Padding für variable Längen]
    end
    
    subgraph "Training Pipeline"
        D --> E[Word2VecEmbedding]
        E --> F[SkipGram Training]
        F --> G[Trained Embeddings]
    end
    
    subgraph "Inferenz Pipeline"
        G --> H[ContextualTransformer]
        H --> I[SemanticAnalyzer]
        I --> J[Satz-Embeddings]
        I --> K[Wort-Embeddings]
    end
    
    subgraph "Anwendungen"
        J --> L[Ähnlichkeitsberechnung]
        K --> M[Semantische Suche]
        J --> N[Dokumentklassifikation]
        K --> O[Wortanalogien]
    end
    
    style A fill:#e1f5fe
    style E fill:#fff3e0
    style H fill:#e8f5e8
    style I fill:#f3e5f5
```

## 📊 Detailliertes UML-Klassendiagramm

```mermaid
classDiagram
    %% Kern-Klassen
    class SemanticAnalyzer {
        -Word2VecEmbedding _wort_embedder
        -ContextualTransformer _kontext_transformer
        -int _embedding_dimension
        -bool _use_weighted_mean
        -dict _wort_haeufigkeiten
        +__init__(wort_embedder, kontext_transformer, embedding_dimension=64)
        +berechne_satz_aehnlichkeiten(saetze, such_text, zeige_top_n=None, min_aehnlichkeit=0.0)
        +berechne_wort_aehnlichkeiten(wort_paare, zeige_embedding_stats=False)
        +finde_aehnlichste_saetze(such_text, saetze, anzahl=3) List[Tuple]
        +berechne_dokument_embedding(dokument) Tensor
        -_erstelle_satz_vektor(satz) Tensor
        -_erstelle_wort_vektor(wort) Tensor
        -_berechne_wort_haeufigkeiten() Dict
        -_berechne_wort_gewicht(wort) float
    }
    
    class Word2VecEmbedding {
        -str _text_korpus
        -int _kontext_fenster
        -int _embedding_dimension
        -int _max_satzelemente
        -bool _use_stopwords
        -list _trainings_paare
        -list _woerter
        -dict _wort_zu_index
        -dict _index_zu_wort
        -SkipGramModell _skipgram_modell
        -optim.Adam _optimierer
        -nn.CrossEntropyLoss _verlust_funktion
        +woerter: list
        +trainings_paare: list
        +woerterbuch_groesse: int
        +embedding_dimension: int
        +embedding_schicht: nn.Embedding
        +__init__(text_korpus, kontext_fenster=5, embedding_dimension=64, max_satzelemente=50)
        +trainiere(epochen=10, batch_groesse=32, lernrate=0.01, zeige_progress=True)
        +erhalte_indizes_von_satz(satz) Tensor
        +erhalte_index_von_wort(wort) Tensor
        +erhalte_embedding_matrix() ndarray
        +berechne_wort_aehnlichkeit(wort1, wort2) float
        +finde_semantische_nachbarn(wort, top_k=10, min_aehnlichkeit=0.0) List[Tuple]
        +visualisiere_embeddings(woerter, methode="pca", titel="Wort Embedding Visualisierung")
        +exportiere_embeddings(datei_pfad, format="txt")
        -_tokenisiere_text(text) list
        -_berechne_wort_haeufigkeiten()
        -_filtere_vokabular()
        -_erstelle_wort_mappings()
        -_generiere_trainings_paare()
    }
    
    class SkipGramModell {
        -int woerterbuch_groesse
        -int embedding_dimension
        -int negative_samples
        -nn.Embedding wort_embeddings
        -nn.Embedding kontext_embeddings
        -nn.Linear ausgabe_schicht
        +__init__(woerterbuch_groesse, embedding_dimension, negative_samples=5, embedding_init='normal')
        +forward(zentrum_indices) Tensor
        +berechne_aehnlichkeits_matrix() Tensor
        +get_embedding(wort_index) Tensor
        +normalisiere_embeddings()
        -_initialisiere_embeddings(init_method)
    }
    
    class SkipGramDatenSatz {
        -list paare
        -int negative_samples
        -ndarray wort_verteilung
        -ndarray einzigartige_indizes
        +__init__(paare, negative_samples=5)
        +__len__() int
        +__getitem__(index) Tuple[Tensor, Tensor]
        +erstelle_batch_mit_padding(batch_groesse) Generator
        +generiere_negative_samples(zentrum_index, kontext_index, anzahl=5) List[int]
        -_berechne_wort_verteilung()
    }
    
    class ContextualTransformer {
        -Word2VecEmbedding wort_embedder
        -int embedding_dimension
        -int anzahl_koepfe
        -int anzahl_schichten
        -int maximale_sequenz_laenge
        -float dropout_rate
        -bool use_pre_norm
        -nn.Embedding embedding_schicht
        -nn.Parameter positions_kodierung
        -nn.ModuleList transformer_schichten
        -nn.LayerNorm final_norm
        -nn.Linear ausgabe_projektion
        +__init__(wort_embedder, embedding_dimension=64, anzahl_koepfe=4, anzahl_schichten=2, maximale_sequenz_laenge=100, dropout_rate=0.1, use_pre_norm=True)
        +forward(indices, maskierung=None) Tensor
        +berechne_aufmerksamkeits_gewichte(indices, schicht_index=0) Tensor
        +berechne_kontextuelle_aehnlichkeit(indices1, indices2) float
        +extrahiere_features(indices, ebene="last") Tensor
        -_initialisiere_positions_kodierung()
        -_erstelle_padding_maskierung(indices) Tensor
    }
    
    class TransformerSchicht {
        -int embedding_dimension
        -int anzahl_koepfe
        -float dropout_rate
        -int feedforward_dimension
        -bool use_pre_norm
        -str activation
        -MehrkopfAufmerksamkeit selbst_aufmerksamkeit
        -nn.LayerNorm norm1
        -nn.LayerNorm norm2
        -nn.Sequential feed_forward
        -nn.Dropout dropout1
        -nn.Dropout dropout2
        +__init__(embedding_dimension, anzahl_koepfe=4, dropout_rate=0.1, feedforward_dimension=None, use_pre_norm=True, activation='gelu')
        +forward(x, maskierung=None) Tensor
        +berechne_aufmerksamkeits_gewichte(x, maskierung=None) Tensor
        -_get_activation(activation_name) nn.Module
        -_initialisiere_parameter()
    }
    
    class MehrkopfAufmerksamkeit {
        -int embedding_dimension
        -int anzahl_koepfe
        -float dropout_rate
        -bool bias
        -int dimension_pro_kopf
        -nn.Linear query_projektion
        -nn.Linear key_projektion
        -nn.Linear value_projektion
        -nn.Linear ausgabe_projektion
        -nn.Dropout aufmerksamkeit_dropout
        -float skalierung
        +__init__(embedding_dimension, anzahl_koepfe=4, dropout=0.1, bias=True)
        +forward(query, schluessel, wert, maskierung=None) Tensor
        +berechne_aufmerksamkeits_karten(query, schluessel, maskierung=None) Tuple[Tensor, Tensor]
        -_skalierte_aufmerksamkeit(query, schluessel, wert, maskierung) Tuple[Tensor, Tensor]
        -_initialisiere_parameter()
    }
    
    %% Abhängigkeiten und Beziehungen
    SemanticAnalyzer --> Word2VecEmbedding : verwendet
    SemanticAnalyzer --> ContextualTransformer : verwendet
    
    Word2VecEmbedding --> SkipGramModell : trainiert
    Word2VecEmbedding --> SkipGramDatenSatz : erzeugt
    
    ContextualTransformer --> TransformerSchicht : enthält in layers
    ContextualTransformer --> Word2VecEmbedding : verwendet embedding_schicht
    
    TransformerSchicht --> MehrkopfAufmerksamkeit : enthält
    
    SkipGramDatenSatz ..|> torch.utils.data.Dataset : erbt
    SkipGramModell ..|> nn.Module : erbt
    ContextualTransformer ..|> nn.Module : erbt
    TransformerSchicht ..|> nn.Module : erbt
    MehrkopfAufmerksamkeit ..|> nn.Module : erbt
```

## 🔄 Sequenzdiagramm - Komplette Pipeline

```mermaid
sequenceDiagram
    participant Benutzer
    participant Haupt as einbettung_erweitert.py
    participant SA as SemanticAnalyzer
    participant WE as Word2VecEmbedding
    participant SG as SkipGramModell
    participant DS as SkipGramDatenSatz
    participant CT as ContextualTransformer
    participant TS as TransformerSchicht
    participant MA as MehrkopfAufmerksamkeit
    
    Benutzer->>Haupt: Startet Programm
    Haupt->>WE: Word2VecEmbedding(korpus, kontext_fenster=5, embedding_dim=64)
    WE->>WE: Tokenisiert Korpus
    WE->>WE: Erstellt Wort-zu-Index Mapping
    WE->>DS: SkipGramDatenSatz(trainings_paare)
    WE->>SG: SkipGramModell(vocab_size, embedding_dim)
    
    Haupt->>WE: trainiere(epochen=15)
    loop Jede Epoche
        WE->>DS: Batch-Generierung
        DS-->>WE: (zentrum_indices, kontext_indices)
        WE->>SG: forward(zentrum_indices)
        SG-->>WE: logits
        WE->>WE: Berechnet CrossEntropyLoss
        WE->>WE: Backpropagation
        WE->>WE: Optimizer.step()
    end
    
    Haupt->>CT: ContextualTransformer(WE, embedding_dim=64)
    CT->>CT: Initialisiert Positionskodierungen
    
    Haupt->>SA: SemanticAnalyzer(WE, CT, embedding_dim=64)
    
    Benutzer->>Haupt: Führt Analysen durch
    Haupt->>SA: berechne_satz_aehnlichkeiten(saetze, suchtext)
    
    loop Für jeden Satz
        SA->>WE: erhalte_indizes_von_satz(satz)
        WE-->>SA: indices (mit Padding)
        SA->>CT: forward(indices, maskierung)
        CT->>CT: _erstelle_padding_maskierung(indices)
        CT->>CT: wort_embeddings + positions_kodierung
        
        loop Für jede TransformerSchicht
            CT->>TS: forward(embeddings, maskierung)
            TS->>MA: forward(query, key, value, maskierung)
            MA-->>TS: aufmerksamkeits_output
            TS->>TS: Feed-Forward + Residual
            TS-->>CT: transformierte_vektoren
        end
        
        CT-->>SA: kontextuelle_embeddings
        SA->>SA: Gewichteter Durchschnitt (mit Maskierung)
    end
    
    SA->>SA: Berechnet Kosinus-Ähnlichkeiten
    SA-->>Haupt: Ergebnisse
    Haupt-->>Benutzer: Ausgabe mit Visualisierung
```

## 📁 Dateistruktur

```
NaturalLanguageProcessing_Verbessert/
├── einbettung_erweitert.py                    # Hauptskript mit erweiterten Funktionen
├── semantik_analysator.py                     # High-Level Semantic Analyzer
├── wort_embedding/
│   ├── __init__.py                           # Package Initialisierung
│   ├── wort2vec_embedding.py                 # Word2VecEmbedding Hauptklasse
│   ├── skipgram_modell.py                    # SkipGram Neural Network
│   └── skipgram_datensatz.py                 # Dataset mit Padding-Unterstützung
└── transformer_erweitert/
    ├── __init__.py                           # Package Initialisierung
    ├── kontext_transformer.py                # Contextual Transformer
    ├── transformer_schicht.py                # Transformer Layer
    └── mehrkopf_aufmerksamkeit.py            # Multi-Head Attention
```

## 🚀 Schnellstart

### Installation
```bash
# 1. Repository klonen oder Dateien erstellen
git clone <repository-url>
cd NaturalLanguageProcessing_Verbessert

# 2. Abhängigkeiten installieren
pip install torch scikit-learn numpy

# 3. Optionale Visualisierungs-Tools
pip install matplotlib seaborn
```

### Grundlegende Verwendung
```python
from semantik_analysator import SemanticAnalyzer
from wort_embedding.wort2vec_embedding import Word2VecEmbedding
from transformer_erweitert.kontext_transformer import ContextualTransformer

# 1. Korpus vorbereiten
korpus = "Der Hund bellt. Die Katze schnurrt. Vögel zwitschern."
saetze = [s.strip() for s in korpus.split(".") if s.strip()]

# 2. Word2Vec Embeddings trainieren
wort_embedder = Word2VecEmbedding(
    text_korpus=korpus,
    kontext_fenster=5,
    embedding_dimension=64,
    max_satzelemente=50
)
wort_embedder.trainiere(epochen=10)

# 3. Transformer initialisieren
transformer = ContextualTransformer(
    wort_embedder=wort_embedder,
    embedding_dimension=64,
    anzahl_koepfe=4,
    anzahl_schichten=2
)

# 4. Semantic Analyzer erstellen
analysator = SemanticAnalyzer(wort_embedder, transformer)

# 5. Analysen durchführen
analysator.berechne_satz_aehnlichkeiten(saetze, "lauter Hund")
analysator.berechne_wort_aehnlichkeiten([("Hund", "Katze"), ("Katze", "Tier")])
```

## ⚙️ Konfiguration

### Word2VecEmbedding Parameter
| Parameter | Typ | Standard | Beschreibung |
|-----------|-----|----------|--------------|
| `text_korpus` | str | - | Eingabetext für Training |
| `kontext_fenster` | int | 5 | Größe des Kontextfensters |
| `embedding_dimension` | int | 64 | Dimension der Word Embeddings |
| `max_satzelemente` | int | 50 | Maximale Satzlänge für Padding |
| `use_stopwords` | bool | False | Deutsche Stopwörter filtern |
| `min_wort_haeufigkeit` | int | 1 | Minimale Wortfrequenz |
| `negative_samples` | int | 5 | Negative Sampling Anzahl |

### ContextualTransformer Parameter
| Parameter | Typ | Standard | Beschreibung |
|-----------|-----|----------|--------------|
| `embedding_dimension` | int | 64 | Eingabe-/Ausgabedimension |
| `anzahl_koepfe` | int | 4 | Anzahl Aufmerksamkeitsköpfe |
| `anzahl_schichten` | int | 2 | Anzahl Transformer-Schichten |
| `maximale_sequenz_laenge` | int | 100 | Maximale Sequenzlänge |
| `dropout_rate` | float | 0.1 | Dropout für Regularisierung |
| `use_pre_norm` | bool | True | Pre-Normalization aktivieren |

## 📊 Datenfluss-Diagramm

```mermaid
flowchart TD
    A[Rohtext Eingabe] --> B[Tokenisierung]
    B --> C{Wort im Vokabular?}
    C -->|Ja| D[Wort Index]
    C -->|Nein| E[OOV Behandlung]
    D --> F[Word2Vec Embedding]
    E --> F
    F --> G[Positionskodierung +]
    G --> H[Multi-Head Attention]
    H --> I[Feed-Forward Netzwerk]
    I --> J{Layer Normalization}
    J -->|Repeat N times| H
    J --> K[Kontextuelles Embedding]
    K --> L[Satz-Vektor<br/>gewichteter Durchschnitt]
    K --> M[Wort-Vektor<br/>direkte Extraktion]
    L --> N[Ähnlichkeitsberechnung]
    M --> N
    N --> O[Ergebnisausgabe]
```

## 🎯 Anwendungsbeispiele

### 1. Semantische Suche
```python
# Finde ähnliche Sätze
top_saetze = analysator.finde_aehnlichste_saetze(
    such_text="Tiere in der Natur",
    saetze=saetze_liste,
    anzahl=5
)

for satz, aehnlichkeit in top_saetze:
    print(f"{aehnlichkeit:.3f}: {satz}")
```

### 2. Dokument-Embeddings
```python
# Berechne Embedding für ein gesamtes Dokument
dokument = "Langer Text mit mehreren Sätzen..."
dokument_embedding = analysator.berechne_dokument_embedding(dokument)
print(f"Dokument-Embedding Shape: {dokument_embedding.shape}")
```

### 3. Wort-Analogien
```python
# Finde semantisch ähnliche Wörter
nachbarn = wort_embedder.finde_semantische_nachbarn(
    wort="König",
    top_k=10,
    min_aehnlichkeit=0.5
)
```

### 4. Visualisierung
```python
# Visualisiere Wort-Embeddings
wort_embedder.visualisiere_embeddings(
    woerter=["Hund", "Katze", "Auto", "Haus", "Baum"],
    methode="pca",
    titel="Deutsche Wort-Embeddings"
)
```

## 🔧 Erweiterte Features

### 1. Variable Satzlängen
```python
# Das System verarbeitet automatisch unterschiedliche Satzlängen
saetze_unterschiedlich = [
    "Kurz.",  # 1 Wort
    "Der Hund bellt laut.",  # 4 Wörter
    "Die schnelle braune Katze springt über den faulen Hund."  # 9 Wörter
]
```

### 2. Batch-Verarbeitung
```python
# Effiziente Verarbeitung mehrerer Sätze
embeddings = []
for satz in saetze_batch:
    indices = wort_embedder.erhalte_indizes_von_satz(satz)
    embedding = transformer(indices.unsqueeze(0))
    embeddings.append(embedding)
```

### 3. Export-Funktionen
```python
# Exportiere trainierte Embeddings
wort_embedder.exportiere_embeddings(
    datei_pfad="meine_embeddings",
    format="json"  # txt, csv, oder json
)
```

## 📈 Performance-Metriken

### Training Performance
| Metrik | Wert | Beschreibung |
|--------|------|--------------|
| Trainingsepochen | 10-50 | Abhängig von Korpusgröße |
| Batch-Größe | 32-128 | Optimierbar für RAM |
| Embedding-Dimension | 64-300 | Höhere Dimension = bessere Qualität |
| Kontextfenster | 2-10 | Typisch: 5 für Word2Vec |

### Inferenz Performance
| Operation | Komplexität | Beispiel-Laufzeit |
|-----------|-------------|-------------------|
| Satz-Embedding | O(n × d) | ~1ms pro Satz |
| Wort-Embedding | O(d) | ~0.1ms pro Wort |
| Ähnlichkeitsberechnung | O(d) | ~0.5ms pro Vergleich |

## 🐛 Fehlerbehandlung

Das System enthält umfangreiche Fehlerbehandlung für:
- **Unbekannte Wörter**: Automatische Fallback-Strategien
- **Leere Eingaben**: Nullvektor-Rückgabe
- **Lange Sätze**: Intelligentes Truncating
- **Speicherüberlauf**: Batch-Verarbeitung mit Progress-Anzeige

```python
try:
    vektor = analysator._erstelle_wort_vektor("UnbekanntesWort")
    if torch.norm(vektor) == 0:
        print("⚠️  Wort nicht im Vokabular")
except Exception as e:
    print(f"❌ Fehler: {str(e)}")
```

## 📚 Pädagogischer Wert

Dieses Projekt demonstriert:

### 1. Moderne NLP-Konzepte
- Word2Vec mit Skip-Gram Architektur
- Transformer mit Self-Attention
- Kontextuelle vs. statische Embeddings

### 2. Software-Engineering Prinzipien
- Clean Code mit deutschen Kommentaren
- Objektorientiertes Design
- Erweiterbarkeit und Wartbarkeit

### 3. Maschinelles Lernen
- Training von neuronalen Netzen
- Hyperparameter-Optimierung
- Evaluation und Validierung

## 🚀 Nächste Schritte

### Geplante Erweiterungen
1. **Multilingual Support**: Mehrsprachige Embeddings
2. **GPU Acceleration**: CUDA-Unterstützung für Training
3. **Pretrained Models**: Integration von BERT/GloVe
4. **REST API**: Web-Schnittstelle für das Modell
5. **Fine-Tuning**: Domain-spezifisches Training

### Forschungsanwendungen
- Semantische Textähnlichkeit
- Dokumentenklassifikation
- Sentiment-Analyse
- Frage-Antwort-Systeme

## 📞 Support und Beiträge

### Bekannte Probleme
1. **Große Korpora**: Benötigt ausreichend RAM
2. **Sehr lange Sätze**: Werden auf `max_satzelemente` gekürzt
3. **Seltene Wörter**: Können schlechte Embeddings haben

### Beitragen
1. Fork das Repository
2. Erstelle einen Feature-Branch
3. Commit deine Änderungen
4. Push zum Branch
5. Erstelle einen Pull Request

## 📄 Lizenz

Dieses Projekt steht unter der MIT-Lizenz.

## 🙏 Danksagung

- **PyTorch Team**: Für das ausgezeichnete Deep Learning Framework
- **Word2Vec Autoren**: Für die bahnbrechende Embedding-Technik
- **Transformer Autoren**: Für die Aufmerksamkeits-Architektur

---

**Hinweis**: Dieses Projekt ist für Bildungs- und Forschungszwecke konzipiert. Für Produktionseinsatz sind zusätzliche Optimierungen und Tests empfohlen.

**Letztes Update**: Dezember 2024  
**Version**: 2.0.0  
**Autor**: KI-Assistent  
**Kompatibilität**: Python 3.7+, PyTorch 1.9+