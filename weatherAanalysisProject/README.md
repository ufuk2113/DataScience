# Weather Analysis Project

## 📊 Projektübersicht

Dieses Data-Science-Projekt analysiert Wetterdaten, um Entscheidungen über das "Draußen Essen" bzw. "Aussenverkauf" basierend auf verschiedenen Wetterbedingungen zu unterstützen. Die Analyse umfasst die Berechnung des Informationsgewinns, statistische Auswertungen und Kategorisierung numerischer Daten.

## 🎯 Aufgabenstellung

### Hauptziele:
1. **Tabellenblatt 1 (Kategorische Daten)**: Bestimmung des Informationsgewinns der Features bezüglich der Klasse "Draussen Essen"
2. **Tabellenblatt 2 (Numerische Daten)**: 
   - Kategorisierung der numerischen Daten
   - Berechnung des Informationsgewinns bezüglich "Aussenverkauf"
   - Statistische Analyse der Features

### Datenkategorisierung (Tabellenblatt 2):

#### Windgeschwindigkeit:
- 1-5 km/h
- 6-11 km/h
- 12-19 km/h
- >19 km/h

#### Temperatur:
- < 18°C: kalt
- 18-28°C: mild
- > 28°C: heiss

#### Luftfeuchtigkeit:
- > 5: hoch
- ≤ 5: niedrig

### Statistische Analyse (Tabellenblatt 2):
- Mittelwert aller Features
- Median aller Features
- Standardabweichung aller Features
- Korrelationen zwischen:
  - Luftfeuchtigkeit und Temperatur
  - Windgeschwindigkeit und Luftfeuchtigkeit
  - Windgeschwindigkeit und Temperatur

## 📁 Projektstruktur

```
weather_analysis_project/
├── weather_analysis_project.py    # Hauptanalyse-Skript
├── requirements.txt               # Python-Abhängigkeiten
├── README.md                      # Diese Datei
└── K4.0026_1.5.C.01_ProjectData.xlsx  # Datendatei (Excel)
```

## 🏗️ UML-Struktur

### Klasse: DataAnalyzer

```
┌─────────────────────────────────────────────────────────────────┐
│                        DataAnalyzer                             │
├─────────────────────────────────────────────────────────────────┤
│ - file_path: str                                                │
│ - df1: DataFrame                                                │
│ - df2: DataFrame                                                │
│ - df2_categorized: DataFrame                                    │
├─────────────────────────────────────────────────────────────────┤
│ + __init__(file_path: str)                                      │
│ + load_data()                                                   │
│ + calculate_information_gain() -> float                         │
│ + categorize_numerical_data()                                   │
│ + analyze_sheet1() -> dict                                      │
│ + analyze_sheet2() -> dict                                      │
│ + perform_statistical_analysis() -> tuple                       │
│ + additional_analyses()                                         │
│ + generate_summary()                                            │
│ + run_complete_analysis()                                       │
└─────────────────────────────────────────────────────────────────┘
```

### Methodenkategorien:

#### Data Management:
- `load_data()`: Lädt Daten aus Excel-Datei
- `categorize_numerical_data()`: Kategorisiert numerische Daten gemäß Aufgabenstellung

#### Analysis:
- `calculate_information_gain()`: Berechnet Informationsgewinn für Features
- `analyze_sheet1()`: Führt Analyse für kategorische Daten durch
- `analyze_sheet2()`: Führt Analyse für numerische Daten durch

#### Statistics:
- `perform_statistical_analysis()`: Berechnet statistische Kennzahlen und Korrelationen
- `additional_analyses()`: Führt zusätzliche Analysen durch

#### Control:
- `run_complete_analysis()`: Steuert den kompletten Analyseprozess
- `generate_summary()`: Erstellt Zusammenfassung der Ergebnisse

## 🛠️ Installation und Einrichtung

### Voraussetzungen:
- Python 3.7 oder höher
- pip (Python Package Manager)

### Installation:

1. Repository klonen oder Dateien herunterladen
2. Abhängigkeiten installieren:

```bash
pip install -r requirements.txt
```

Die benötigten Pakete sind:
- `pandas>=1.3.0`
- `numpy>=1.21.0`
- `openpyxl>=3.0.0`
- `scipy>=1.7.0`

Das Skript installiert fehlende Pakete automatisch beim ersten Start.

## 🚀 Verwendung

### Standardausführung:

```bash
python weather_analysis_project.py
```

### Ausführungsablauf:

1. **Automatische Paketinstallation**: Prüft und installiert benötigte Abhängigkeiten
2. **Datenladen**: Lädt Daten aus beiden Tabellenblättern der Excel-Datei
3. **Datenkategorisierung**: Kategorisiert numerische Daten aus Tabellenblatt 2
4. **Analyse Tabellenblatt 1**: Berechnet Informationsgewinn für kategorische Daten
5. **Analyse Tabellenblatt 2**: Berechnet Informationsgewinn für kategorisierte Daten
6. **Statistische Analyse**: Berechnet Mittelwerte, Mediane, Standardabweichungen und Korrelationen
7. **Zusätzliche Analysen**: Zeigt Verteilungen und Kreuztabellen
8. **Zusammenfassung**: Präsentiert die wichtigsten Erkenntnisse

## 📈 Ausgabe

Das Programm generiert folgende Ausgaben:

### Tabellenblatt 1 Analyse:
- Datenübersicht und Struktur
- Informationsgewinn für jedes Feature
- Identifikation des wichtigsten Features

### Tabellenblatt 2 Analyse:
- Original- und kategorisierte Datenübersicht
- Informationsgewinn für kategorisierte Features
- Identifikation des wichtigsten Features

### Statistische Analyse:
- Mittelwert, Median und Standardabweichung für Temperatur, Luftfeuchtigkeit und Windgeschwindigkeit
- Korrelationskoeffizienten zwischen den Features

### Zusätzliche Analysen:
- Verteilung der Kategorien
- Kreuztabellen für Aussenverkauf nach Kategorien

### Zusammenfassung:
- Wichtigste Features aus beiden Tabellenblättern
- Stärkste Korrelation
- Interpretation der Ergebnisse

## 📝 Code-Beispiel

```python
# Hauptprogramm ausführen
def main():
    file_path = 'K4.0026_1.5.C.01_ProjectData.xlsx'
    analyzer = DataAnalyzer(file_path)
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()
```

## 🔧 Anpassungen

### Excel-Datei-Pfad anpassen:
```python
# In der main()-Funktion:
file_path = 'IHR_PFAD/ZUR_DATEI.xlsx'
```

### Zusätzliche Features analysieren:
- Features in `analyze_sheet1()`: Liste `features_tb1` erweitern
- Features in `analyze_sheet2()`: Liste `features_tb2` erweitern

## 📊 Interpretation der Ergebnisse

### Informationsgewinn:
- **> 0**: Feature ist relevant für die Entscheidung
- **Höherer Wert**: Größerer Einfluss auf die Entscheidung
- **0**: Kein Informationsgewinn (unabhängig von der Zielvariable)

### Korrelation:
- **Nahe 0**: Kein linearer Zusammenhang
- **Nahe +1**: Starker positiver linearer Zusammenhang
- **Nahe -1**: Starker negativer linearer Zusammenhang

## 🧪 Testdaten

Die Excel-Datei enthält zwei Tabellenblätter:

### Tabellenblatt 1:
- 14 Datensätze mit kategorischen Daten
- Features: Wetteraussicht, Temperaturkategorie, Luftfeuchtigkeit, Windstärke
- Zielvariable: Draussen Essen (ja/nein)

### Tabellenblatt 2:
- 115 Datensätze mit numerischen Daten
- Features: Temperatur, Luftfeuchtigkeit, Windgeschwindigkeit
- Zielvariable: Aussenverkauf (ja/nein)

## 📄 Lizenz

Dieses Projekt wurde für akademische Zwecke erstellt.

## 👥 Autor

**Ufuk Baysal**
- Datum: 21.10.2025
- Projekt: Weather Analysis Project

---

*Hinweis: Stellen Sie sicher, dass die Excel-Datei im selben Verzeichnis wie das Python-Skript liegt oder passen Sie den Pfad entsprechend an.*