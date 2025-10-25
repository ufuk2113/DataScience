# Weather Analysis Project - Dokumentation

## UML Klassendiagramm

### DataAnalyzer Klasse

```plaintext
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
│ + calculate_information_gain(data: DataFrame,                   │
│   feature: str, target: str) -> float                           │
│ + categorize_numerical_data()                                   │
│ + analyze_sheet1() -> dict                                      │
│ + analyze_sheet2() -> dict                                      │
│ + perform_statistical_analysis() -> tuple                       │
│ + additional_analyses()                                         │
│ + generate_summary(ig_tb1: dict, ig_tb2: dict,                  │
│   correlations: dict)                                           │
│ + run_complete_analysis()                                       │
└─────────────────────────────────────────────────────────────────┘

DataAnalyzer
├── Attribute
│   ├── file_path: str
│   ├── df1: pd.DataFrame
│   ├── df2: pd.DataFrame
│   └── df2_categorized: pd.DataFrame
│
├── Datenmanagement
│   ├── load_data()
│   └── categorize_numerical_data()
│
├── Analyse-Methoden
│   ├── calculate_information_gain()
│   ├── analyze_sheet1()
│   ├── analyze_sheet2()
│   └── perform_statistical_analysis()
│
└── Steuerungs-Methoden
    ├── run_complete_analysis()
    └── generate_summary()

```
