"""
main.py - ML-Testdatengenerator für deutsche Nutzerprofile

Dieses Skript erzeugt realistische deutsche Nutzerdaten für Machine-Learning-Tests.
Es demonstriert verschiedene Anwendungsfälle und Exportformate.

Verwendung:
    python main.py
"""

import pandas as pd
from user_generator import UserGenerator
import argparse
import sys
import os


def generate_ml_dataset(user_count=1000, export_all_formats=False):
    """
    Erzeugt einen umfangreichen ML-Datensatz mit deutschen Nutzerprofilen.
    
    Args:
        user_count (int): Anzahl der zu generierenden Nutzer
        export_all_formats (bool): Ob alle Exportformate erzeugt werden sollen
    
    Returns:
        pd.DataFrame: Generierter Datensatz
    """
    
    print(f"🚀 Starte Generierung von {user_count} deutschen Nutzerprofilen...")
    
    # Generator initialisieren
    generator = UserGenerator()
    
    # Datensatz erstellen
    df = generator.create_dataset(user_count)
    
    # Datenqualitäts-Checks
    print(f"✅ Datensatz erfolgreich generiert:")
    print(f"   - Größe: {len(df)} Zeilen × {len(df.columns)} Spalten")
    print(f"   - Alter: {df['age'].min()} - {df['age'].max()} Jahre")
    print(f"   - Einkommen: {df['income'].min()} - {df['income'].max()} €")
    print(f"   - Städte: {df['city'].nunique()} verschiedene Städte")
    print(f"   - Kategorien: {df['category'].value_counts().to_dict()}")
    
    return df, generator


def export_for_ml_analysis(df, generator, base_filename="ml_users"):
    """
    Exportiert den Datensatz in verschiedene Formate für ML-Analysen.
    
    Args:
        df (pd.DataFrame): Der zu exportierende Datensatz
        generator (UserGenerator): Generator-Instanz
        base_filename (str): Basis-Dateiname für Exporte
    """
    
    print(f"\n📊 Exportiere Daten für ML-Analyse...")
    
    # 1. CSV (primär für ML - am weitesten verbreitet)
    csv_file = f"{base_filename}.csv"
    generator.export_csv(df, csv_file)
    print(f"   ✅ CSV: {csv_file}")
    
    # 2. JSON (für strukturierte Daten und APIs)
    json_file = f"{base_filename}.json"
    generator.export_json(df, json_file)
    print(f"   ✅ JSON: {json_file}")
    
    # 3. Excel (für manuelle Datenexploration)
    xlsx_file = f"{base_filename}.xlsx"
    generator.export_xlsx(df, xlsx_file)
    print(f"   ✅ Excel: {xlsx_file}")
    
    # 4. SQL (für Datenbank-Import)
    sql_file = f"{base_filename}.sql"
    generator.export_sql(df, sql_file)
    print(f"   ✅ SQL: {sql_file}")
    
    return {
        'csv': csv_file,
        'json': json_file,
        'xlsx': xlsx_file,
        'sql': sql_file
    }


def analyze_ml_features(df):
    """
    Analysiert die generierten Features für Machine Learning.
    
    Args:
        df (pd.DataFrame): Der zu analysierende Datensatz
    """
    
    print(f"\n🔍 ML-Feature-Analyse:")
    
    # Numerische Features
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    print(f"   📈 Numerische Features ({len(numeric_features)}): {numeric_features}")
    
    # Kategorische Features
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    print(f"   📊 Kategorische Features ({len(categorical_features)}): {categorical_features}")
    
    # Feature-Statistiken
    print(f"\n   📋 Feature-Übersicht:")
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            print(f"      - {col}: {df[col].min()} - {df[col].max()} (Mittelwert: {df[col].mean():.2f})")
        else:
            unique_vals = df[col].nunique()
            sample_vals = df[col].value_counts().head(3).to_dict()
            print(f"      - {col}: {unique_vals} eindeutige Werte (Top: {sample_vals})")


def demonstrate_ml_use_cases(df):
    """
    Demonstriert verschiedene ML-Anwendungsfälle mit den generierten Daten.
    
    Args:
        df (pd.DataFrame): Der generierte Datensatz
    """
    
    print(f"\n🎯 ML-Anwendungsfälle mit diesen Daten:")
    
    # 1. Klassifikation: Nutzerkategorie vorhersagen
    print(f"   1. 📊 Klassifikation: Nutzerkategorie vorhersagen")
    print(f"      - Features: Alter, Einkommen, Haushaltsgröße, Beruf, Ort")
    print(f"      - Target: 'category' (Basic, Premium, Business, VIP)")
    print(f"      - Algorithmen: Random Forest, XGBoost, SVM")
    
    # 2. Regression: Einkommen vorhersagen
    print(f"   2. 📈 Regression: Einkommen vorhersagen")
    print(f"      - Features: Alter, Beruf, Firma, Familienstand, Ort")
    print(f"      - Target: 'income' (kontinuierlich)")
    print(f"      - Algorithmen: Linear Regression, Gradient Boosting")
    
    # 3. Clustering: Nutzergruppen identifizieren
    print(f"   3. 🎯 Clustering: Ähnliche Nutzergruppen finden")
    print(f"      - Features: Alle demografischen und geografischen Daten")
    print(f"      - Algorithmen: K-Means, DBSCAN, Hierarchical Clustering")
    
    # 4. Anomalie-Erkennung: Ungewöhnliche Profile finden
    print(f"   4. 🔍 Anomalie-Erkennung: Ungewöhnliche Nutzerprofile")
    print(f"      - Features: Alle numerischen Daten")
    print(f"      - Algorithmen: Isolation Forest, Local Outlier Factor")
    
    # 5. Geospatial Analysis: Regionale Muster erkennen
    print(f"   5. 🗺️  Geospatial Analysis: Regionale Verteilungen")
    print(f"      - Features: latitude, longitude, city")
    print(f"      - Tools: GeoPandas, Folium für Visualisierung")


def create_sample_datasets():
    """
    Erzeugt verschiedene Beispiel-Datensätze für unterschiedliche ML-Szenarien.
    """
    
    scenarios = {
        "small_quick_test": {
            "count": 100,
            "desc": "Kleiner Datensatz für schnelle Tests und Prototyping"
        },
        "medium_validation": {
            "count": 1000, 
            "desc": "Mittlerer Datensatz für Modellvalidierung"
        },
        "large_production": {
            "count": 10000,
            "desc": "Großer Datensatz für Produktionsmodelle"
        }
    }
    
    print(f"\n🎪 Erzeuge Beispiel-Datensätze für verschiedene ML-Szenarien:")
    
    for scenario_name, config in scenarios.items():
        print(f"\n   📂 Szenario: {scenario_name}")
        print(f"      {config['desc']}")
        print(f"      Generiere {config['count']} Nutzer...")
        
        df, generator = generate_ml_dataset(config['count'])
        export_for_ml_analysis(df, generator, f"ml_dataset_{scenario_name}")
        
        print(f"      ✅ {scenario_name} komplett!")


def main():
    """
    Hauptfunktion - Koordiniert die Datengenerierung und Exporte.
    """
    
    # Argument-Parser für Kommandozeilen-Interface
    parser = argparse.ArgumentParser(description='Generiere ML-Testdaten für deutsche Nutzerprofile')
    parser.add_argument('--count', type=int, default=1000, 
                       help='Anzahl der zu generierenden Nutzer (default: 1000)')
    parser.add_argument('--scenarios', action='store_true',
                       help='Erzeuge verschiedene Beispiel-Datensätze')
    parser.add_argument('--analyze', action='store_true', 
                       help='Führe detaillierte Feature-Analyse durch')
    
    args = parser.parse_args()
    
    try:
        print("=" * 70)
        print("🤖 ML-TESTDATENGENERATOR - DEUTSCHE NUTZERPROFILE")
        print("=" * 70)
        
        if args.scenarios:
            # Verschiedene Beispiel-Datensätze erzeugen
            create_sample_datasets()
        else:
            # Einzelnen Datensatz erzeugen
            df, generator = generate_ml_dataset(args.count)
            
            # Daten exportieren
            export_for_ml_analysis(df, generator)
            
            if args.analyze:
                # Detaillierte Analyse
                analyze_ml_features(df)
                demonstrate_ml_use_cases(df)
        
        print(f"\n🎉 Alle Aufgaben erfolgreich abgeschlossen!")
        print(f"💡 Tipp: Verwende --scenarios für verschiedene Datensatz-Größen")
        print(f"💡 Tipp: Verwende --analyze für detaillierte Feature-Analyse")
        
    except Exception as e:
        print(f"❌ Fehler: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()