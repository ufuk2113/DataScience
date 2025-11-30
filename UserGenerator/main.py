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