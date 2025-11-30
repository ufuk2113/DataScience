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
    pass
  

if __name__ == "__main__":
    main()