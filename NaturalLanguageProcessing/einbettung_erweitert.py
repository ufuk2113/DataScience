#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Einbettung Erweitert - Hauptskript für erweiterte NLP-Funktionen

Dieses Skript demonstriert die erweiterten Funktionen des neuen NLP-Systems:
- Unterstützung für variable Satzlängen
- Batch-Verarbeitung mit Padding
- Erweiterte Ähnlichkeitsberechnungen
- Visualisierungsmöglichkeiten

Autor: Ufuk Baysal
Datum: 2025

"""

import sys
import os
import time
import numpy as np
from typing import List, Tuple, Dict, Optional

# Pfadanpassung für Importe
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from semantik_analysator import SemanticAnalyzer
from wort_embedding.wort2vec_embedding import Word2VecEmbedding
from transformer_erweitert.kontext_transformer import ContextualTransformer


def main():
    """Hauptfunktion für erweiterte NLP-Demonstration."""
    
    print("=" * 80)
    print("ERWEITERTES NLP-SYSTEM - DEMONSTRATION")
    print("=" * 80)
    
    # 1. KORPUS VORBEREITEN (mit variablen Satzlängen)
    print("\n1. KORPUS VORBEREITUNG")
    print("-" * 40)
    
    korpus: str = """Der schnelle braune Fuchs springt über den faulen Hund. 
    Die Katze sitzt auf der Matte und schnurrt leise vor sich hin. 
    Vögel zwitschern fröhlich in den Bäumen des großen Waldes. 
    Autos fahren laut hupend über die belebte Stadtstraße. 
    Hunde und Katzen sind beliebte Haustiere in deutschen Haushalten. 
    Der Regen fällt sanft auf die Dächer der alten Gebäude. 
    Kinder spielen fröhlich im Park mit bunten Bällen und springen Seil. 
    Die Sonne scheint hell am blauen Himmel und wärmt die Erde. 
    Bücher stehen ordentlich sortiert in den Regalen der großen Bibliothek. 
    Computer verarbeiten Daten mit hoher Geschwindigkeit und Präzision."""
    
    # Sätze extrahieren (unterschiedliche Längen)
    saetze: List[str] = [s.strip() for s in korpus.split(".") if len(s.strip()) > 0]
    
    print(f"Anzahl Sätze: {len(saetze)}")
    print(f"Satzlängen: {[len(s.split()) for s in saetze]}")
    print(f"Korpusgröße: {len(korpus.split())} Wörter")
    
    # 2. WORD2VEC EMBEDDING TRAINIEREN
    print("\n\n2. WORD2VEC EMBEDDING TRAINING")
    print("-" * 40)
    
    start_zeit = time.perf_counter()
    
    # Word2Vec mit erweiterten Parametern initialisieren
    wort_embedder = Word2VecEmbedding(
        text_korpus=korpus,
        kontext_fenster=5,
        embedding_dimension=64,  # Höhere Dimension für bessere Repräsentation
        max_satzelemente=50,     # Maximale Satzlänge für Padding
        use_stopwords=False      # Stopwort-Filterung deaktivieren
    )
    
    print(f"Wörterbuch-Größe: {wort_embedder.woerterbuch_groesse}")
    print(f"Embedding-Dimension: {wort_embedder.embedding_dimension}")
    print(f"Anzahl Trainingspaare: {len(wort_embedder.trainings_paare)}")
    
    # Training mit Progress-Anzeige
    wort_embedder.trainiere(
        epochen=15,
        batch_groesse=32,
        lernrate=0.01,
        zeige_progress=True
    )
    
    # 3. CONTEXTUAL TRANSFORMER INITIALISIEREN
    print("\n\n3. CONTEXTUAL TRANSFORMER INITIALISIERUNG")
    print("-" * 40)
    
    kontext_transformer = ContextualTransformer(
        wort_embedder=wort_embedder,
        embedding_dimension=64,
        anzahl_koepfe=4,          # Mehr Aufmerksamkeitsköpfe
        anzahl_schichten=2,       # Mehr Transformer-Schichten
        maximale_sequenz_laenge=100,
        dropout_rate=0.1          # Regularisierung
    )
    
    print(f"Transformer Parameter: {sum(p.numel() for p in kontext_transformer.parameters()):,}")
    print(f"Aufmerksamkeitsköpfe: {kontext_transformer.anzahl_koepfe}")
    print(f"Transformer-Schichten: {kontext_transformer.anzahl_schichten}")
    
    # 4. SEMANTIC ANALYZER ERSTELLEN
    print("\n\n4. SEMANTIC ANALYZER ERSTELLEN")
    print("-" * 40)
    
    semantik_analysator = SemanticAnalyzer(
        wort_embedder=wort_embedder,
        kontext_transformer=kontext_transformer,
        embedding_dimension=64
    )
    
    # 5. ERWEITERTE ANALYSEN DURCHFÜHREN
    print("\n\n5. ERWEITERTE SEMANTISCHE ANALYSEN")
    print("-" * 40)
    
    # 5.1 Satzähnlichkeiten mit variablem Suchtext
    suchtexte = ["lautes Auto", "fröhliche Tiere", "große Gebäude"]
    
    for suchtext in suchtexte:
        print(f"\n🔍 Suche nach: '{suchtext}'")
        print("-" * 40)
        
        semantik_analysator.berechne_satz_aehnlichkeiten(
            saetze=saetze,
            such_text=suchtext,
            zeige_top_n=3,          # Zeige Top 3 Ergebnisse
            min_aehnlichkeit=0.3    # Minimale Ähnlichkeit
        )
    
    # 5.2 Wortähnlichkeiten mit erweiterter Analyse
    print("\n\n📊 WORTÄHNLICHKEITEN - DETAILLIERTE ANALYSE")
    print("=" * 60)
    
    wort_paare = [
        ("Hund", "Katze"),
        ("Auto", "Fahrzeug"),
        ("Haus", "Gebäude"),
        ("schnell", "langsam"),
        ("groß", "klein"),
        ("Sonne", "Mond"),
        ("Buch", "Bibliothek"),
        ("Kind", "Erwachsener")
    ]
    
    semantik_analysator.berechne_wort_aehnlichkeiten(
        wort_paare=wort_paare,
        zeige_embedding_stats=True
    )
    
    # 5.3 Finde ähnlichste Sätze (Top-N Suche)
    print("\n\n🎯 TOP-N ÄHNLICHSTE SÄTZE FINDEN")
    print("=" * 60)
    
    such_beispiele = [
        "Tiere in der Natur",
        "Verkehr in der Stadt",
        "Aktivitäten von Menschen"
    ]
    
    for beispiel in such_beispiele:
        print(f"\nTop 3 Sätze ähnlich zu: '{beispiel}'")
        print("-" * 40)
        
        top_saetze = semantik_analysator.finde_aehnlichste_saetze(
            such_text=beispiel,
            saetze=saetze,
            anzahl=3
        )
        
        for i, (satz, aehnlichkeit) in enumerate(top_saetze, 1):
            print(f"{i}. [{aehnlichkeit:.3f}] {satz}")
    
    # 6. PERFORMANCE-METRIKEN
    print("\n\n⏱️  PERFORMANCE-METRIKEN")
    print("=" * 60)
    
    ende_zeit = time.perf_counter()
    gesamtzeit = ende_zeit - start_zeit
    
    print(f"Gesamtlaufzeit: {gesamtzeit:.2f} Sekunden")
    print(f"Verarbeitete Wörter: {len(wort_embedder.woerter)}")
    print(f"Embedding-Dimension: {wort_embedder.embedding_dimension}")
    print(f"Transformer Parameter: {sum(p.numel() for p in kontext_transformer.parameters()):,}")
    
    # 7. SPEZIELLE FEHLERBEHANDLUNG UND TESTS
    print("\n\n🧪 SPEZIELLE TESTS UND FEHLERHANDLUNG")
    print("=" * 60)
    
    # Test: Unbekannte Wörter
    print("\nTest mit unbekannten Wörtern:")
    unbekannte_woerter = ["Dinosaurier", "Quantencomputer", "Xylophon"]
    for wort in unbekannte_woerter:
        try:
            vektor = semantik_analysator._erstelle_wort_vektor(wort)
            print(f"  '{wort}': {'Erfolgreich' if vektor.sum() != 0 else 'Nicht gefunden'}")
        except Exception as e:
            print(f"  '{wort}': Fehler - {str(e)[:50]}...")
    
    # Test: Leere Sätze
    print("\nTest mit leeren Sätzen:")
    leere_saetze = ["", "   ", "..."]
    for satz in leere_saetze:
        try:
            vektor = semantik_analysator._erstelle_satz_vektor(satz)
            print(f"  Leerer Satz '{satz[:10]}...': Vektor erstellt")
        except Exception as e:
            print(f"  Leerer Satz: Fehler - {str(e)[:50]}...")
    
    print("\n" + "=" * 80)
    print("PROGRAMM ERFOLGREICH ABGESCHLOSSEN")
    print("=" * 80)


if __name__ == "__main__":
    main()