#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FileBasedEmbeddingExporter - Universal Klasse für Embedding Export aus Dateien

flexiblen Lösung für Embedding Training und Export aus verschiedenen Dateiformaten.

Merkmale:
- Liest aus TXT, JSON, CSV, MD Dateien
- Unterstützt Einzeldateien und Verzeichnisse
- Automatische Textbereinigung
- Kommandozeilen- und Programm-API
- Umfassende Fehlerbehandlung

Autor: Ufuk Baysal
Datum: 2026
"""

import sys
import os
import argparse
import json
import csv
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
import re

# Pfad für Importe anpassen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from wort_embedding.wort2vec_embedding import Word2VecEmbedding
    from transformer_erweitert.kontext_transformer import ContextualTransformer
    from semantik_analysator import SemanticAnalyzer
    IMPORT_SUCCESS = True
except ImportError as e:
    print(f"⚠️  Import-Warnung: {e}")
    print("   Stelle sicher, dass alle Module vorhanden sind.")
    IMPORT_SUCCESS = False


class FileBasedEmbeddingExporter:
    """
    Universal-Klasse für Embedding Training und Export aus Dateien.
    
    Diese Klasse kombiniert alle Funktionen für:
    - Lesen verschiedener Dateiformate (TXT, JSON, CSV, MD)
    - Batch-Verarbeitung von Verzeichnissen
    - Automatische Textbereinigung
    - Word2Vec Training mit konfigurierbaren Parametern
    - Export in multiple Formate (TXT, JSON, CSV)
    - Erweiterte NLP-Analysen (optional)
    
    UML-Beziehung:
    FileBasedEmbeddingExporter ── verwendet ──▶ Word2VecEmbedding
                                      │
                                      └── verwendet ──▶ ContextualTransformer (optional)
                                              │
                                              └── verwendet ──▶ SemanticAnalyzer (optional)
    
    Beispiel:
        >>> exporter = FileBasedEmbeddingExporter()
        >>> exporter.export_from_file("mein_korpus.txt", output_name="meine_embeddings")
    """
    
    # Klassenvariablen für Default-Konfiguration
    DEFAULT_CONFIG = {
        'training': {
            'epochs': 15,
            'embedding_dim': 64,
            'window_size': 5,
            'max_seq_length': 100,
            'batch_size': 32,
            'learning_rate': 0.01
        },
        'preprocessing': {
            'min_sentence_length': 3,
            'max_sentence_length': 100,
            'remove_stopwords': False,
            'clean_text': True,
            'encoding': 'utf-8'
        },
        'export': {
            'formats': ['txt', 'json'],
            'include_metadata': True,
            'timestamp_in_name': True
        }
    }
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialisiert den FileBasedEmbeddingExporter.
        
        Args:
            config: Optionales Konfigurations-Dictionary
        
        Beispiel:
            >>> config = {'training': {'epochs': 20}}
            >>> exporter = FileBasedEmbeddingExporter(config)
        """
        if not IMPORT_SUCCESS:
            raise ImportError("Erforderliche Module können nicht importiert werden.")
        
        # Konfiguration mergen
        self.config = self.DEFAULT_CONFIG.copy()
        if config:
            self._merge_config(config)
        
        # Zustandsvariablen
        self.word_embedder = None
        self.context_transformer = None
        self.semantic_analyzer = None
        self.corpus_text = ""
        self.stats = {
            'files_processed': 0,
            'total_words': 0,
            'vocabulary_size': 0,
            'training_time': 0,
            'export_files': []
        }
        
        print("✅ FileBasedEmbeddingExporter initialisiert")
    
    def _merge_config(self, user_config: Dict) -> None:
        """
        Merge Benutzerkonfiguration mit Default-Konfiguration.
        
        Args:
            user_config: Benutzer-Konfiguration
        
        UML-Methode: Private Hilfsmethode
        """
        def recursive_merge(base, user):
            for key, value in user.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    recursive_merge(base[key], value)
                else:
                    base[key] = value
        
        recursive_merge(self.config, user_config)
    
    def read_text_from_file(self, file_path: str, encoding: str = None) -> str:
        """
        Liest Text aus einer einzelnen Datei.
        
        Unterstützte Formate: .txt, .md, .json, .csv
        
        Args:
            file_path: Pfad zur Datei
            encoding: Datei-Encoding (None = auto-detect)
        
        Returns:
            Extrahierter Text
        
        Raises:
            FileNotFoundError: Wenn Datei nicht existiert
            ValueError: Wenn Datei leer ist
        
        UML-Methode: Public API Methode
        """
        if not encoding:
            encoding = self.config['preprocessing']['encoding']
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Datei nicht gefunden: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext in ['.txt', '.md']:
                text = self._read_txt_file(file_path, encoding)
            elif file_ext == '.json':
                text = self._read_json_file(file_path, encoding)
            elif file_ext == '.csv':
                text = self._read_csv_file(file_path, encoding)
            else:
                # Unbekanntes Format: Versuche als Text zu lesen
                text = self._read_txt_file(file_path, encoding)
            
            if not text or not text.strip():
                raise ValueError(f"Datei ist leer: {file_path}")
            
            print(f"✅ Datei gelesen: {os.path.basename(file_path)}")
            print(f"   Format: {file_ext}, Wörter: {len(text.split())}")
            
            return text
            
        except Exception as e:
            print(f"❌ Fehler beim Lesen von {file_path}: {str(e)[:100]}")
            raise
    
    def _read_txt_file(self, file_path: str, encoding: str) -> str:
        """
        Liest Text aus TXT oder MD Datei.
        
        UML-Methode: Private Hilfsmethode
        """
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            return f.read()
    
    def _read_json_file(self, file_path: str, encoding: str) -> str:
        """
        Extrahiert Text aus JSON Datei.
        
        UML-Methode: Private Hilfsmethode
        """
        with open(file_path, 'r', encoding=encoding) as f:
            data = json.load(f)
        
        return self._extract_text_from_json(data)
    
    def _read_csv_file(self, file_path: str, encoding: str) -> str:
        """
        Liest Text aus CSV Datei.
        
        UML-Methode: Private Hilfsmethode
        """
        texts = []
        with open(file_path, 'r', encoding=encoding, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                # Kombiniere alle Zellen der Zeile
                row_text = ' '.join(str(cell) for cell in row if cell)
                if row_text:
                    texts.append(row_text)
        
        return ' '.join(texts)
    
    def _extract_text_from_json(self, data: Any) -> str:
        """
        Extrahiert Text aus JSON-Struktur.
        
        UML-Methode: Private Hilfsmethode
        """
        texts = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str):
                    texts.append(value)
                elif isinstance(value, (dict, list)):
                    texts.append(self._extract_text_from_json(value))
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, (dict, list)):
                    texts.append(self._extract_text_from_json(item))
        elif isinstance(data, str):
            texts.append(data)
        
        return ' '.join(texts)
    
    def read_from_directory(self, directory_path: str, 
                          file_extensions: List[str] = None) -> str:
        """
        Liest alle Textdateien aus einem Verzeichnis.
        
        Args:
            directory_path: Pfad zum Verzeichnis
            file_extensions: Liste von Dateiendungen (z.B. ['.txt', '.md'])
        
        Returns:
            Kombinierter Text aller Dateien
        
        UML-Methode: Public API Methode
        """
        if not file_extensions:
            file_extensions = ['.txt', '.md', '.json', '.csv']
        
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Verzeichnis nicht gefunden: {directory_path}")
        
        all_texts = []
        processed_files = 0
        
        print(f"📂 Verarbeite Verzeichnis: {directory_path}")
        
        for ext in file_extensions:
            for file_path in Path(directory_path).glob(f"*{ext}"):
                try:
                    text = self.read_text_from_file(str(file_path))
                    all_texts.append(f"\n\n--- Datei: {file_path.name} ---\n\n{text}")
                    processed_files += 1
                    self.stats['files_processed'] += 1
                except Exception as e:
                    print(f"⚠️  Überspringe {file_path.name}: {str(e)[:50]}")
        
        if not all_texts:
            raise ValueError(f"Keine lesbaren Dateien gefunden in {directory_path}")
        
        combined_text = " ".join(all_texts)
        print(f"✅ {processed_files} Dateien verarbeitet, {len(combined_text.split())} Wörter")
        
        return combined_text
    
    def preprocess_text(self, text: str, 
                       min_length: int = None,
                       max_length: int = None) -> str:
        """
        Bereinigt und verarbeitet Text für das Training.
        
        Args:
            text: Rohtext
            min_length: Minimale Satzlänge in Wörtern
            max_length: Maximale Satzlänge in Wörtern
        
        Returns:
            Bereinigter Text
        
        UML-Methode: Public API Methode
        """
        if not self.config['preprocessing']['clean_text']:
            return text
        
        if min_length is None:
            min_length = self.config['preprocessing']['min_sentence_length']
        if max_length is None:
            max_length = self.config['preprocessing']['max_sentence_length']
        
        print("\n🧹 Text-Vorverarbeitung...")
        print(f"   Min. Satzlänge: {min_length} Wörter")
        print(f"   Max. Satzlänge: {max_length} Wörter")
        
        # Sätze extrahieren (einfache Segmentierung)
        sentences = []
        for sentence in re.split(r'[.!?]+', text):
            sentence = sentence.strip()
            if sentence:
                sentences.append(sentence)
        
        original_count = len(sentences)
        
        # Nach Länge filtern
        filtered_sentences = []
        for sentence in sentences:
            word_count = len(sentence.split())
            if min_length <= word_count <= max_length:
                filtered_sentences.append(sentence)
        
        filtered_count = len(filtered_sentences)
        
        print(f"   Sätze vorher: {original_count}")
        print(f"   Sätze nachher: {filtered_count}")
        print(f"   Entfernt: {original_count - filtered_count} Sätze")
        
        # Text wieder zusammensetzen
        cleaned_text = ". ".join(filtered_sentences) + "."
        self.stats['total_words'] = len(cleaned_text.split())
        
        return cleaned_text
    
    def train_embeddings(self, corpus_text: str) -> Word2VecEmbedding:
        """
        Trainiert Word2Vec Embeddings auf dem gegebenen Text.
        
        Args:
            corpus_text: Trainingskorpus
        
        Returns:
            Trainiertes Word2VecEmbedding Objekt
        
        UML-Methode: Public API Methode
        """
        print("\n🏋️  Word2Vec Training startet...")
        
        training_config = self.config['training']
        
        start_time = time.time()
        
        # Word2VecEmbedding initialisieren
        self.word_embedder = Word2VecEmbedding(
            text_korpus=corpus_text,
            kontext_fenster=training_config['window_size'],
            embedding_dimension=training_config['embedding_dim'],
            max_satzelemente=training_config['max_seq_length'],
            use_stopwords=self.config['preprocessing']['remove_stopwords']
        )
        
        print(f"📊 Vokabular: {self.word_embedder.woerterbuch_groesse} Wörter")
        print(f"📊 Trainingspaare: {len(self.word_embedder.trainings_paare)}")
        
        # Training durchführen
        self.word_embedder.trainiere(
            epochen=training_config['epochs'],
            batch_groesse=training_config['batch_size'],
            lernrate=training_config['learning_rate'],
            zeige_progress=True
        )
        
        training_time = time.time() - start_time
        self.stats['training_time'] = training_time
        self.stats['vocabulary_size'] = self.word_embedder.woerterbuch_groesse
        
        print(f"✅ Training abgeschlossen in {training_time:.1f} Sekunden")
        
        return self.word_embedder
    
    def export_embeddings(self, 
                         output_base_name: str,
                         formats: List[str] = None) -> List[str]:
        """
        Exportiert die trainierten Embeddings in verschiedene Formate.
        
        Args:
            output_base_name: Basisname für Ausgabedateien (ohne Endung)
            formats: Liste von Export-Formaten ['txt', 'json', 'csv']
        
        Returns:
            Liste der erstellten Dateipfade
        
        UML-Methode: Public API Methode
        """
        if self.word_embedder is None:
            raise ValueError("Keine trainierten Embeddings vorhanden. Führe zuerst train_embeddings() aus.")
        
        if formats is None:
            formats = self.config['export']['formats']
        
        created_files = []
        
        print(f"\n💾 Embedding Export startet...")
        
        for fmt in formats:
            try:
                if self.config['export']['timestamp_in_name']:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    output_name = f"{output_base_name}_{timestamp}_{fmt}"
                else:
                    output_name = f"{output_base_name}_{fmt}"
                
                self.word_embedder.exportiere_embeddings(
                    datei_pfad=output_name,
                    format=fmt
                )
                
                file_path = f"{output_name}.{fmt}"
                created_files.append(file_path)
                self.stats['export_files'].append(file_path)
                
                print(f"✅ {fmt.upper()}: {file_path}")
                
            except Exception as e:
                print(f"❌ Fehler beim Export als {fmt}: {str(e)[:100]}")
        
        return created_files
    
    def initialize_transformer(self) -> ContextualTransformer:
        """
        Initialisiert einen ContextualTransformer basierend auf den trainierten Embeddings.
        
        Returns:
            ContextualTransformer Objekt
        
        UML-Methode: Public API Methode (optional)
        """
        if self.word_embedder is None:
            raise ValueError("Keine trainierten Embeddings vorhanden.")
        
        print("\n🤖 Initialisiere Contextual Transformer...")
        
        self.context_transformer = ContextualTransformer(
            wort_embedder=self.word_embedder,
            embedding_dimension=self.config['training']['embedding_dim'],
            anzahl_koepfe=4,
            anzahl_schichten=2,
            dropout_rate=0.1
        )
        
        print(f"✅ Transformer mit {sum(p.numel() for p in self.context_transformer.parameters()):,} Parametern")
        
        return self.context_transformer
    
    def initialize_semantic_analyzer(self) -> SemanticAnalyzer:
        """
        Initialisiert einen SemanticAnalyzer für erweiterte NLP-Analysen.
        
        Returns:
            SemanticAnalyzer Objekt
        
        UML-Methode: Public API Methode (optional)
        """
        if self.word_embedder is None:
            raise ValueError("Keine trainierten Embeddings vorhanden.")
        
        if self.context_transformer is None:
            self.initialize_transformer()
        
        print("\n🔍 Initialisiere Semantic Analyzer...")
        
        self.semantic_analyzer = SemanticAnalyzer(
            wort_embedder=self.word_embedder,
            kontext_transformer=self.context_transformer,
            embedding_dimension=self.config['training']['embedding_dim']
        )
        
        print("✅ Semantic Analyzer bereit für Analysen")
        
        return self.semantic_analyzer
    
    def run_semantic_analysis(self, query_text: str, sentences: List[str] = None) -> Dict:
        """
        Führt semantische Analysen durch.
        
        Args:
            query_text: Suchtext für Ähnlichkeitsanalyse
            sentences: Optionale Liste von Sätzen zum Vergleichen
        
        Returns:
            Dictionary mit Analyse-Ergebnissen
        
        UML-Methode: Public API Methode (optional)
        """
        if self.semantic_analyzer is None:
            self.initialize_semantic_analyzer()
        
        results = {
            'query': query_text,
            'similar_sentences': [],
            'word_analyses': []
        }
        
        # Satzähnlichkeiten berechnen
        if sentences:
            print(f"\n🔍 Finde ähnliche Sätze zu: '{query_text}'")
            similar_sentences = self.semantic_analyzer.finde_aehnlichste_saetze(
                such_text=query_text,
                saetze=sentences,
                anzahl=5
            )
            
            for i, (sentence, similarity) in enumerate(similar_sentences, 1):
                results['similar_sentences'].append({
                    'rank': i,
                    'sentence': sentence,
                    'similarity': float(similarity)
                })
                print(f"   {i}. [{similarity:.3f}] {sentence[:80]}...")
        
        return results
    
    def export_from_file(self, 
                        input_path: str,
                        output_name: str = "embeddings",
                        is_directory: bool = False) -> Dict:
        """
        Komplette Pipeline: Datei lesen → Training → Export.
        
        Args:
            input_path: Pfad zur Eingabedatei oder Verzeichnis
            output_name: Basisname für Ausgabedateien
            is_directory: True wenn input_path ein Verzeichnis ist
        
        Returns:
            Dictionary mit Statistiken und Dateipfaden
        
        UML-Methode: Haupt-Public API Methode
        """
        print("=" * 70)
        print("🚀 FILE-BASED EMBEDDING EXPORT PIPELINE")
        print("=" * 70)
        
        try:
            # 1. Text laden
            print(f"\n1. 📖 DATEN LADEN")
            print("-" * 40)
            
            if is_directory:
                self.corpus_text = self.read_from_directory(input_path)
            else:
                self.corpus_text = self.read_text_from_file(input_path)
            
            # 2. Text vorverarbeiten
            print(f"\n2. 🧹 TEXT VORVERARBEITEN")
            print("-" * 40)
            
            self.corpus_text = self.preprocess_text(self.corpus_text)
            
            # 3. Embeddings trainieren
            print(f"\n3. 🏋️  EMBEDDINGS TRAINIEREN")
            print("-" * 40)
            
            self.train_embeddings(self.corpus_text)
            
            # 4. Embeddings exportieren
            print(f"\n4. 💾 EMBEDDINGS EXPORTIEREN")
            print("-" * 40)
            
            export_files = self.export_embeddings(output_name)
            
            # 5. Zusammenfassung
            print(f"\n5. 📊 ZUSAMMENFASSUNG")
            print("-" * 40)
            
            summary = {
                'input_path': input_path,
                'input_type': 'directory' if is_directory else 'file',
                'total_words': self.stats['total_words'],
                'vocabulary_size': self.stats['vocabulary_size'],
                'files_processed': self.stats['files_processed'],
                'training_time_seconds': self.stats['training_time'],
                'exported_files': export_files,
                'embedding_dimension': self.config['training']['embedding_dim'],
                'training_epochs': self.config['training']['epochs']
            }
            
            for key, value in summary.items():
                if key != 'exported_files':
                    print(f"   {key.replace('_', ' ').title()}: {value}")
            
            print(f"\n📁 Exportierte Dateien:")
            for file_path in export_files:
                if os.path.exists(file_path):
                    size_kb = os.path.getsize(file_path) / 1024
                    print(f"   • {os.path.basename(file_path)} ({size_kb:.1f} KB)")
            
            print(f"\n🎉 PIPELINE ERFOLGREICH ABGESCHLOSSEN!")
            
            return summary
            
        except Exception as e:
            print(f"\n❌ FEHLER: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def get_config_summary(self) -> str:
        """
        Gibt eine Zusammenfassung der aktuellen Konfiguration zurück.
        
        Returns:
            Formatierte Konfigurationsübersicht
        
        UML-Methode: Public API Methode
        """
        summary = "📋 KONFIGURATIONSÜBERSICHT\n"
        summary += "=" * 50 + "\n"
        
        for section, settings in self.config.items():
            summary += f"\n{section.upper()}:\n"
            for key, value in settings.items():
                summary += f"  {key}: {value}\n"
        
        return summary
    
    def save_config(self, file_path: str) -> None:
        """
        Speichert die aktuelle Konfiguration in einer JSON Datei.
        
        Args:
            file_path: Pfad zur Konfigurationsdatei
        
        UML-Methode: Public API Methode
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Konfiguration gespeichert in {file_path}")
    
    def load_config(self, file_path: str) -> None:
        """
        Lädt Konfiguration aus einer JSON Datei.
        
        Args:
            file_path: Pfad zur Konfigurationsdatei
        
        UML-Methode: Public API Methode
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            loaded_config = json.load(f)
        
        self._merge_config(loaded_config)
        print(f"✅ Konfiguration geladen aus {file_path}")


def main():
    """
    Hauptfunktion für Kommandozeilen-Nutzung.
    """
    parser = argparse.ArgumentParser(
        description='Universal Embedding Exporter - Trainiert und exportiert Word Embeddings aus Dateien',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # Einzelne Datei verarbeiten
  %(prog)s mein_korpus.txt --output meine_embeddings
  
  # Verzeichnis verarbeiten
  %(prog)s texte/ --directory --output verzeichnis_embeddings
  
  # Mit angepasster Konfiguration
  %(prog)s korpus.txt --epochs 20 --dimension 128 --formats txt,json,csv
  
  # Konfiguration speichern/laden
  %(prog)s --save-config config.json
  %(prog)s --load-config config.json --input korpus.txt

  # Semantische Analyse durchführen
  %(prog)s korpus.txt --analyze "Hund Katze" --sentences saetze.txt
        """
    )
    
    # Hauptargumente
    parser.add_argument(
        'input',
        nargs='?',
        help='Eingabedatei oder Verzeichnis (optional bei --load-config)'
    )
    
    parser.add_argument(
        '--directory', '-d',
        action='store_true',
        help='Behandelt Eingabe als Verzeichnis'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='embeddings',
        help='Basisname für Ausgabedateien (Standard: embeddings)'
    )
    
    # Training-Parameter
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=15,
        help='Anzahl Trainingsepochen (Standard: 15)'
    )
    
    parser.add_argument(
        '--dimension', '-dim',
        type=int,
        default=64,
        help='Embedding-Dimension (Standard: 64)'
    )
    
    parser.add_argument(
        '--window', '-w',
        type=int,
        default=5,
        help='Kontextfenster-Größe (Standard: 5)'
    )
    
    # Export-Parameter
    parser.add_argument(
        '--formats', '-f',
        default='txt,json',
        help='Export-Formate kommagetrennt (Standard: txt,json)'
    )
    
    # Konfigurations-Management
    parser.add_argument(
        '--save-config',
        metavar='FILE',
        help='Speichert aktuelle Konfiguration in JSON Datei'
    )
    
    parser.add_argument(
        '--load-config',
        metavar='FILE',
        help='Lädt Konfiguration aus JSON Datei'
    )
    
    # Semantische Analyse (optional)
    parser.add_argument(
        '--analyze',
        metavar='QUERY',
        help='Führt semantische Analyse mit Query-Text durch'
    )
    
    parser.add_argument(
        '--sentences',
        metavar='FILE',
        help='Datei mit Sätzen für semantische Analyse'
    )
    
    # Vorverarbeitungs-Parameter
    parser.add_argument(
        '--no-clean',
        action='store_true',
        help='Deaktiviert Textbereinigung'
    )
    
    parser.add_argument(
        '--min-length',
        type=int,
        default=3,
        help='Minimale Satzlänge in Wörtern (Standard: 3)'
    )
    
    parser.add_argument(
        '--max-length',
        type=int,
        default=100,
        help='Maximale Satzlänge in Wörtern (Standard: 100)'
    )
    
    # Zusätzliche Optionen
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Ausführliche Ausgabe'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='FileBasedEmbeddingExporter v1.0.0'
    )
    
    args = parser.parse_args()
    
    # Exporter erstellen
    try:
        exporter = FileBasedEmbeddingExporter()
        
        # Konfiguration anpassen basierend auf Kommandozeilen-Argumenten
        user_config = {
            'training': {
                'epochs': args.epochs,
                'embedding_dim': args.dimension,
                'window_size': args.window
            },
            'preprocessing': {
                'clean_text': not args.no_clean,
                'min_sentence_length': args.min_length,
                'max_sentence_length': args.max_length
            },
            'export': {
                'formats': [fmt.strip() for fmt in args.formats.split(',')]
            }
        }
        
        exporter._merge_config(user_config)
        
        # Konfiguration speichern/laden
        if args.save_config:
            exporter.save_config(args.save_config)
            print(f"✅ Konfiguration gespeichert in {args.save_config}")
            return
        
        if args.load_config:
            exporter.load_config(args.load_config)
            print("✅ Konfiguration geladen")
        
        # Haupt-Pipeline ausführen
        if args.input:
            print("🚀 Starte Embedding Pipeline...\n")
            
            summary = exporter.export_from_file(
                input_path=args.input,
                output_name=args.output,
                is_directory=args.directory
            )
            
            # Semantische Analyse (optional)
            if args.analyze:
                print(f"\n🔍 SEMANTISCHE ANALYSE")
                print("-" * 40)
                
                sentences = []
                if args.sentences and os.path.exists(args.sentences):
                    with open(args.sentences, 'r', encoding='utf-8') as f:
                        sentences = [line.strip() for line in f if line.strip()]
                
                results = exporter.run_semantic_analysis(
                    query_text=args.analyze,
                    sentences=sentences
                )
                
                # Ergebnisse speichern
                results_file = f"{args.output}_analysis.json"
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                
                print(f"✅ Analyse-Ergebnisse gespeichert in {results_file}")
            
        elif not args.save_config and not args.load_config:
            print("ℹ️  Keine Eingabedatei angegeben.")
            print("   Verwende --help für Hilfe.")
            
    except Exception as e:
        print(f"\n❌ PROGRAMM ABGEBROCHEN: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()