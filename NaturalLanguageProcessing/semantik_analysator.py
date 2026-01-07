#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SemanticAnalyzer - High-Level NLP Wrapper mit erweiterten Funktionen

Diese Klasse kombiniert Word2Vec Embeddings mit einem kontextuellen Transformer,
um semantische Repräsentationen von Wörtern und Sätzen zu berechnen.

Merkmale:
- Unterstützung für variable Satzlängen
- Batch-Verarbeitung mit Padding
- Erweiterte Ähnlichkeitsberechnungen
- Top-N Suche
- Dokument-Embeddings

Autor: Ufuk Baysal
Datum: 2025
"""

import sys
import os
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import torch
from torch import Tensor
from sklearn.metrics.pairwise import cosine_similarity


class SemanticAnalyzer:
    """
    SemanticAnalyzer - High-Level NLP Wrapper
    
    Kombiniert Word2Vec Embeddings mit kontextueller Aufmerksamkeit
    eines Transformers, um semantische Repräsentationen zu berechnen.
    
    Args:
        wort_embedder (Word2VecEmbedding): Trainiertes Word2Vec Modell
        kontext_transformer (ContextualTransformer): Transformer für kontextuelle Repräsentationen
        embedding_dimension (int): Dimension des Embedding-Raums (Standard: 64)
        use_weighted_mean (bool): Verwendung von TF-IDF ähnlichen Gewichten für Satzvektoren
    """
    
    def __init__(self, 
                 wort_embedder: Any,  # Word2VecEmbedding Instanz
                 kontext_transformer: Any,  # ContextualTransformer Instanz
                 embedding_dimension: int = 64,
                 use_weighted_mean: bool = True):
        """
        Initialisiert den Semantic Analyzer.
        
        Args:
            wort_embedder: Trainiertes Word2Vec Embedding Modell
            kontext_transformer: Transformer für kontextuelle Repräsentationen
            embedding_dimension: Dimension der Embeddings
            use_weighted_mean: Wenn True, werden Wortvektoren gewichtet gemittelt
        """
        self._wort_embedder = wort_embedder
        self._kontext_transformer = kontext_transformer
        self._embedding_dimension = embedding_dimension
        self._use_weighted_mean = use_weighted_mean
        
        # Wortfrequenzen für gewichtete Mittelung berechnen
        self._wort_haeufigkeiten = self._berechne_wort_haeufigkeiten()
        
        print(f"✅ SemanticAnalyzer initialisiert:")
        print(f"   - Embedding Dimension: {embedding_dimension}")
        print(f"   - Wörterbuch Größe: {wort_embedder.woerterbuch_groesse}")
        print(f"   - Gewichtete Mittelung: {use_weighted_mean}")
    
    def _berechne_wort_haeufigkeiten(self) -> Dict[str, float]:
        """
        Berechnet die Häufigkeiten aller Wörter im Korpus.
        
        Returns:
            Dictionary mit Wort->Häufigkeit Mappings
        """
        haeufigkeiten = {}
        tokens = self._wort_embedder.woerter
        
        for token in tokens:
            if token in haeufigkeiten:
                haeufigkeiten[token] += 1
            else:
                haeufigkeiten[token] = 1
        
        # Normalisieren
        total = sum(haeufigkeiten.values())
        for wort in haeufigkeiten:
            haeufigkeiten[wort] /= total
        
        return haeufigkeiten
    
    def _berechne_wort_gewicht(self, wort: str) -> float:
        """
        Berechnet das Gewicht eines Wortes für die gewichtete Mittelung.
        Seltener Wörter erhalten höhere Gewichte (TF-IDF ähnlich).
        
        Args:
            wort: Das zu bewertende Wort
            
        Returns:
            Gewicht zwischen 0 und 1
        """
        if not self._use_weighted_mean:
            return 1.0
        
        wort_lower = wort.lower()
        if wort_lower in self._wort_haeufigkeiten:
            # Inverse Häufigkeit (seltene Wörter bekommen höheres Gewicht)
            haeufigkeit = self._wort_haeufigkeiten[wort_lower]
            return 1.0 / (1.0 + haeufigkeit * 10)  # Glättung
        else:
            return 0.5  # Standardgewicht für unbekannte Wörter
    
    def _erstelle_satz_vektor(self, satz: str) -> Tensor:
        """
        Konvertiert einen Satz in einen einzigen Vektor.
        
        Verwendet SkipGram Embeddings + Transformer Self-Attention.
        Die Repräsentation wird als gewichteter Durchschnitt der
        aufmerksamkeits-kodierten Token-Embeddings berechnet.
        
        Args:
            satz: Eingabesatz
            
        Returns:
            Tensor: Satzvektor der Form (embedding_dimension,)
            
        Raises:
            ValueError: Wenn der Satz leer ist oder keine gültigen Wörter enthält
        """
        # Input Validation
        if not satz or not satz.strip():
            # Rückgabe eines Nullvektors für leere Sätze
            return torch.zeros(self._embedding_dimension)
        
        try:
            # Wortindizes erhalten
            indices: Tensor = self._wort_embedder.erhalte_indizes_von_satz(satz=satz)
            
            # Prüfen ob gültige Indizes vorhanden sind
            if indices.numel() == 0 or indices.sum() == 0:
                return torch.zeros(self._embedding_dimension)
            
            # Kontextuelle Kodierung mit Transformer
            with torch.no_grad():
                # Padding-Maskierung für variable Längen
                maskierung = (indices != 0).float()  # Annahme: 0 ist Padding
                encoded = self._kontext_transformer(indices.unsqueeze(0), maskierung)
                encoded = encoded.squeeze(0)
            
            # Gewichteten Durchschnitt berechnen
            satz_woerter = self._wort_embedder._tokenisiere_text(satz)
            
            if len(satz_woerter) != encoded.shape[0]:
                # Falls Längen nicht übereinstimmen, einfachen Durchschnitt nehmen
                return torch.mean(encoded, dim=0)
            
            # Gewichte berechnen
            gewichte = []
            for i, wort in enumerate(satz_woerter):
                if i < encoded.shape[0]:  # Sicherstellen, dass Index gültig ist
                    gewicht = self._berechne_wort_gewicht(wort)
                    gewichte.append(gewicht)
            
            if not gewichte:
                return torch.mean(encoded, dim=0)
            
            # Normalisierte Gewichte
            gewichte_tensor = torch.tensor(gewichte, dtype=torch.float32)
            if len(gewichte_tensor) != encoded.shape[0]:
                gewichte_tensor = torch.ones(encoded.shape[0])
            
            gewichte_tensor = gewichte_tensor / gewichte_tensor.sum()
            
            # Gewichteter Durchschnitt
            if len(gewichte_tensor.shape) == 1:
                gewichte_tensor = gewichte_tensor.unsqueeze(1)
            
            satz_vektor = torch.sum(encoded * gewichte_tensor, dim=0)
            
            # Normalisieren
            norm = torch.norm(satz_vektor)
            if norm > 0:
                satz_vektor = satz_vektor / norm
            
            return satz_vektor
            
        except Exception as e:
            print(f"⚠️  Fehler bei Satzvektor-Erstellung für '{satz[:50]}...': {str(e)}")
            # Fallback: Einfacher Nullvektor
            return torch.zeros(self._embedding_dimension)
    
    def berechne_satz_aehnlichkeiten(self, 
                                   saetze: List[str], 
                                   such_text: str,
                                   zeige_top_n: Optional[int] = None,
                                   min_aehnlichkeit: float = 0.0) -> None:
        """
        Vergleicht einen Query-Satz mit einer Liste von Kandidatensätzen
        und gibt ihre Kosinus-Ähnlichkeiten aus.
        
        Findet den ähnlichsten Satz gemäß dem gelernten
        Embeddings + Attention Modell.
        
        Args:
            saetze: Liste von Kandidatensätzen
            such_text: Query-Satz zum Vergleichen
            zeige_top_n: Nur die Top-N ähnlichsten Sätze anzeigen (None für alle)
            min_aehnlichkeit: Minimale Ähnlichkeit zum Anzeigen
            
        Returns:
            None, gibt Ergebnisse direkt aus
        """
        # Input Validation
        if not saetze:
            print("⚠️  Keine Sätze zum Vergleichen vorhanden.")
            return
        
        if not such_text or not such_text.strip():
            print("⚠️  Suchtext ist leer.")
            return
        
        try:
            print(f"\n🔍 Query: '{such_text}'")
            print("=" * 80)
            
            # Query-Vektor erstellen
            query_vektor = self._erstelle_satz_vektor(such_text)
            
            if torch.norm(query_vektor) == 0:
                print("⚠️  Query-Vektor ist leer (keine gültigen Wörter gefunden)")
                return
            
            query_numpy = query_vektor.detach().numpy().reshape(1, -1)
            
            # Alle Satzvektoren und Ähnlichkeiten berechnen
            ergebnisse = []
            for i, satz in enumerate(saetze):
                try:
                    if not satz or not satz.strip():
                        continue
                    
                    satz_vektor = self._erstelle_satz_vektor(satz)
                    
                    if torch.norm(satz_vektor) == 0:
                        continue
                    
                    satz_numpy = satz_vektor.detach().numpy().reshape(1, -1)
                    aehnlichkeit = cosine_similarity(query_numpy, satz_numpy)[0][0]
                    
                    # Nur Sätze über Minimum-Ähnlichkeit behalten
                    if aehnlichkeit >= min_aehnlichkeit:
                        ergebnisse.append((satz, aehnlichkeit, i))
                        
                except Exception as e:
                    print(f"⚠️  Fehler bei Satz {i} ('{satz[:30]}...'): {str(e)[:50]}")
                    continue
            
            # Sortieren nach Ähnlichkeit (absteigend)
            ergebnisse.sort(key=lambda x: x[1], reverse=True)
            
            # Top-N limitieren wenn gewünscht
            if zeige_top_n is not None and zeige_top_n > 0:
                ergebnisse = ergebnisse[:zeige_top_n]
            
            # Ergebnisse ausgeben
            if not ergebnisse:
                print("Keine ähnlichen Sätze gefunden.")
                return
            
            for satz, aehnlichkeit, original_idx in ergebnisse:
                # Visualisierung der Ähnlichkeit mit Sternen
                sterne = "★" * min(5, int(aehnlichkeit * 5) + 1)
                
                print(f"Satz #{original_idx + 1}: {satz}")
                print(f"   Ähnlichkeit: {aehnlichkeit:.4f} {sterne}")
                print(f"   Länge: {len(satz.split())} Wörter")
                print("-" * 80)
            
            # Bestes Ergebnis extrahieren
            bester_satz, beste_aehnlichkeit, _ = ergebnisse[0]
            
            print(f"\n📊 Zusammenfassung:")
            print(f"   Gegebener Satz: '{such_text}'")
            print(f"   Gefundener Satz: '{bester_satz[:100]}...'")
            print(f"   Höchste Ähnlichkeit: {beste_aehnlichkeit:.4f}")
            print(f"   Gefundene ähnliche Sätze: {len(ergebnisse)}")
            
        except Exception as e:
            print(f"❌ Fehler bei Satzähnlichkeitsberechnung: {str(e)}")
    
    def _erstelle_wort_vektor(self, wort: str) -> Tensor:
        """
        Konvertiert ein Wort in seine Vektor-Repräsentation.
        
        Verwendet SkipGram Embeddings + Transformer Self-Attention.
        
        Args:
            wort: Eingabewort
            
        Returns:
            Tensor: Wortvektor der Form (embedding_dimension,)
        """
        try:
            # Input Validation
            if not wort or not wort.strip():
                return torch.zeros(self._embedding_dimension)
            
            wort_lower = wort.lower().strip()
            
            # Wortindex erhalten
            idx: Tensor = self._wort_embedder.erhalte_index_von_wort(wort=wort_lower)
            
            # Prüfen ob Wort im Vokabular ist
            if idx.numel() == 0 or idx.sum() == 0:
                # Fallback: Versuche, das Wort zu tokenisieren und Embeddings zu mitteln
                tokenisiert = self._wort_embedder._tokenisiere_text(wort_lower)
                if tokenisiert:
                    vektoren = []
                    for token in tokenisiert:
                        token_idx = self._wort_embedder.erhalte_index_von_wort(token)
                        if token_idx.sum() > 0:
                            with torch.no_grad():
                                vektor = self._kontext_transformer(
                                    token_idx.unsqueeze(0), 
                                    None
                                ).squeeze(0).squeeze(0)
                            vektoren.append(vektor)
                    
                    if vektoren:
                        return torch.mean(torch.stack(vektoren), dim=0)
                
                # Wenn nichts gefunden, Nullvektor zurückgeben
                return torch.zeros(self._embedding_dimension)
            
            # Kontextuelle Kodierung
            with torch.no_grad():
                # Annahme: einzelnes Wort
                encoded = self._kontext_transformer(idx.unsqueeze(0), None)
            
            vektor = encoded.squeeze(0).squeeze(0)
            
            # Normalisieren
            norm = torch.norm(vektor)
            if norm > 0:
                vektor = vektor / norm
            
            return vektor
            
        except Exception as e:
            print(f"⚠️  Fehler bei Wortvektor-Erstellung für '{wort}': {str(e)}")
            return torch.zeros(self._embedding_dimension)
    
    def berechne_wort_aehnlichkeiten(self, 
                                   wort_paare: List[Tuple[str, str]],
                                   zeige_embedding_stats: bool = False) -> None:
        """
        Berechnet und gibt Kosinus-Ähnlichkeitswerte für Wortpaare aus.
        
        Args:
            wort_paare: Liste von Wortpaaren (wort1, wort2)
            zeige_embedding_stats: Wenn True, werden Embedding-Statistiken angezeigt
            
        Returns:
            None, gibt Ergebnisse direkt aus
        """
        if not wort_paare:
            print("⚠️  Keine Wortpaare zum Vergleichen.")
            return
        
        print(f"\n📊 Wort-Ähnlichkeiten (Word2Vec + Transformer)")
        print("=" * 80)
        
        gesamt_aehnlichkeit = 0
        erfolgreiche_paare = 0
        
        for w1, w2 in wort_paare:
            try:
                if not w1 or not w2:
                    print(f"⚠️  Leeres Wort in Paar übersprungen")
                    continue
                
                v1 = self._erstelle_wort_vektor(w1)
                v2 = self._erstelle_wort_vektor(w2)
                
                # Prüfen ob Vektoren gültig sind
                if torch.norm(v1) == 0 or torch.norm(v2) == 0:
                    print(f"❌ {w1} ↔ {w2}: Embedding nicht gefunden")
                    continue
                
                # In numpy konvertieren für Ähnlichkeitsberechnung
                v1_np = v1.detach().numpy().reshape(1, -1)
                v2_np = v2.detach().numpy().reshape(1, -1)
                
                # Kosinus-Ähnlichkeit berechnen
                aehnlichkeit = cosine_similarity(v1_np, v2_np)[0][0]
                
                # Semantische Beziehung klassifizieren
                if aehnlichkeit > 0.7:
                    beziehung = "Sehr ähnlich"
                elif aehnlichkeit > 0.4:
                    beziehung = "Ähnlich"
                elif aehnlichkeit > 0.1:
                    beziehung = "Leicht ähnlich"
                elif aehnlichkeit > -0.1:
                    beziehung = "Neutral"
                elif aehnlichkeit > -0.4:
                    beziehung = "Gegensätzlich"
                else:
                    beziehung = "Sehr gegensätzlich"
                
                # Ausgabe
                print(f"{w1:15} ↔ {w2:15}: {aehnlichkeit:7.4f}  [{beziehung}]")
                
                if zeige_embedding_stats:
                    print(f"       Normen: ||{w1}||={torch.norm(v1):.3f}, ||{w2}||={torch.norm(v2):.3f}")
                
                gesamt_aehnlichkeit += aehnlichkeit
                erfolgreiche_paare += 1
                
            except Exception as e:
                print(f"⚠️  Fehler bei Paar ({w1}, {w2}): {str(e)[:50]}")
                continue
        
        if erfolgreiche_paare > 0:
            durchschnitt = gesamt_aehnlichkeit / erfolgreiche_paare
            print(f"\n📈 Durchschnittliche Ähnlichkeit: {durchschnitt:.4f}")
            print(f"   Erfolgreich verarbeitete Paare: {erfolgreiche_paare}/{len(wort_paare)}")
        else:
            print("\n❌ Keine Wortpaare konnten verarbeitet werden.")
    
    def finde_aehnlichste_saetze(self, 
                                such_text: str, 
                                saetze: List[str],
                                anzahl: int = 3) -> List[Tuple[str, float]]:
        """
        Findet die N ähnlichsten Sätze zu einem gegebenen Suchtext.
        
        Args:
            such_text: Der Suchtext/Query
            saetze: Liste von Kandidatensätzen
            anzahl: Anzahl der zurückzugebenden ähnlichsten Sätze
            
        Returns:
            Liste von Tupeln (Satz, Ähnlichkeitswert), sortiert absteigend
        """
        if not such_text or not saetze:
            return []
        
        try:
            query_vektor = self._erstelle_satz_vektor(such_text)
            
            if torch.norm(query_vektor) == 0:
                return []
            
            query_numpy = query_vektor.detach().numpy().reshape(1, -1)
            
            ergebnisse = []
            for satz in saetze:
                try:
                    satz_vektor = self._erstelle_satz_vektor(satz)
                    
                    if torch.norm(satz_vektor) == 0:
                        continue
                    
                    satz_numpy = satz_vektor.detach().numpy().reshape(1, -1)
                    aehnlichkeit = cosine_similarity(query_numpy, satz_numpy)[0][0]
                    
                    ergebnisse.append((satz, aehnlichkeit))
                    
                except Exception:
                    continue
            
            # Sortieren und begrenzen
            ergebnisse.sort(key=lambda x: x[1], reverse=True)
            return ergebnisse[:anzahl]
            
        except Exception as e:
            print(f"⚠️  Fehler bei finde_aehnlichste_saetze: {str(e)}")
            return []
    
    def berechne_dokument_embedding(self, dokument: str) -> Tensor:
        """
        Berechnet ein Dokument-Embedding als gewichteten Durchschnitt aller Satz-Embeddings.
        
        Args:
            dokument: Textdokument (kann mehrere Sätze enthalten)
            
        Returns:
            Tensor: Dokument-Embedding Vektor
        """
        if not dokument:
            return torch.zeros(self._embedding_dimension)
        
        try:
            # Sätze extrahieren (einfache Segmentierung)
            saetze = [s.strip() for s in dokument.split('.') if s.strip()]
            
            if not saetze:
                return torch.zeros(self._embedding_dimension)
            
            # Embeddings für alle Sätze berechnen
            satz_embeddings = []
            satz_gewichte = []
            
            for satz in saetze:
                satz_vektor = self._erstelle_satz_vektor(satz)
                
                if torch.norm(satz_vektor) > 0:
                    satz_embeddings.append(satz_vektor)
                    # Gewicht basierend auf Satzlänge (längere Sätze bekommen mehr Gewicht)
                    satz_gewichte.append(len(satz.split()))
            
            if not satz_embeddings:
                return torch.zeros(self._embedding_dimension)
            
            # Gewichteten Durchschnitt berechnen
            satz_embeddings_tensor = torch.stack(satz_embeddings)
            gewichte_tensor = torch.tensor(satz_gewichte, dtype=torch.float32)
            gewichte_tensor = gewichte_tensor / gewichte_tensor.sum()
            
            # Dimensionen anpassen für gewichtete Summe
            if len(gewichte_tensor.shape) == 1:
                gewichte_tensor = gewichte_tensor.unsqueeze(1)
            
            dokument_embedding = torch.sum(satz_embeddings_tensor * gewichte_tensor, dim=0)
            
            # Normalisieren
            norm = torch.norm(dokument_embedding)
            if norm > 0:
                dokument_embedding = dokument_embedding / norm
            
            return dokument_embedding
            
        except Exception as e:
            print(f"⚠️  Fehler bei Dokument-Embedding Berechnung: {str(e)}")
            return torch.zeros(self._embedding_dimension)
    
    def visualisiere_aehnlichkeits_matrix(self, 
                                        woerter: List[str],
                                        titel: str = "Wort-Ähnlichkeitsmatrix") -> None:
        """
        Visualisiert eine Ähnlichkeitsmatrix für eine Liste von Wörtern.
        
        Args:
            woerter: Liste von Wörtern zur Visualisierung
            titel: Titel der Visualisierung
            
        Note:
            Erfordert matplotlib und seaborn
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if len(woerter) < 2:
                print("⚠️  Mindestens 2 Wörter benötigt für Visualisierung")
                return
            
            # Embeddings für alle Wörter berechnen
            embeddings = []
            gueltige_woerter = []
            
            for wort in woerter:
                vektor = self._erstelle_wort_vektor(wort)
                if torch.norm(vektor) > 0:
                    embeddings.append(vektor.detach().numpy())
                    gueltige_woerter.append(wort)
            
            if len(gueltige_woerter) < 2:
                print("⚠️  Nicht genug gültige Wort-Embeddings gefunden")
                return
            
            # Ähnlichkeitsmatrix berechnen
            embeddings_array = np.array(embeddings)
            aehnlichkeits_matrix = cosine_similarity(embeddings_array)
            
            # Plot erstellen
            plt.figure(figsize=(10, 8))
            sns.heatmap(aehnlichkeits_matrix,
                       annot=True,
                       fmt='.2f',
                       cmap='RdYlBu',
                       center=0,
                       xticklabels=gueltige_woerter,
                       yticklabels=gueltige_woerter)
            
            plt.title(titel)
            plt.tight_layout()
            plt.show()
            
            print(f"✅ Ähnlichkeitsmatrix für {len(gueltige_woerter)} Wörter erstellt")
            
        except ImportError:
            print("❌ Visualisierung deaktiviert: matplotlib oder seaborn nicht installiert")
            print("   Installieren mit: pip install matplotlib seaborn")
        except Exception as e:
            print(f"⚠️  Fehler bei Visualisierung: {str(e)}")