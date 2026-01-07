#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Word2VecEmbedding - Verbesserte Skip-Gram Implementierung

Diese Klasse implementiert ein verbessertes Skip-Gram Modell mit:
- Unterstützung für variable Satzlängen
- Padding für Batch-Verarbeitung
- Erweiterte Tokenisierung mit Unicode-Unterstützung
- Negative Sampling Option
- Embedding Normalisierung

Autor: Ufuk Baysal
Datum: 2025
"""

import re
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from collections import Counter
from torch import Tensor

# Relative Imports
from .skipgram_modell import SkipGramModell
from .skipgram_datensatz import SkipGramDatenSatz


class Word2VecEmbedding:
    """
    Verbesserte Wrapper-Klasse für Skip-Gram Training mit erweiterten Funktionen.
    
    Args:
        text_korpus (str): Eingabetext
        kontext_fenster (int, optional): Größe des Kontextfensters. Standard ist 5.
        embedding_dimension (int, optional): Dimension der Embeddings. Standard ist 64.
        max_satzelemente (int, optional): Maximale Satzlänge für Padding. Standard ist 50.
        use_stopwords (bool, optional): Stopwort-Filterung aktivieren. Standard ist False.
        min_wort_haeufigkeit (int, optional): Minimale Wortfrequenz für Vokabular. Standard ist 1.
        negative_samples (int, optional): Anzahl negativer Samples. Standard ist 5.
    
    Beispiel:
        >>> korpus = "Der schnelle Fuchs springt über den Hund"
        >>> embedder = Word2VecEmbedding(korpus, kontext_fenster=3, embedding_dimension=32)
        >>> embedder.trainiere(epochen=10)
        >>> vektor = embedder.erhalte_wort_vektor("Fuchs")
    """
    
    def __init__(self, 
                 text_korpus: str, 
                 kontext_fenster: int = 5,
                 embedding_dimension: int = 64,
                 max_satzelemente: int = 50,
                 use_stopwords: bool = False,
                 min_wort_haeufigkeit: int = 1,
                 negative_samples: int = 5):
        """
        Initialisiert den Word2Vec Embedding Trainer.
        """
        # Parameter speichern
        self._text_korpus: str = text_korpus
        self._kontext_fenster: int = kontext_fenster
        self._embedding_dimension: int = embedding_dimension
        self._max_satzelemente: int = max_satzelemente
        self._use_stopwords: bool = use_stopwords
        self._min_wort_haeufigkeit: int = min_wort_haeufigkeit
        self._negative_samples: int = negative_samples
        
        # Datenstrukturen
        self._trainings_paare: List[Tuple[int, int]] = []
        self._woerter: List[str] = []
        self._wort_zu_index: Dict[str, int] = {}
        self._index_zu_wort: Dict[int, str] = {}
        self._wort_haeufigkeiten: Dict[str, int] = {}
        
        # Deutsche Stopwörter
        self._deutsche_stopwoerter = set([
            'der', 'die', 'das', 'und', 'oder', 'aber', 'den', 'dem', 'des',
            'ein', 'eine', 'einer', 'einem', 'einen', 'eines', 'in', 'auf',
            'unter', 'über', 'vor', 'hinter', 'neben', 'zwischen', 'mit',
            'ohne', 'durch', 'für', 'gegen', 'um', 'bis', 'seit', 'während',
            'weil', 'obwohl', 'dass', 'wenn', 'als', 'wie', 'was', 'wer',
            'wo', 'wann', 'warum', 'wie', 'nicht', 'auch', 'noch', 'schon',
            'erst', 'nur', 'gerade', 'etwa', 'etwas', 'ganz', 'sehr', 'viel',
            'wenig', 'mehr', 'weniger', 'meist', 'oft', 'selten', 'immer',
            'nie', 'manchmal', 'zu', 'von', 'aus', 'bei', 'nach', 'zu',
            'aus', 'bei', 'mit', 'seit', 'von', 'zu', 'bis', 'durch', 'für',
            'gegen', 'ohne', 'um', 'an', 'auf', 'hinter', 'in', 'neben',
            'über', 'unter', 'vor', 'zwischen'
        ])
        
        # Initialisierung durchführen
        self._initialisiere_daten()
        
        # Modelle und Optimierer
        self._skipgram_modell = SkipGramModell(
            self._woerterbuch_groesse, 
            embedding_dimension,
            negative_samples=negative_samples
        )
        
        self._optimierer = optim.Adam(self._skipgram_modell.parameters(), lr=0.01)
        self._verlust_funktion = nn.CrossEntropyLoss()
        
        print(f"✅ Word2VecEmbedding initialisiert:")
        print(f"   - Vokabulargröße: {self._woerterbuch_groesse}")
        print(f"   - Embedding Dimension: {embedding_dimension}")
        print(f"   - Kontextfenster: {kontext_fenster}")
        print(f"   - Maximale Satzlänge: {max_satzelemente}")
        print(f"   - Stopwort-Filterung: {use_stopwords}")
    
    def _initialisiere_daten(self) -> None:
        """
        Initialisiert alle Datenstrukturen basierend auf dem Korpus.
        """
        # Tokenisierung
        self._woerter = self._tokenisiere_text(self._text_korpus)
        
        # Wortfrequenzen berechnen
        self._berechne_wort_haeufigkeiten()
        
        # Vokabular basierend auf Mindesthäufigkeit filtern
        self._filtere_vokabular()
        
        # Wort-zu-Index Mapping erstellen
        self._erstelle_wort_mappings()
        
        # Trainingspaare generieren
        self._generiere_trainings_paare()
    
    def _tokenisiere_text(self, text: str) -> List[str]:
        """
        Tokenisiert einen Text in Wörter mit erweiterter Unicode-Unterstützung.
        
        Args:
            text: Eingabetext
            
        Returns:
            Liste von Token/Wörtern
            
        Beispiel:
            >>> embedder._tokenisiere_text("Hallo, Welt!")
            ['hallo', 'welt']
        """
        if not text:
            return []
        
        try:
            # Zu Kleinbuchstaben konvertieren
            text = text.lower()
            
            # Erweiterte Unicode-Unterstützung für Deutsch
            # Behält deutsche Umlaute und ß bei, aber normalisiere sie
            text = re.sub(r'[^\wäöüß\s\-]', ' ', text)  # # Bindestriche behalten Entfernt Interpunktion
             # Spezielle Fälle behandeln
            text = text.replace('ß', 'ss')  # ß zu ss für bessere Matching
            text = re.sub(r'\s+', ' ', text)  # Normalisiert Leerzeichen
            text = text.strip()
            
            if not text:
                return []
            
            # Tokenisierung
            tokens = text.split()
            
            """             # Stopwort-Filterung
            if self._use_stopwords:
                tokens = [token for token in tokens 
                         if token not in self._deutsche_stopwoerter]
            
            return tokens 
            """
             # Einfache Lemmatisierung für deutsche Adjektivendungen
            bereinigte_tokens = []
            for token in tokens:
                # Entferne häufige Adjektivendungen
                if token.endswith('es'):
                    token = token[:-2]
                elif token.endswith('e'):
                    token = token[:-1]
                elif token.endswith('en'):
                    token = token[:-2]
                elif token.endswith('er'):
                    token = token[:-2]
                
                if token and len(token) > 1:  # Einzelbuchstaben ignorieren
                    bereinigte_tokens.append(token)
            
            return bereinigte_tokens
    
        except Exception as e:
            print(f"⚠️  Fehler bei Tokenisierung: {str(e)}")
            return []
    
    def _berechne_wort_haeufigkeiten(self) -> None:
        """
        Berechnet Häufigkeiten aller Wörter im Korpus.
        """
        if not self._woerter:
            return
        
        self._wort_haeufigkeiten = Counter(self._woerter)
    
    def _filtere_vokabular(self) -> None:
        """
        Filtert das Vokabular basierend auf Mindesthäufigkeit.
        """
        if not self._wort_haeufigkeiten:
            return
        
        # Wörter filtern, die die Mindesthäufigkeit erreichen
        gefilterte_woerter = []
        for wort, haeufigkeit in self._wort_haeufigkeiten.items():
            if haeufigkeit >= self._min_wort_haeufigkeit:
                gefilterte_woerter.append(wort)
        
        self._woerter = gefilterte_woerter
        
        # Häufigkeiten aktualisieren
        self._berechne_wort_haeufigkeiten()
    
    def _erstelle_wort_mappings(self) -> None:
        """
        Erstellt Wort-zu-Index und Index-zu-Wort Mappings.
        """
        if not self._woerter:
            return
        
        # Einzigartige Wörter
        einzigartige_woerter = sorted(set(self._woerter))
        
        # Mappings erstellen
        self._wort_zu_index = {wort: idx for idx, wort in enumerate(einzigartige_woerter)}
        self._index_zu_wort = {idx: wort for wort, idx in self._wort_zu_index.items()}
        
        self._woerterbuch_groesse = len(einzigartige_woerter)
    
    def _generiere_trainings_paare(self) -> None:
        """
        Generiert (Zentrum, Kontext) Paare für Skip-Gram Training.
        """
        if not self._woerter:
            return
        
        self._trainings_paare = []
        
        # Durch alle Positionen im Text iterieren
        for i, zentrum_wort in enumerate(self._woerter):
            # Nur wenn Wort im Vokabular ist
            if zentrum_wort not in self._wort_zu_index:
                continue
            
            zentrum_idx = self._wort_zu_index[zentrum_wort]
            
            # Kontextfenster definieren
            start_idx = max(0, i - self._kontext_fenster)
            end_idx = min(len(self._woerter), i + self._kontext_fenster + 1)
            
            # Kontextwörter sammeln
            for j in range(start_idx, end_idx):
                if i != j:  # Zentrumwort selbst ausschließen
                    kontext_wort = self._woerter[j]
                    
                    if kontext_wort in self._wort_zu_index:
                        kontext_idx = self._wort_zu_index[kontext_wort]
                        self._trainings_paare.append((zentrum_idx, kontext_idx))
        
        print(f"   - Generierte Trainingspaare: {len(self._trainings_paare)}")
    
    @property
    def woerter(self) -> List[str]:
        """
        Gibt die tokenisierten Wörter des Korpus zurück.
        
        Returns:
            Liste von Token/Wörtern
        """
        return self._woerter.copy()
    
    @property
    def trainings_paare(self) -> List[Tuple[int, int]]:
        """
        Gibt die generierten Trainingspaare zurück.
        
        Returns:
            Liste von (Zentrum_Index, Kontext_Index) Paaren
        """
        return self._trainings_paare.copy()
    
    @property
    def woerterbuch_groesse(self) -> int:
        """
        Gibt die Größe des Wörterbuchs zurück.
        
        Returns:
            Anzahl einzigartiger Wörter im Vokabular
        """
        return self._woerterbuch_groesse
    
    @property
    def embedding_dimension(self) -> int:
        """
        Gibt die Embedding-Dimension zurück.
        
        Returns:
            Dimension der Word Embeddings
        """
        return self._embedding_dimension
    
    @property
    def embedding_schicht(self) -> nn.Embedding:
        """
        Gibt die Embedding-Schicht des Skip-Gram Modells zurück.
        
        Returns:
            PyTorch Embedding Layer
        """
        return self._skipgram_modell.wort_embeddings
    
    def trainiere(self, 
                 epochen: int = 10, 
                 batch_groesse: int = 32,
                 lernrate: float = 0.01,
                 zeige_progress: bool = True) -> None:
        """
        Trainiert das Skip-Gram Modell.
        
        Args:
            epochen: Anzahl der Trainingsepochen
            batch_groesse: Größe der Trainingbatches
            lernrate: Lernrate für den Optimierer
            zeige_progress: Wenn True, wird Fortschritt angezeigt
        """
        if not self._trainings_paare:
            print("⚠️  Keine Trainingspaare verfügbar.")
            return
        
        # Lernrate setzen
        for param_group in self._optimierer.param_groups:
            param_group['lr'] = lernrate
        
        # Dataset und DataLoader erstellen
        dataset = SkipGramDatenSatz(self._trainings_paare)
        daten_lader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_groesse, 
            shuffle=True
        )
        
        if zeige_progress:
            print(f"\n🏋️  Training gestartet ({epochen} Epochen, Batch-Größe: {batch_groesse})")
            print("-" * 50)
        
        # Training loop
        for epoche in range(epochen):
            gesamt_verlust = 0.0
            anzahl_batches = 0
            
            for zentrum_indices, kontext_indices in daten_lader:
                # Gradienten zurücksetzen
                self._optimierer.zero_grad()
                
                # Forward Pass
                ausgabe = self._skipgram_modell(zentrum_indices)
                
                # Verlust berechnen
                verlust = self._verlust_funktion(ausgabe, kontext_indices)
                
                # Backward Pass
                verlust.backward()
                
                # Parameter aktualisieren
                self._optimierer.step()
                
                # Statistik aktualisieren
                gesamt_verlust += verlust.item()
                anzahl_batches += 1
            
            # Durchschnittlichen Verlust berechnen
            if anzahl_batches > 0:
                durchschnitt_verlust = gesamt_verlust / anzahl_batches
                
                if zeige_progress:
                    progress_bar = "█" * int((epoche + 1) / epochen * 20)
                    print(f"Epoche {epoche + 1:3d}/{epochen} | "
                          f"Verlust: {durchschnitt_verlust:7.4f} | "
                          f"[{progress_bar:20}]")
        
        if zeige_progress:
            print("-" * 50)
            print("✅ Training abgeschlossen")
    
    def erhalte_indizes_von_satz(self, satz: str) -> Tensor:
        """
        Konvertiert einen Satz in Wortindizes mit Padding für variable Längen.
        
        Args:
            satz: Eingabesatz
            
        Returns:
            Tensor: Tensor von Wortindizes mit Padding (Länge: max_satzelemente)
            
        Beispiel:
            >>> embedder.erhalte_indizes_von_satz("Der Fuchs springt")
            tensor([12, 5, 8, 0, 0, 0, ...])  # Mit Padding
        """
        if not satz:
            return torch.zeros(self._max_satzelemente, dtype=torch.long)
        
        # Satz tokenisieren
        woerter = self._tokenisiere_text(satz)
        
        if not woerter:
            return torch.zeros(self._max_satzelemente, dtype=torch.long)
        
        # Indizes sammeln
        indizes = []
        for wort in woerter:
            if wort in self._wort_zu_index:
                indizes.append(self._wort_zu_index[wort])
            else:
                # Unbekannte Wörter als 0 behandeln (kann als Padding interpretiert werden)
                indizes.append(0)
        
        # Auf maximale Länge beschränken
        indizes = indizes[:self._max_satzelemente]
        
        # Padding hinzufügen falls nötig
        if len(indizes) < self._max_satzelemente:
            padding = [0] * (self._max_satzelemente - len(indizes))
            indizes.extend(padding)
        
        return torch.tensor(indizes, dtype=torch.long)
    
    def erhalte_index_von_wort(self, wort: str) -> Tensor:
        """
        Verbesserte Methode zum Finden von Wortindizes.
        """
        wort_lower = wort.lower().strip()
        
        # 1. Direkter Treffer
        if wort_lower in self._wort_zu_index:
            return torch.tensor([self._wort_zu_index[wort_lower]], dtype=torch.long)
        
        # 2. Versuche mit einfacher Lemmatisierung
        grundformen = [
            wort_lower,
            wort_lower.rstrip('es'),
            wort_lower.rstrip('e'),
            wort_lower.rstrip('en'),
            wort_lower.rstrip('er'),
            wort_lower.rstrip('n'),
            wort_lower.rstrip('s')
        ]
        
        for form in grundformen:
            if form in self._wort_zu_index:
                print(f"  🔍 '{wort}' gefunden als '{form}'")
                return torch.tensor([self._wort_zu_index[form]], dtype=torch.long)
        
        # 3. Fallback: Suche nach Teilwörtern
        for vocab_wort, idx in self._wort_zu_index.items():
            if vocab_wort in wort_lower or wort_lower in vocab_wort:
                print(f"  🔍 Teilwort-Ersatz für '{wort}': '{vocab_wort}'")
                return torch.tensor([idx], dtype=torch.long)
        
        # 4. Endgültiger Fallback
        print(f"  ⚠️  Wort '{wort}' nicht im Vokabular gefunden")
        return torch.zeros(1, dtype=torch.long)
    
    def erhalte_embedding_matrix(self) -> np.ndarray:
        """
        Gibt die gesamte Embedding-Matrix zurück.
        
        Returns:
            NumPy Array der Form (woerterbuch_groesse, embedding_dimension)
        """
        with torch.no_grad():
            embeddings = self._skipgram_modell.wort_embeddings.weight.detach().numpy()
        
        return embeddings
    
    def berechne_wort_aehnlichkeit(self, wort1: str, wort2: str) -> float:
        """
        Berechnet die Kosinus-Ähnlichkeit zwischen zwei Wörtern.
        
        Args:
            wort1: Erstes Wort
            wort2: Zweites Wort
            
        Returns:
            Kosinus-Ähnlichkeit zwischen -1 und 1
            
        Beispiel:
            >>> embedder.berechne_wort_aehnlichkeit("König", "Königin")
            0.85
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        idx1 = self.erhalte_index_von_wort(wort1)
        idx2 = self.erhalte_index_von_wort(wort2)
        
        # Prüfen ob Wörter gefunden wurden
        if idx1.sum() == 0 or idx2.sum() == 0:
            return 0.0
        
        with torch.no_grad():
            embedding1 = self._skipgram_modell.wort_embeddings(idx1).detach().numpy()
            embedding2 = self._skipgram_modell.wort_embeddings(idx2).detach().numpy()
        
        aehnlichkeit = cosine_similarity(embedding1, embedding2)[0][0]
        return float(aehnlichkeit)
    
    def finde_semantische_nachbarn(self, 
                                  wort: str, 
                                  top_k: int = 10,
                                  min_aehnlichkeit: float = 0.0) -> List[Tuple[str, float]]:
        """
        Findet die semantisch ähnlichsten Wörter zu einem gegebenen Wort.
        
        Args:
            wort: Zielwort
            top_k: Anzahl der zurückzugebenden Nachbarn
            min_aehnlichkeit: Minimale Ähnlichkeit
            
        Returns:
            Liste von Tupeln (Nachbarwort, Ähnlichkeit), sortiert absteigend
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        idx = self.erhalte_index_von_wort(wort)
        
        if idx.sum() == 0:
            return []
        
        with torch.no_grad():
            ziel_embedding = self._skipgram_modell.wort_embeddings(idx).detach().numpy()
            alle_embeddings = self._skipgram_modell.wort_embeddings.weight.detach().numpy()
        
        # Ähnlichkeiten berechnen
        aehnlichkeiten = cosine_similarity(ziel_embedding, alle_embeddings)[0]
        
        # Ergebnisse sammeln und sortieren
        ergebnisse = []
        for i, aehnlichkeit in enumerate(aehnlichkeiten):
            if i != idx.item() and aehnlichkeit >= min_aehnlichkeit:
                wort_name = self._index_zu_wort.get(i, f"UNK_{i}")
                ergebnisse.append((wort_name, float(aehnlichkeit)))
        
        # Sortieren und begrenzen
        ergebnisse.sort(key=lambda x: x[1], reverse=True)
        return ergebnisse[:top_k]
    
    def visualisiere_embeddings(self, 
                              woerter: List[str],
                              methode: str = "pca",
                              titel: str = "Wort Embedding Visualisierung") -> None:
        """
        Visualisiert Wort-Embeddings mit PCA oder t-SNE.
        
        Args:
            woerter: Liste von zu visualisierenden Wörtern
            methode: 'pca' oder 'tsne'
            titel: Titel der Visualisierung
            
        Note:
            Erfordert matplotlib und sklearn
        """
        try:
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA
            from sklearn.manifold import TSNE
            
            if not woerter:
                print("⚠️  Keine Wörter zur Visualisierung")
                return
            
            # Embeddings für die gegebenen Wörter sammeln
            embeddings = []
            gueltige_woerter = []
            
            for wort in woerter:
                idx = self.erhalte_index_von_wort(wort)
                if idx.sum() > 0:
                    with torch.no_grad():
                        embedding = self._skipgram_modell.wort_embeddings(idx).detach().numpy()
                    embeddings.append(embedding.squeeze())
                    gueltige_woerter.append(wort)
            
            if len(gueltige_woerter) < 2:
                print("⚠️  Nicht genug gültige Wort-Embeddings gefunden")
                return
            
            embeddings_array = np.array(embeddings)
            
            # Dimensionalitätsreduktion
            if methode.lower() == "pca":
                reducer = PCA(n_components=2)
                reduced = reducer.fit_transform(embeddings_array)
            elif methode.lower() == "tsne":
                reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_array)-1))
                reduced = reducer.fit_transform(embeddings_array)
            else:
                print(f"⚠️  Unbekannte Methode: {methode}. Verwende PCA.")
                reducer = PCA(n_components=2)
                reduced = reducer.fit_transform(embeddings_array)
            
            # Plot erstellen
            plt.figure(figsize=(12, 10))
            plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6, s=100)
            
            # Wortlabels hinzufügen
            for i, wort in enumerate(gueltige_woerter):
                plt.annotate(wort, 
                           (reduced[i, 0], reduced[i, 1]),
                           fontsize=9,
                           alpha=0.8)
            
            plt.title(f"{titel} ({methode.upper()})")
            plt.xlabel(f"{methode.upper()} Komponente 1")
            plt.ylabel(f"{methode.upper()} Komponente 2")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            print(f"✅ {len(gueltige_woerter)} Wort-Embeddings visualisiert mit {methode.upper()}")
            
        except ImportError as e:
            print(f"❌ Visualisierung deaktiviert: {str(e)}")
            print("   Installieren Sie: pip install matplotlib scikit-learn")
        except Exception as e:
            print(f"⚠️  Fehler bei Embedding-Visualisierung: {str(e)}")
    
    def exportiere_embeddings(self, 
                            datei_pfad: str,
                            format: str = "txt") -> None:
        """
        Exportiert die trainierten Embeddings in verschiedene Formate.
        
        Args:
            datei_pfad: Pfad zur Ausgabedatei (ohne Endung)
            format: Exportformat ('txt', 'csv', 'json')
        """
        import json
        import csv
        
        if format.lower() == "txt":
            # Word2Vec kompatibles Format
            with open(f"{datei_pfad}.txt", 'w', encoding='utf-8') as f:
                f.write(f"{self._woerterbuch_groesse} {self._embedding_dimension}\n")
                
                with torch.no_grad():
                    for idx in range(self._woerterbuch_groesse):
                        wort = self._index_zu_wort[idx]
                        embedding = self._skipgram_modell.wort_embeddings(
                            torch.tensor([idx])
                        ).detach().numpy().squeeze()
                        
                        embedding_str = ' '.join(f"{val:.6f}" for val in embedding)
                        f.write(f"{wort} {embedding_str}\n")
            
            print(f"✅ Embeddings exportiert nach {datei_pfad}.txt")
            
        elif format.lower() == "csv":
            with open(f"{datei_pfad}.csv", 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Header schreiben
                header = ['wort'] + [f'dim_{i}' for i in range(self._embedding_dimension)]
                writer.writerow(header)
                
                with torch.no_grad():
                    for idx in range(self._woerterbuch_groesse):
                        wort = self._index_zu_wort[idx]
                        embedding = self._skipgram_modell.wort_embeddings(
                            torch.tensor([idx])
                        ).detach().numpy().squeeze()
                        
                        row = [wort] + embedding.tolist()
                        writer.writerow(row)
            
            print(f"✅ Embeddings exportiert nach {datei_pfad}.csv")
            
        elif format.lower() == "json":
            embeddings_dict = {}
            
            with torch.no_grad():
                for idx in range(self._woerterbuch_groesse):
                    wort = self._index_zu_wort[idx]
                    embedding = self._skipgram_modell.wort_embeddings(
                        torch.tensor([idx])
                    ).detach().numpy().squeeze().tolist()
                    
                    embeddings_dict[wort] = embedding
            
            with open(f"{datei_pfad}.json", 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': {
                        'vocab_size': self._woerterbuch_groesse,
                        'embedding_dim': self._embedding_dimension,
                        'window_size': self._kontext_fenster
                    },
                    'embeddings': embeddings_dict
                }, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Embeddings exportiert nach {datei_pfad}.json")
            
        else:
            print(f"⚠️  Unbekanntes Format: {format}. Verwende 'txt'.")
            self.exportiere_embeddings(datei_pfad, "txt")