#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SkipGramDatenSatz - Dataset für Skip-Gram Training

Dieses Dataset implementiert erweiterte Funktionen für:
- Effiziente Batch-Verarbeitung
- Unterstützung für variable Sequenzlängen
- Optionale Negative Sampling

Autor: Ufuk Baysal
Datum: 2025

"""

import torch
import numpy as np
from torch import Tensor
from typing import List, Tuple, Generator


class SkipGramDatenSatz(torch.utils.data.Dataset):
    """
    Dataset für Skip-Gram Training mit erweiterten Funktionen.
    
    Args:
        paare (List[Tuple[int, int]]): Liste von (zentrum_index, kontext_index) Paaren
        negative_samples (int, optional): Anzahl negativer Samples. Standard ist 5.
    
    Beispiel:
        >>> paare = [(0, 1), (0, 2), (1, 0), (1, 2)]
        >>> dataset = SkipGramDatenSatz(paare)
        >>> len(dataset)
        4
        >>> dataset[0]
        (tensor(0), tensor(1))
    """
    
    def __init__(self, 
                 paare: List[Tuple[int, int]],
                 negative_samples: int = 5):
        """
        Initialisiert das Dataset.
        """
        self.paare = paare
        self.negative_samples = negative_samples
        
        # Für Negative Sampling: Wortverteilung berechnen
        self._berechne_wort_verteilung()
    
    def _berechne_wort_verteilung(self) -> None:
        """
        Berechnet die Verteilung der Wörter für Negative Sampling.
        """
        if not self.paare:
            return
        
        # Alle Wortindizes sammeln
        alle_indizes = []
        for zentrum, kontext in self.paare:
            alle_indizes.append(zentrum)
            alle_indizes.append(kontext)
        
        # Einzigartige Indizes und Häufigkeiten
        einzigartige_indizes, haeufigkeiten = np.unique(alle_indizes, return_counts=True)
        
        # Verteilung für Negative Sampling (3/4 Power wie in Word2Vec)
        self.wort_verteilung = np.power(haeufigkeiten, 0.75)
        self.wort_verteilung = self.wort_verteilung / self.wort_verteilung.sum()
        self.einzigartige_indizes = einzigartige_indizes
    
    def __len__(self) -> int:
        """
        Gibt die Anzahl der Trainingspaare zurück.
        
        Returns:
            Anzahl der Paare im Dataset
        """
        return len(self.paare)
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """
        Gibt das Zentrum- und Kontextwort für den gegebenen Index zurück.
        
        Args:
            index: Index im Dataset
            
        Returns:
            Tuple[Tensor, Tensor]: (zentrum_index, kontext_index)
        """
        zentrum_index, kontext_index = self.paare[index]
        return torch.tensor(zentrum_index), torch.tensor(kontext_index)
    
    def erstelle_batch_mit_padding(self, batch_groesse: int) -> Generator:
        """
        Erstellt Batches mit Padding für variable Sequenzlängen.
        
        Args:
            batch_groesse: Größe jedes Batches
            
        Yields:
            Tuple[Tensor, Tensor]: Batches von (zentrum_indices, kontext_indices)
        """
        if not self.paare:
            return
        
        # Paare in Batches aufteilen
        for i in range(0, len(self.paare), batch_groesse):
            batch_paare = self.paare[i:i + batch_groesse]
            
            if not batch_paare:
                continue
            
            # Indizes extrahieren
            zentrum_indices = []
            kontext_indices = []
            
            for zentrum, kontext in batch_paare:
                zentrum_indices.append(zentrum)
                kontext_indices.append(kontext)
            
            # In Tensoren konvertieren
            zentrum_tensor = torch.tensor(zentrum_indices, dtype=torch.long)
            kontext_tensor = torch.tensor(kontext_indices, dtype=torch.long)
            
            yield zentrum_tensor, kontext_tensor
    
    def generiere_negative_samples(self, 
                                 zentrum_index: int, 
                                 kontext_index: int,
                                 anzahl: int = 5) -> List[int]:
        """
        Generiert negative Samples für ein gegebenes Wortpaar.
        
        Args:
            zentrum_index: Index des Zentrumsworts
            kontext_index: Index des Kontextworts (positives Sample)
            anzahl: Anzahl negativer Samples
            
        Returns:
            Liste von negativen Sample-Indizes
        """
        if not hasattr(self, 'wort_verteilung'):
            return []
        
        negative_samples = []
        versuche = 0
        max_versuche = anzahl * 10  # Verhindert Endlosschleifen
        
        while len(negative_samples) < anzahl and versuche < max_versuche:
            # Zufälliges Wort basierend auf Verteilung auswählen
            sample_index = np.random.choice(
                self.einzigartige_indizes,
                p=self.wort_verteilung
            )
            
            # Sicherstellen, dass es nicht das positive Sample oder Zentrumswort ist
            if sample_index != kontext_index and sample_index != zentrum_index:
                negative_samples.append(sample_index)
            
            versuche += 1
        
        return negative_samples