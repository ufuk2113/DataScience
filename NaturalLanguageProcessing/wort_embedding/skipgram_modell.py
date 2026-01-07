#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SkipGramModell - Verbessertes Skip-Gram Neural Network

Dieses Modell implementiert ein erweitertes Skip-Gram Modell mit:
- Zwei Embedding-Schichten (Input und Output)
- Option für Negative Sampling
- Embedding Normalisierung
- Erweiterte Initialisierungsoptionen

Autor: Ufuk Baysal
Datum: 2025

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math


class SkipGramModell(nn.Module):
    """
    Verbessertes Skip-Gram Modell mit zwei Embedding-Schichten.
    
    Args:
        woerterbuch_groesse (int): Größe des Vokabulars
        embedding_dimension (int): Dimension der Word Embeddings
        negative_samples (int, optional): Anzahl negativer Samples. Standard ist 5.
        embedding_init (str, optional): Initialisierungsmethode. Standard ist 'normal'.
    
    Beispiel:
        >>> model = SkipGramModell(vocab_size=1000, embedding_dimension=64)
        >>> zentrum = torch.tensor([2, 5, 10])
        >>> ausgabe = model(zentrum)
        >>> ausgabe.shape
        torch.Size([3, 1000])
    """
    
    def __init__(self, 
                 woerterbuch_groesse: int, 
                 embedding_dimension: int,
                 negative_samples: int = 5,
                 embedding_init: str = 'normal'):
        """
        Initialisiert das Skip-Gram Modell.
        """
        super().__init__()
        
        self.woerterbuch_groesse = woerterbuch_groesse
        self.embedding_dimension = embedding_dimension
        self.negative_samples = negative_samples
        
        # Zwei Embedding-Schichten: eine für Input, eine für Output
        self.wort_embeddings = nn.Embedding(woerterbuch_groesse, embedding_dimension)
        self.kontext_embeddings = nn.Embedding(woerterbuch_groesse, embedding_dimension)
        
        # Ausgabe-Layer
        self.ausgabe_schicht = nn.Linear(embedding_dimension, woerterbuch_groesse, bias=False)
        
        # Embeddings initialisieren
        self._initialisiere_embeddings(embedding_init)
        
        print(f"   SkipGramModell: {woerterbuch_groesse} Wörter, "
              f"{embedding_dimension} Dimensionen")
    
    def _initialisiere_embeddings(self, init_method: str) -> None:
        """
        Initialisiert die Embedding-Schichten.
        
        Args:
            init_method: Initialisierungsmethode ('normal', 'uniform', 'xavier')
        """
        if init_method == 'normal':
            # Normalverteilte Initialisierung (Word2Vec Standard)
            nn.init.normal_(self.wort_embeddings.weight, mean=0.0, std=0.01)
            nn.init.normal_(self.kontext_embeddings.weight, mean=0.0, std=0.01)
            
        elif init_method == 'uniform':
            # Gleichverteilte Initialisierung
            nn.init.uniform_(self.wort_embeddings.weight, a=-0.5, b=0.5)
            nn.init.uniform_(self.kontext_embeddings.weight, a=-0.5, b=0.5)
            
        elif init_method == 'xavier':
            # Xavier/Glorot Initialisierung
            nn.init.xavier_uniform_(self.wort_embeddings.weight)
            nn.init.xavier_uniform_(self.kontext_embeddings.weight)
            
        else:
            print(f"⚠️  Unbekannte Initialisierungsmethode: {init_method}. Verwende 'normal'.")
            nn.init.normal_(self.wort_embeddings.weight, mean=0.0, std=0.01)
            nn.init.normal_(self.kontext_embeddings.weight, mean=0.0, std=0.01)
        
        # Ausgabe-Layer initialisieren
        nn.init.zeros_(self.ausgabe_schicht.weight)
    
    def forward(self, zentrum_indices: Tensor) -> Tensor:
        """
        Führt den Forward Pass durch.
        
        Args:
            zentrum_indices (Tensor): Tensor mit Indizes der Zentrumswörter.
                                     Form: (batch_size,) oder (batch_size, seq_len)
        
        Returns:
            Tensor: Logits für jedes Wort im Vokabular.
                   Form: (batch_size, woerterbuch_groesse) oder
                         (batch_size * seq_len, woerterbuch_groesse)
        
        Beispiel:
            >>> model = SkipGramModell(1000, 64)
            >>> zentrum = torch.tensor([1, 2, 3])
            >>> ausgabe = model(zentrum)
            >>> ausgabe.shape
            torch.Size([3, 1000])
        """
        # Eingabeform überprüfen
        original_shape = zentrum_indices.shape
        
        # Flatten falls nötig (für Satzverarbeitung)
        if len(original_shape) > 1:
            zentrum_indices = zentrum_indices.view(-1)
        
        # Embeddings für Zentrumswörter erhalten
        zentrum_embeddings = self.wort_embeddings(zentrum_indices)
        
        # Durch Ausgabe-Layer propagieren
        logits = self.ausgabe_schicht(zentrum_embeddings)
        
        # Ursprüngliche Form wiederherstellen falls nötig
        if len(original_shape) > 1:
            batch_size, seq_len = original_shape
            logits = logits.view(batch_size, seq_len, -1)
        
        return logits
    
    def berechne_aehnlichkeits_matrix(self) -> Tensor:
        """
        Berechnet die Kosinus-Ähnlichkeitsmatrix aller Wortpaare.
        
        Returns:
            Tensor: Ähnlichkeitsmatrix der Form (woerterbuch_groesse, woerterbuch_groesse)
        """
        with torch.no_grad():
            # Normalisierte Embeddings berechnen
            wort_embeddings = F.normalize(self.wort_embeddings.weight, p=2, dim=1)
            kontext_embeddings = F.normalize(self.kontext_embeddings.weight, p=2, dim=1)
            
            # Durchschnittliche Ähnlichkeit berechnen
            aehnlichkeiten = torch.mm(wort_embeddings, kontext_embeddings.t())
            
            # Symmetrisch machen (optional)
            aehnlichkeiten = (aehnlichkeiten + aehnlichkeiten.t()) / 2
            
            return aehnlichkeiten
    
    def get_embedding(self, wort_index: int) -> Tensor:
        """
        Gibt das Embedding für einen bestimmten Wortindex zurück.
        
        Args:
            wort_index: Index des Wortes im Vokabular
            
        Returns:
            Tensor: Word Embedding der Form (embedding_dimension,)
        """
        with torch.no_grad():
            return self.wort_embeddings(torch.tensor([wort_index])).squeeze(0)
    
    def normalisiere_embeddings(self) -> None:
        """
        Normalisiert alle Embeddings auf Einheitsnorm.
        
        Dies verbessert die Stabilität und Vergleichbarkeit der Embeddings.
        """
        with torch.no_grad():
            # Wort-Embeddings normalisieren
            norm = torch.norm(self.wort_embeddings.weight, p=2, dim=1, keepdim=True)
            self.wort_embeddings.weight.data = self.wort_embeddings.weight.data / norm
            
            # Kontext-Embeddings normalisieren
            norm = torch.norm(self.kontext_embeddings.weight, p=2, dim=1, keepdim=True)
            self.kontext_embeddings.weight.data = self.kontext_embeddings.weight.data / norm
            
        print("✅ Embeddings normalisiert")